//! zygfred-web — the zygfred FM drum engine compiled to WebAssembly.
//!
//! DSP core duplicated from src/bin/zygfred.rs (voice + master chain); this crate adds a
//! numeric-only wasm-bindgen surface so it can run inside an AudioWorklet: trigger / set params /
//! process 128-frame blocks. No strings cross the boundary (AudioWorkletGlobalScope lacks
//! TextDecoder).

use fundsp::prelude64::*;
use wasm_bindgen::prelude::*;

pub const SCOPE_N: usize = 64;
pub const BLOCK: usize = 128; // AudioWorklet render quantum

// ---------- drum voice (generalized FM percussion) ----------

/// One mono FM percussion voice from the 9 sound params. `pitch_mul` detunes it (for L/R).
fn drum_mono(v: [f32; 9], vel: f32, pitch_mul: f64) -> An<impl AudioNode<Inputs = U0, Outputs = U1>> {
    let base = (30.0 * 300.0_f32.powf(v[0])) as f64 * pitch_mul; // 30..9000 Hz (exp)
    let ratio = (0.5 + v[1] * 7.5) as f64; // 0.5..8
    let fm_idx = (v[2] * 8.0) as f64; // modulation index
    let fmdec = (5.0 + v[3] * 70.0) as f64; // fm index decay rate
    let penv = (v[4] * 4.0) as f64; // pitch env amount (multiplier)
    let pdec = (6.0 + v[5] * 80.0) as f64; // pitch env decay rate
    let dec = (40.0 - 37.0 * v[6]) as f64; // amp decay rate (fast..slow)
    let snap = v[7] as f64; // noise amount
    let tone = 200.0 * 50.0_f32.powf(v[8]); // 200..10kHz noise highpass
    let amp = (vel * 0.8) as f64;

    let carrier = lfo(move |t| base * (1.0 + penv * (-t * pdec).exp()));
    let modf = lfo(move |t| base * ratio * (1.0 + penv * (-t * pdec).exp()));
    let midx = lfo(move |t| fm_idx * base * ratio * (-t * fmdec).exp());
    let env = lfo(move |t| amp * (-t * dec).exp());
    let nenv = lfo(move |t| amp * snap * (-t * dec).exp());
    let osc = ((carrier + (modf >> sine()) * midx) >> sine()) * env;
    let noise_part = (noise() >> highpass_hz(tone, 1.0)) * nenv;
    osc + noise_part
}

/// Stereo voice: independently-perturbed L/R takes, detuned apart, R Haas-delayed. 0 in, 2 out.
fn build_drum_stereo(
    vl: [f32; 9],
    vr: [f32; 9],
    vel: f32,
    mul_l: f64,
    mul_r: f64,
    haas: f64,
) -> Box<dyn AudioUnit> {
    let l = drum_mono(vl, vel, mul_l);
    if haas > 0.0005 {
        let r = drum_mono(vr, vel, mul_r) >> delay(haas as f32);
        Box::new(l | r)
    } else {
        // no Haas: skip the delay node entirely (a 0-length delay is a glitch hazard)
        let r = drum_mono(vr, vel, mul_r);
        Box::new(l | r)
    }
}

// ---------- master net ----------

/// Snappy feed-forward bus compressor (mono): envelope follower -> gain. Fast attack/release so
/// transients poke through and the body pumps. `amount` (0..1, live) lowers threshold, raises
/// ratio + makeup. Branch (^) sends the signal to both the passthrough and the detector.
fn compressor(amount: &Shared) -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    let sh = amount.clone();
    let detect = shape_fn(|x: f32| if x < 0.0 { -x } else { x }) >> afollow(0.002, 0.08);
    (pass() ^ detect)
        >> map(move |f: &Frame<f32, U2>| {
            let a = sh.value();
            let thresh = 0.5 - 0.4 * a;
            let ratio = 1.0 + a * 8.0;
            let makeup = 1.0 + a * 1.6;
            let env = if f[1] < 1e-6 { 1e-6 } else { f[1] };
            let g = if env > thresh {
                (thresh / env).powf(1.0 - 1.0 / ratio)
            } else {
                1.0
            };
            f[0] * g * makeup
        })
}

/// Master chain: reverb -> drive -> bits -> compressor -> volume (limiter last as safety).
fn build_net(
    seq_backend: Box<dyn AudioUnit>,
    drive: &Shared,
    reverb_amt: &Shared,
    quant: &Shared,
    comp: &Shared,
    volume: &Shared,
    sr: f64,
) -> Net {
    let mut net = Net::wrap(seq_backend);
    // room reverb (dry/wet)
    let room = reverb2_stereo(14.0, 0.6, 0.5, 1.0, highshelf_hz(4000.0, 1.0, db_amp(-2.0)));
    net = net
        >> ((1.0 - var(reverb_amt) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(reverb_amt) >> follow(0.01) >> split::<U2>()) * room);
    // drive (dry/wet hard tanh)
    let dist = (pass() * 3.0 >> shape(Tanh(2.0))) | (pass() * 3.0 >> shape(Tanh(2.0)));
    net = net
        >> ((1.0 - var(drive) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(drive) >> follow(0.01) >> split::<U2>()) * dist);
    // bit crusher (quant = live levels, 0 = off)
    fn crush(x: f32, levels: f32) -> f32 {
        if levels >= 1.0 { (x * levels).round() / levels } else { x }
    }
    let (q1, q2) = (quant.clone(), quant.clone());
    net = net
        >> (map(move |f: &Frame<f32, U1>| crush(f[0], q1.value()))
            | map(move |f: &Frame<f32, U1>| crush(f[0], q2.value())));
    net = net >> (compressor(comp) | compressor(comp)); // snappy bus comp
    net = net >> ((var(volume) >> follow(0.02) >> split::<U2>()) * multipass::<U2>());
    net = net >> limiter_stereo(0.003, 0.1);
    net.set_sample_rate(sr);
    net
}

// ---------- engine ----------

#[wasm_bindgen]
pub struct Engine {
    sequencer: Sequencer,
    _net: Net,
    backend: BlockRateAdapter,
    drive_sh: Shared,
    reverb_sh: Shared,
    comp_sh: Shared,
    vol_sh: Shared,
    quant_sh: Shared, // bit-crush levels (0 = off), read live by the in-net crusher
    buf_l: [f32; BLOCK],
    buf_r: [f32; BLOCK],
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Engine {
        let sr = sample_rate as f64;
        let mut sequencer = Sequencer::new(0, 2, ReplayMode::None);
        sequencer.set_sample_rate(sr);
        let seq_backend = sequencer.backend();

        let drive_sh = shared(0.0);
        let reverb_sh = shared(0.0);
        let comp_sh = shared(0.0);
        let vol_sh = shared(0.7);
        let quant_sh = shared(0.0);
        let mut net = build_net(
            Box::new(seq_backend),
            &drive_sh,
            &reverb_sh,
            &quant_sh,
            &comp_sh,
            &vol_sh,
            sr,
        );
        let backend = BlockRateAdapter::new(Box::new(net.backend()));

        Engine {
            sequencer,
            _net: net,
            backend,
            drive_sh,
            reverb_sh,
            comp_sh,
            vol_sh,
            quant_sh,
            buf_l: [0.0; BLOCK],
            buf_r: [0.0; BLOCK],
        }
    }

    pub fn set_drive(&mut self, v: f32) {
        self.drive_sh.set_value(v.clamp(0.0, 1.0));
    }

    pub fn set_reverb(&mut self, v: f32) {
        self.reverb_sh.set_value(v.clamp(0.0, 1.0));
    }

    pub fn set_comp(&mut self, v: f32) {
        self.comp_sh.set_value(v.clamp(0.0, 1.0));
    }

    pub fn set_volume(&mut self, v: f32) {
        self.vol_sh.set_value(v.clamp(0.0, 1.0));
    }

    /// Bit-crush quantization levels (0 = off, 2048 = 12-bit, ... 4 = 3-bit).
    pub fn set_bits_levels(&mut self, levels: f32) {
        self.quant_sh.set_value(levels);
    }

    /// Play one hit. The caller (UI thread) owns param state and the per-hit randomisation, so it
    /// passes the already-perturbed L/R takes plus the derived stereo/length values.
    pub fn trigger_voice(
        &mut self,
        vl: &[f32],
        vr: &[f32],
        vel: f32,
        mul_l: f64,
        mul_r: f64,
        haas: f64,
        len: f64,
    ) {
        if vl.len() < 9 || vr.len() < 9 {
            return;
        }
        let vl: [f32; 9] = vl[..9].try_into().unwrap();
        let vr: [f32; 9] = vr[..9].try_into().unwrap();
        let voice = build_drum_stereo(vl, vr, vel, mul_l, mul_r, haas);
        self.sequencer
            .push_relative(0.0, len, Fade::Smooth, 0.001, 0.02, voice);
    }

    /// Render `frames` (<= BLOCK) into the internal L/R buffers.
    pub fn process(&mut self, frames: usize) {
        let n = std::cmp::Ord::min(frames, BLOCK);
        for i in 0..n {
            let (l, r) = self.backend.get_stereo();
            self.buf_l[i] = l;
            self.buf_r[i] = r;
        }
    }

    pub fn left_ptr(&self) -> *const f32 {
        self.buf_l.as_ptr()
    }

    pub fn right_ptr(&self) -> *const f32 {
        self.buf_r.as_ptr()
    }
}

/// Render a dry voice offline and capture a short waveform window (peak-per-bin) for the scope.
/// Runs on the main thread in its own wasm instance — never touches the audio engine.
#[wasm_bindgen]
pub fn capture_scope(params: &[f32], sample_rate: f32) -> Vec<f32> {
    let mut snd = [0.0f32; 9];
    for i in 0..std::cmp::Ord::min(9, params.len()) {
        snd[i] = params[i];
    }
    let mut v: Box<dyn AudioUnit> = Box::new(drum_mono(snd, 0.95, 1.0));
    v.set_sample_rate(sample_rate as f64);
    v.allocate();
    let window = (0.05 * sample_rate) as usize; // ~50 ms
    let step = std::cmp::max(window / SCOPE_N, 1);
    let mut out = vec![0.0f32; SCOPE_N];
    let (mut oi, mut peak, mut cnt) = (0usize, 0.0f32, 0usize);
    for _ in 0..window {
        let s = v.get_mono();
        if s.abs() > peak.abs() {
            peak = s;
        }
        cnt += 1;
        if cnt >= step && oi < SCOPE_N {
            out[oi] = peak;
            oi += 1;
            peak = 0.0;
            cnt = 0;
        }
    }
    out
}
