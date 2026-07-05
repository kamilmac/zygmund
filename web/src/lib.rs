//! zygfred-web — the zygfred FM drum engine compiled to WebAssembly.
//!
//! DSP core duplicated from src/bin/zygfred.rs (voice + master chain); this crate adds a
//! numeric-only wasm-bindgen surface so it can run inside an AudioWorklet: trigger / set params /
//! process 128-frame blocks. No strings cross the boundary (AudioWorkletGlobalScope lacks
//! TextDecoder).

use fundsp::prelude64::*;
use wasm_bindgen::prelude::*;

pub const BLOCK: usize = 128; // AudioWorklet render quantum

// ---------- drum voice (generalized FM percussion) ----------

/// One mono FM percussion voice from the 9 sound params.
fn drum_mono(v: [f32; 9], vel: f32) -> An<impl AudioNode<Inputs = U0, Outputs = U1>> {
    let base = (30.0 * 300.0_f32.powf(v[0])) as f64; // 30..9000 Hz (exp)
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

/// Stereo voice: independently-perturbed L/R takes, R Haas-delayed. 0 in, 2 out.
/// `width` 0..1 blends the right channel from the left take (mono) to the right take (stereo);
/// the left channel always carries the pure left take.
fn build_drum_stereo(
    vl: [f32; 9],
    vr: [f32; 9],
    vel: f32,
    haas: f64,
    width: f32,
) -> Box<dyn AudioUnit> {
    let l = drum_mono(vl, vel);
    let w = width.clamp(0.0, 1.0);
    let mix = move |f: &Frame<f32, U2>| (f[0], (1.0 - w) * f[0] + w * f[1]);
    if haas > 0.0005 {
        let r = drum_mono(vr, vel) >> delay(haas as f32);
        Box::new((l | r) >> map(mix))
    } else {
        // no Haas: skip the delay node entirely (a 0-length delay is a glitch hazard)
        let r = drum_mono(vr, vel);
        Box::new((l | r) >> map(mix))
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
        haas: f64,
        width: f32,
        len: f64,
    ) {
        if vl.len() < 9 || vr.len() < 9 {
            return;
        }
        let vl: [f32; 9] = vl[..9].try_into().unwrap();
        let vr: [f32; 9] = vr[..9].try_into().unwrap();
        let voice = build_drum_stereo(vl, vr, vel, haas, width);
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

/// Render a dry voice offline and return a log-frequency spectrogram: `rows * cols` values in
/// 0..1 (dB-mapped, -60 dB floor), row-major, row 0 = highest frequency. `seconds` sets the
/// analysis span. Runs on the main thread in its own wasm instance — never touches the engine.
#[wasm_bindgen]
pub fn capture_spectrogram(
    params: &[f32],
    sample_rate: f32,
    cols: usize,
    rows: usize,
    seconds: f32,
) -> Vec<f32> {
    use core::f32::consts::PI as PI32;
    const FFT: usize = 1024;
    let mut snd = [0.0f32; 9];
    for i in 0..std::cmp::Ord::min(9, params.len()) {
        snd[i] = params[i];
    }
    let mut v: Box<dyn AudioUnit> = Box::new(drum_mono(snd, 0.95));
    v.set_sample_rate(sample_rate as f64);
    v.allocate();
    let n = std::cmp::Ord::max((seconds * sample_rate) as usize, FFT + cols);
    let mut samples = vec![0.0f32; n];
    for s in samples.iter_mut() {
        *s = v.get_mono();
    }

    let hann: Vec<f32> = (0..FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * PI32 * i as f32 / (FFT as f32 - 1.0)).cos())
        .collect();
    let hop = std::cmp::Ord::max((n - FFT) / std::cmp::Ord::max(cols - 1, 1), 1);
    let bin_hz = sample_rate / FFT as f32;
    let f_min = 30.0f32;
    let f_max = (sample_rate / 2.0).min(12000.0);
    let mut out = vec![0.0f32; rows * cols];
    let mut re = vec![0.0f32; FFT];
    let mut im = vec![0.0f32; FFT];
    for c in 0..cols {
        let start = c * hop;
        if start + FFT > n {
            break;
        }
        for i in 0..FFT {
            re[i] = samples[start + i] * hann[i];
            im[i] = 0.0;
        }
        fft_inplace(&mut re, &mut im);
        for r in 0..rows {
            // log-frequency bands, top row = f_max
            let f_hi = f_min * (f_max / f_min).powf(1.0 - r as f32 / rows as f32);
            let f_lo = f_min * (f_max / f_min).powf(1.0 - (r as f32 + 1.0) / rows as f32);
            let b_lo = (f_lo / bin_hz) as usize;
            let b_hi = std::cmp::Ord::min(
                std::cmp::Ord::max((f_hi / bin_hz) as usize, b_lo + 1),
                FFT / 2,
            );
            let mut mag = 0.0f32;
            for b in b_lo..b_hi {
                let m = re[b] * re[b] + im[b] * im[b];
                if m > mag {
                    mag = m;
                }
            }
            // hann-windowed full-scale sine peaks near FFT/4 — normalize against that
            let db = 10.0 * (mag / ((FFT * FFT) as f32 / 16.0) + 1e-12).log10();
            out[r * cols + c] = ((db + 54.0) / 54.0).clamp(0.0, 1.0);
        }
    }
    out
}

/// Render a dry voice offline and return a min/max envelope: `2 * cols` values, [min, max] per
/// column, spanning `seconds`. Runs on the main thread in its own wasm instance.
#[wasm_bindgen]
pub fn capture_envelope(params: &[f32], sample_rate: f32, cols: usize, seconds: f32) -> Vec<f32> {
    let mut snd = [0.0f32; 9];
    for i in 0..std::cmp::Ord::min(9, params.len()) {
        snd[i] = params[i];
    }
    let mut v: Box<dyn AudioUnit> = Box::new(drum_mono(snd, 0.95));
    v.set_sample_rate(sample_rate as f64);
    v.allocate();
    let n = std::cmp::Ord::max((seconds * sample_rate) as usize, cols);
    let step = std::cmp::Ord::max(n / cols, 1);
    let mut out = vec![0.0f32; 2 * cols];
    for c in 0..cols {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for _ in 0..step {
            let s = v.get_mono();
            if s < lo {
                lo = s;
            }
            if s > hi {
                hi = s;
            }
        }
        out[2 * c] = lo;
        out[2 * c + 1] = hi;
    }
    out
}

/// Iterative radix-2 FFT, in place. Length must be a power of two.
fn fft_inplace(re: &mut [f32], im: &mut [f32]) {
    use core::f32::consts::PI as PI32;
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI32 / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (sr, si) = (re[i + k + len / 2], im[i + k + len / 2]);
                let (vr, vi) = (sr * cr - si * ci, sr * ci + si * cr);
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}
