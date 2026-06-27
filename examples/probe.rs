//! Full-chain zygfred probe: stereo voice -> 2-ch sequencer -> drive/reverb/comp/limiter.
use fundsp::prelude64::*;

fn drum_mono(v: [f32; 9], vel: f32, pitch_mul: f64) -> An<impl AudioNode<Inputs = U0, Outputs = U1>> {
    let base = (30.0 * 300.0_f32.powf(v[0])) as f64 * pitch_mul;
    let ratio = (0.5 + v[1] * 7.5) as f64;
    let fm_idx = (v[2] * 8.0) as f64;
    let fmdec = (5.0 + v[3] * 70.0) as f64;
    let penv = (v[4] * 4.0) as f64;
    let pdec = (6.0 + v[5] * 80.0) as f64;
    let dec = (40.0 - 37.0 * v[6]) as f64;
    let snap = v[7] as f64;
    let tone = 200.0 * 50.0_f32.powf(v[8]);
    let amp = (vel * 0.8) as f64;
    let carrier = lfo(move |t| base * (1.0 + penv * (-t * pdec).exp()));
    let modf = lfo(move |t| base * ratio * (1.0 + penv * (-t * pdec).exp()));
    let midx = lfo(move |t| fm_idx * base * ratio * (-t * fmdec).exp());
    let env = lfo(move |t| amp * (-t * dec).exp());
    let nenv = lfo(move |t| amp * snap * (-t * dec).exp());
    ((carrier + (modf >> sine()) * midx) >> sine()) * env + (noise() >> highpass_hz(tone, 1.0)) * nenv
}

fn main() {
    let sr = 48000.0;
    let mut seq = Sequencer::new(0, 2, ReplayMode::None);
    seq.set_sample_rate(sr);
    let backend = seq.backend();
    let vol = shared(0.7); let drive = shared(0.0); let comp = shared(0.0);
    let mut net = Net::wrap(Box::new(backend));
    let dist = (pass()*3.0 >> shape(Tanh(2.0))) | (pass()*3.0 >> shape(Tanh(2.0)));
    net = net >> ((1.0 - var(&drive) >> follow(0.01) >> split::<U2>())*multipass::<U2>() & (var(&drive) >> follow(0.01) >> split::<U2>())*dist);
    let room = reverb2_stereo(14.0, 0.6, 0.5, 1.0, highshelf_hz(4000.0,1.0,db_amp(-2.0)));
    let rev = shared(0.0);
    net = net >> ((1.0 - var(&rev) >> follow(0.01) >> split::<U2>())*multipass::<U2>() & (var(&rev) >> follow(0.01) >> split::<U2>())*room);
    net = net >> ((var(&vol) >> follow(0.02) >> split::<U2>())*multipass::<U2>());
    let _ = comp;
    net = net >> limiter_stereo(0.003, 0.1);
    net.set_sample_rate(sr);
    // push a kick (stereo, haas 5ms, detune)
    let snd = [0.18f32,0.10,0.20,0.70,0.55,0.65,0.45,0.10,0.10];
    let l = drum_mono(snd, 0.9, 0.9988);
    let r = drum_mono(snd, 0.9, 1.0012) >> delay(0.005f32);
    let voice: Box<dyn AudioUnit> = Box::new(l | r);
    seq.push_relative(0.0, 0.8, Fade::Smooth, 0.001, 0.02, voice);
    let mut a = BlockRateAdapter::new(Box::new(net.backend()));
    let (mut ml, mut mr, mut nan) = (0f32,0f32,false);
    for _ in 0..(0.8*sr) as usize { let (l,r)=a.get_stereo(); if !l.is_finite()||!r.is_finite(){nan=true;} ml=ml.max(l.abs()); mr=mr.max(r.abs()); }
    println!("full-chain kick: L={ml:.3} R={mr:.3} nan={nan}");
}
