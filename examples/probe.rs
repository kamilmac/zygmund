//! Stereo FM drum probe: confirm L != R (the per-channel difference works) and stays finite.
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

fn check(name: &str, snd: [f32; 9], det: f64, haas: f32) {
    // two slightly different takes (simulating the per-channel randomiser) + detune + Haas
    let mut vr = snd; vr[2] += 0.05; vr[7] += 0.05; // R nudged
    let l = drum_mono(snd, 0.9, 2f64.powf(-det / 2.0 / 1200.0));
    let r = drum_mono(vr, 0.9, 2f64.powf(det / 2.0 / 1200.0)) >> delay(haas);
    let mut v: Box<dyn AudioUnit> = Box::new(l | r);
    v.set_sample_rate(48000.0); v.allocate();
    let (mut ml, mut mr, mut diff, mut nan) = (0f32, 0f32, 0f32, false);
    for _ in 0..24000 {
        let (l, r) = v.get_stereo();
        if !l.is_finite() || !r.is_finite() { nan = true; }
        ml = ml.max(l.abs()); mr = mr.max(r.abs()); diff = diff.max((l - r).abs());
    }
    println!("{name:6} L={ml:.3} R={mr:.3} L-R_max={diff:.3} nan={nan}");
}

fn main() {
    check("kick",  [0.18,0.10,0.20,0.70,0.55,0.65,0.45,0.10,0.10], 4.0, 0.003);
    check("snare", [0.42,0.25,0.40,0.60,0.20,0.70,0.25,0.70,0.50], 12.0, 0.008);
    check("hihat", [0.72,0.45,0.60,0.20,0.00,0.50,0.12,0.60,0.80], 16.0, 0.012);
}
