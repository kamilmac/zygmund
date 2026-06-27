//! Probe the generalized FM drum voice with the 3 default presets.
use fundsp::prelude64::*;

fn build_drum(v: [f32; 9], vel: f32) -> Box<dyn AudioUnit> {
    let base = (30.0 * 300.0_f32.powf(v[0])) as f64;
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
    let osc = ((carrier + (modf >> sine()) * midx) >> sine()) * env;
    Box::new(osc + (noise() >> highpass_hz(tone, 1.0)) * nenv)
}

fn peak(name: &str, v: [f32; 9]) {
    let mut u = build_drum(v, 0.9);
    u.set_sample_rate(48000.0);
    u.allocate();
    let (mut m, mut nan) = (0f32, false);
    for _ in 0..24000 {
        let x = u.get_mono();
        if !x.is_finite() { nan = true; }
        m = m.max(x.abs());
    }
    println!("{name:6} peak={m:.4} nan={nan}");
}

fn main() {
    peak("kick", [0.18, 0.10, 0.20, 0.70, 0.55, 0.65, 0.45, 0.10, 0.10]);
    peak("snare", [0.42, 0.25, 0.40, 0.60, 0.20, 0.70, 0.25, 0.70, 0.50]);
    peak("hihat", [0.72, 0.45, 0.60, 0.20, 0.00, 0.50, 0.12, 0.60, 0.80]);
}
