//! Offline probe for the FM drum voices — renders each and reports peak + NaN.
use fundsp::prelude64::*;

fn kick() -> Box<dyn AudioUnit> {
    let (base, drop) = (55.0_f64, 150.0_f64);
    let carrier = lfo(move |t| base + drop * (-t * 40.0).exp());
    let modf = lfo(move |t| base + drop * (-t * 40.0).exp());
    let mi = lfo(move |t| 105.0 * (-t * 26.0).exp());
    let env = lfo(move |t| 0.9 * (-t * 10.0).exp());
    let click = lfo(move |t| 0.4 * (-t * 350.0).exp());
    Box::new(((carrier + (modf >> sine()) * mi) >> sine()) * env + noise() * click)
}

fn snare() -> Box<dyn AudioUnit> {
    let base = 210.0_f64;
    let cf = lfo(move |t| base + 90.0 * (-t * 55.0).exp());
    let mf = lfo(move |t| (base + 90.0 * (-t * 55.0).exp()) * 1.6);
    let mi = lfo(move |t| 125.0 * (-t * 45.0).exp());
    let benv = lfo(move |t| 0.56 * (-t * 30.0).exp());
    let nenv = lfo(move |t| 0.88 * (-t * 20.0).exp());
    let body = ((cf + (mf >> sine()) * mi) >> sine()) * benv;
    Box::new(body + (noise() >> highpass_hz(1400.0, 1.0)) * nenv)
}

fn hihat() -> Box<dyn AudioUnit> {
    let base = 6000.0f32;
    let fm_amt = 0.6f32 * 4000.0;
    let metallic = (constant(base) + (constant(base * 1.43) >> sine()) * fm_amt) >> sine();
    let env = lfo(move |t| 0.6 * (-t * 50.0).exp());
    Box::new((metallic * 0.45 + (noise() >> highpass_hz(6500.0, 1.0)) * 0.9) * env)
}

fn peak(name: &str, mut v: Box<dyn AudioUnit>, sr: f64, secs: f64) {
    v.set_sample_rate(sr);
    v.allocate();
    let (mut maxv, mut nan) = (0f32, false);
    for _ in 0..(secs * sr) as usize {
        let x = v.get_mono();
        if !x.is_finite() { nan = true; }
        maxv = maxv.max(x.abs());
    }
    println!("{name:6} peak = {maxv:.4}  nan={nan}");
}

fn main() {
    let sr = 48000.0;
    peak("kick", kick(), sr, 0.6);
    peak("snare", snare(), sr, 0.4);
    peak("hihat", hihat(), sr, 0.2);
}
