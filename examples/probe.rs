//! Offline probe: render the synth graph headless and report peak amplitude,
//! to tell whether silence is in the DSP graph or the audio device path.
use fundsp::prelude64::*;

fn main() {
    let sr = 48000.0;

    // 1) bare voice — note the pre-arm tick at gate=0 before flipping to 1
    let gate = shared(0.0);
    let mut voice =
        (constant(440.0) >> saw()) * (var(&gate) >> adsr_live(0.02, 0.15, 0.6, 0.3)) * 0.5;
    voice.set_sample_rate(sr);
    voice.allocate();
    voice.get_mono(); // arm the adsr (observe gate<=0 once)
    gate.set_value(1.0); // next tick triggers attack
    let mut maxv = 0f32;
    for _ in 0..(sr as usize) {
        maxv = maxv.max(voice.get_mono().abs());
    }
    println!("bare voice peak  = {maxv:.4}");

    // 2) full path: sequencer -> net (pan, reverb, volume) -> block adapter
    let mut seq = Sequencer::new(0, 1, ReplayMode::None);
    seq.set_sample_rate(sr);
    let backend = seq.backend();
    let reverb_amt = shared(0.25);
    let volume = shared(0.5);
    let cutoff = shared(8000.0);
    let resonance = shared(0.2);
    let mut net = Net::wrap(Box::new(backend));
    net = net >> ((pass() | (var(&cutoff) >> follow(0.01)) | var(&resonance)) >> moog());
    net = net >> pan(0.0);
    let wet = reverb2_stereo(10.0, 2.0, 0.5, 1.0, highshelf_hz(5000.0, 1.0, db_amp(-1.0)));
    net = net
        >> ((1.0 - var(&reverb_amt) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(&reverb_amt) >> follow(0.01) >> split::<U2>()) * wet);
    net = net >> ((var(&volume) >> follow(0.02) >> split::<U2>()) * multipass::<U2>());
    net.set_sample_rate(sr);

    let g2 = shared(0.0);
    let mut v: Box<dyn AudioUnit> =
        Box::new((constant(440.0) >> saw()) * (var(&g2) >> adsr_live(0.02, 0.15, 0.6, 0.3)) * 0.5);
    v.allocate();
    v.get_mono();
    g2.set_value(1.0);
    seq.push_relative(0.0, f64::INFINITY, Fade::Smooth, 0.004, 0.01, v);

    let mut adapter = BlockRateAdapter::new(Box::new(net.backend()));
    let mut maxs = 0f32;
    for _ in 0..(sr as usize) {
        let (l, _r) = adapter.get_stereo();
        maxs = maxs.max(l.abs());
    }
    println!("full path peak   = {maxs:.4}");
}
