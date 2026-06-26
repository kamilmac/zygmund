//! tui-synth — a compact polyphonic ADSR synthesizer that plays from the terminal.
//!
//! Architecture (the load-bearing boundary):
//!   UI thread  --(fundsp Sequencer frontend: lock-free)-->  audio thread (cpal callback)
//! The UI thread owns all view + parameter state and pushes note events into the Sequencer.
//! The audio thread owns only DSP state and never allocates/locks/blocks.
//! Each note is a fresh voice that bakes the *current* ADSR; live knobs (reverb, volume)
//! ride atomic `Shared` values the audio graph reads each block.

use std::collections::HashMap;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use fundsp::prelude64::*;

use ratatui::crossterm::{
    event::{
        self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        canvas::{Canvas, Line as CanvasLine},
        Block, BorderType, Borders, Paragraph,
    },
    Frame, Terminal,
};

// ---------- palette ----------
const ACCENT: Color = Color::Rgb(125, 207, 255); // soft cyan-blue
const IDLE: Color = Color::Rgb(165, 170, 185);
const DIM: Color = Color::Rgb(95, 98, 120);
const BG: Color = Color::Rgb(18, 18, 26);

const VEL: f32 = 0.5;

/// Output bit-depth options: (label, quantization levels). 0.0 = off (full float).
/// levels = 2^(bits-1); the callback rounds each sample to `levels` steps.
const BIT_OPTIONS: [(&str, f32); 4] = [
    ("off", 0.0),
    ("16-bit", 32768.0),
    ("12-bit", 2048.0),
    ("8-bit", 128.0),
];

// ---------- waveform ----------
#[derive(Clone, Copy, PartialEq)]
enum Wave {
    Sine,
    Saw,
    Square,
    Triangle,
}
impl Wave {
    const ALL: [Wave; 4] = [Wave::Sine, Wave::Saw, Wave::Square, Wave::Triangle];
    fn label(self) -> &'static str {
        match self {
            Wave::Sine => "sine",
            Wave::Saw => "saw",
            Wave::Square => "square",
            Wave::Triangle => "triangle",
        }
    }
    fn step(self, d: i32) -> Wave {
        let i = Wave::ALL.iter().position(|w| *w == self).unwrap() as i32;
        let n = Wave::ALL.len() as i32;
        Wave::ALL[(((i + d) % n + n) % n) as usize]
    }
}

// ---------- selectable parameter ----------
#[derive(Clone, Copy, PartialEq)]
enum Param {
    Attack,
    Decay,
    Sustain,
    Release,
    Drift,
    Noise,
    Hiss,
    Cutoff,
    Resonance,
    Drive,
    RevAmount,
    RevRoom,
    RevTime,
    RevDiffusion,
    Volume,
    Comp,
    Bits,
    Chaos,
}
impl Param {
    const ALL: [Param; 18] = [
        Param::Attack,
        Param::Decay,
        Param::Sustain,
        Param::Release,
        Param::Drift,
        Param::Noise,
        Param::Hiss,
        Param::Cutoff,
        Param::Resonance,
        Param::Drive,
        Param::RevAmount,
        Param::RevRoom,
        Param::RevTime,
        Param::RevDiffusion,
        Param::Volume,
        Param::Comp,
        Param::Bits,
        Param::Chaos,
    ];
    fn step(self, d: i32) -> Param {
        let i = Param::ALL.iter().position(|p| *p == self).unwrap() as i32;
        let n = Param::ALL.len() as i32;
        Param::ALL[(((i + d) % n + n) % n) as usize]
    }
}

// ---------- DSP construction ----------

/// One polyphonic voice: drifting oscillator * ADSR envelope * velocity. Mono (0 in, 1 out).
/// `drift` is the analog-style pitch instability (fraction of pitch); each voice gets its own
/// `seed` so notes in a chord wander independently. ADSR + drift bake in at build time.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn build_voice(
    wave: Wave,
    hz: f32,
    vel: f32,
    gate: &Shared,
    a: f32,
    d: f32,
    s: f32,
    r: f32,
    drift: f32,
    noise_amt: f32,
    pitch_mod: &Shared,
    seed: u64,
) -> Box<dyn AudioUnit> {
    // Pitch = per-voice drift wander * a live global pitch-mod (chaos bends held notes in real time).
    // prelude64's lfo passes time as f64, so keep the wander math in f64.
    let (hz, drift) = (hz as f64, drift as f64);
    let pm = pitch_mod.clone();
    let pitch = lfo(move |t| hz * (1.0 + drift * spline_noise(seed, t * 3.0)) * pm.value() as f64);
    let env = var(gate) >> adsr_live(a, d, s, r);
    // oscillator + white noise, then shaped by the envelope and velocity
    match wave {
        Wave::Sine => Box::new(((pitch >> sine()) + noise() * noise_amt) * env * vel),
        Wave::Saw => Box::new(((pitch >> saw()) + noise() * noise_amt) * env * vel),
        Wave::Square => Box::new(((pitch >> square()) + noise() * noise_amt) * env * vel),
        Wave::Triangle => Box::new(((pitch >> triangle()) + noise() * noise_amt) * env * vel),
    }
}

/// Build a ready-to-trigger voice. `adsr_live` only attacks after it has seen the gate at <=0
/// once, so we tick it a single time at gate=0 to arm it, then raise the gate to 1.0 — the
/// attack then fires on the first tick inside the sequencer (no cross-thread race).
#[allow(clippy::too_many_arguments)]
fn make_voice(
    wave: Wave,
    hz: f32,
    a: f32,
    d: f32,
    s: f32,
    r: f32,
    drift: f32,
    noise_amt: f32,
    pitch_mod: &Shared,
    seed: u64,
) -> (Box<dyn AudioUnit>, Shared) {
    let gate = shared(0.0);
    let mut voice = build_voice(wave, hz, VEL, &gate, a, d, s, r, drift, noise_amt, pitch_mod, seed);
    voice.allocate();
    voice.get_mono(); // observe gate<=0 once
    gate.set_value(1.0);
    (voice, gate)
}

/// The reverb tail node (room/time/diffusion bake into its delay lines, so changing them means
/// rebuilding this node and crossfading it in).
fn create_reverb(room: f32, time: f32, diffusion: f32) -> Box<dyn AudioUnit> {
    Box::new(reverb2_stereo(
        room,
        time,
        diffusion,
        1.0,
        highshelf_hz(5000.0, 1.0, db_amp(-1.0)),
    ))
}

/// Wrap the sequencer backend into the master stereo chain: pan -> dry/wet reverb -> volume.
/// `reverb_amt` and `volume` are read live (smoothed); the reverb node is returned by id so its
/// room/time/diffusion can be crossfaded live.
/// Parallel mid-focused soft saturator (stereo): boost ~900 Hz into a tanh, then trim it back so
/// the harmonics land in the midrange. Blended dry/wet by `drive` (0 = clean).
fn saturator() -> An<impl AudioNode<Inputs = U2, Outputs = U2>> {
    let sat_l =
        bell_hz(900.0, 0.6, db_amp(6.0)) >> shape(Tanh(2.0)) >> bell_hz(900.0, 0.6, db_amp(-3.0));
    let sat_r =
        bell_hz(900.0, 0.6, db_amp(6.0)) >> shape(Tanh(2.0)) >> bell_hz(900.0, 0.6, db_amp(-3.0));
    sat_l | sat_r
}

#[allow(clippy::too_many_arguments)]
fn build_net(
    seq_backend: Box<dyn AudioUnit>,
    cutoff: &Shared,
    resonance: &Shared,
    drive: &Shared,
    reverb_amt: &Shared,
    volume: &Shared,
    comp: &Shared,
    hiss: &Shared,
    room: f32,
    time: f32,
    diffusion: f32,
    sr: f64,
) -> (Net, NodeId) {
    let mut net = Net::wrap(seq_backend);
    // Moog ladder lowpass on the mono bus — cutoff/resonance are live (cutoff smoothed).
    net = net >> ((pass() | (var(cutoff) >> follow(0.01)) | var(resonance)) >> moog());
    net = net >> pan(0.0); // mono -> stereo
    // Mid saturator, blended dry/wet by drive.
    net = net
        >> ((1.0 - var(drive) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(drive) >> follow(0.01) >> split::<U2>()) * saturator());
    let (reverb, reverb_id) = Net::wrap_id(create_reverb(room, time, diffusion));
    net = net
        >> ((1.0 - var(reverb_amt) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(reverb_amt) >> follow(0.01) >> split::<U2>()) * reverb);
    net = net >> ((var(volume) >> follow(0.02) >> split::<U2>()) * multipass::<U2>());
    // Constant analog-style noise floor (pink, independent of note level) added to the bus.
    net = net
        >> (multipass::<U2>()
            + ((pink() | pink()) * (var(hiss) >> follow(0.05) >> split::<U2>())));
    // Output compressor: drive into a lookahead limiter to gently squash peaks.
    net = net
        >> ((var(comp) >> follow(0.02) >> split::<U2>()) * multipass::<U2>())
        >> limiter_stereo(0.005, 0.1);
    net.set_sample_rate(sr);
    (net, reverb_id)
}

// ---------- keyboard -> note ----------

/// Computer-keyboard piano: home row = white keys, upper row = black keys. Returns a semitone
/// offset above the octave's C.
fn key_to_semitone(c: char) -> Option<i32> {
    Some(match c {
        'a' => 0,  // C
        'w' => 1,  // C#
        's' => 2,  // D
        'e' => 3,  // D#
        'd' => 4,  // E
        'f' => 5,  // F
        't' => 6,  // F#
        'g' => 7,  // G
        'y' => 8,  // G#
        'h' => 9,  // A
        'u' => 10, // A#
        'j' => 11, // B
        'k' => 12, // C (next octave)
        _ => return None,
    })
}

// ---------- presets ----------

/// A full snapshot of the synth's settings, saved to disk so slots survive restarts.
#[derive(Clone, Copy)]
struct Preset {
    wave: usize,
    octave: i32,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    drift: f32,
    noise: f32,
    cutoff: f32,
    resonance: f32,
    reverb_amt: f32,
    volume: f32,
    room: f32,
    time: f32,
    diffusion: f32,
    chaos: f32,
    drive: f32,
    comp: f32,
    hiss: f32,
    bits_idx: usize,
}

impl Preset {
    fn to_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.wave, self.octave, self.attack, self.decay, self.sustain, self.release,
            self.drift, self.noise, self.cutoff, self.resonance, self.reverb_amt, self.volume,
            self.room, self.time, self.diffusion, self.chaos, self.bits_idx, self.drive, self.comp,
            self.hiss,
        )
    }

    fn from_line(s: &str) -> Option<Preset> {
        let p: Vec<&str> = s.split(',').collect();
        if p.len() != 20 {
            return None;
        }
        Some(Preset {
            wave: p[0].parse().ok()?,
            octave: p[1].parse().ok()?,
            attack: p[2].parse().ok()?,
            decay: p[3].parse().ok()?,
            sustain: p[4].parse().ok()?,
            release: p[5].parse().ok()?,
            drift: p[6].parse().ok()?,
            noise: p[7].parse().ok()?,
            cutoff: p[8].parse().ok()?,
            resonance: p[9].parse().ok()?,
            reverb_amt: p[10].parse().ok()?,
            volume: p[11].parse().ok()?,
            room: p[12].parse().ok()?,
            time: p[13].parse().ok()?,
            diffusion: p[14].parse().ok()?,
            chaos: p[15].parse().ok()?,
            bits_idx: p[16].parse().ok()?,
            drive: p[17].parse().ok()?,
            comp: p[18].parse().ok()?,
            hiss: p[19].parse().ok()?,
        })
    }
}

fn presets_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".tui-synth-presets.txt")
}

fn load_presets() -> [Option<Preset>; 9] {
    let mut out = [None; 9];
    if let Ok(text) = std::fs::read_to_string(presets_path()) {
        for (i, line) in text.lines().take(9).enumerate() {
            if line != "-" {
                out[i] = Preset::from_line(line);
            }
        }
    }
    out
}

fn save_presets(presets: &[Option<Preset>; 9]) {
    let mut s = String::new();
    for p in presets {
        match p {
            Some(p) => s.push_str(&p.to_line()),
            None => s.push('-'),
        }
        s.push('\n');
    }
    let _ = std::fs::write(presets_path(), s);
}

// ---------- application state (UI thread) ----------
struct App {
    sequencer: Sequencer, // frontend; pushing notes is the lock-free bridge to audio
    net: Net,             // frontend; kept for live reverb crossfades + keeps the backend alive
    reverb_id: NodeId,
    // audio-thread mirrors of the live params (written each frame from the base values below)
    cutoff_sh: Shared,
    reso_sh: Shared,
    drive_sh: Shared,
    reverb_sh: Shared,
    vol_sh: Shared,
    comp_sh: Shared,
    hiss_sh: Shared,
    pitch_mod: Shared, // live global pitch multiplier (~1.0); chaos bends held notes through it
    quant: Shared,     // output bit-depth quantization levels (read in the audio callback)
    bits_idx: usize,
    // base (user-set) values — chaos perturbs these on the way to the shareds / new notes
    cutoff: f32,
    resonance: f32,
    reverb_amt: f32,
    volume: f32,
    room: f32,
    time: f32,
    diffusion: f32,
    wave: Wave,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    drift: f32,
    noise: f32,
    hiss: f32,  // constant background noise floor
    drive: f32, // mid saturator dry/wet (0 = clean)
    comp: f32,  // output compression amount (drive into the limiter)
    chaos: f32, // global instability: continuous wander + per-note variation
    octave: i32,
    selected: Param,
    active: HashMap<char, (EventId, Shared)>, // held notes -> (sequencer event, gate)
    seed: u64,                                // per-voice drift seed source
    rng: u64,                                 // per-note chaos RNG state
    clock: Instant,                           // for continuous wander
    presets: [Option<Preset>; 9],
    toast: Option<String>, // transient "saved/loaded preset N" notification
    toast_until: f32,
    supports_release: bool,
}

impl App {
    fn note_on(&mut self, key: char, semitone: i32) {
        if self.active.contains_key(&key) {
            return;
        }
        let (hz, a, d, s, r, drift, noise, seed) = self.voice_params(semitone);
        let (voice, gate) =
            make_voice(self.wave, hz, a, d, s, r, drift, noise, &self.pitch_mod, seed);
        let id = self
            .sequencer
            .push_relative(0.0, f64::INFINITY, Fade::Smooth, 0.004, 0.01, voice);
        self.active.insert(key, (id, gate));
    }

    fn note_off(&mut self, key: char) {
        if let Some((id, gate)) = self.active.remove(&key) {
            gate.set_value(-1.0); // start the ADSR release
            let tail = self.release as f64 + 0.1;
            self.sequencer.edit_relative(id, tail, 0.05); // remove voice after the tail
        }
    }

    /// Fallback for terminals without key-release reporting: a fixed-length note.
    fn play_fixed(&mut self, semitone: i32) {
        let (hz, a, d, s, r, drift, noise, seed) = self.voice_params(semitone);
        let (voice, _gate) =
            make_voice(self.wave, hz, a, d, s, r, drift, noise, &self.pitch_mod, seed);
        let len = (a + d + 0.4 + r) as f64;
        self.sequencer
            .push_relative(0.0, len, Fade::Smooth, 0.004, r as f64, voice);
    }

    /// Per-note voice parameters, with chaos applied as random per-note variation.
    /// Returns (hz, attack, decay, sustain, release, drift_amt, noise_amt, seed).
    #[allow(clippy::type_complexity)]
    fn voice_params(&mut self, semitone: i32) -> (f32, f32, f32, f32, f32, f32, f32, u64) {
        let c = self.chaos;
        let a = (self.attack * (1.0 + 0.3 * c * self.rnd())).clamp(0.001, 2.0);
        let d = (self.decay * (1.0 + 0.3 * c * self.rnd())).clamp(0.001, 2.0);
        let s = (self.sustain + 0.1 * c * self.rnd()).clamp(0.0, 1.0);
        let r = (self.release * (1.0 + 0.3 * c * self.rnd())).clamp(0.001, 3.0);
        let drift = (self.drift + 0.3 * c * self.rnd().abs()).clamp(0.0, 1.0) * 0.015;
        let noise = (self.noise + 0.2 * c * self.rnd().abs()).clamp(0.0, 1.0) * 0.5;
        let detune = 1.0 + 0.015 * c * self.rnd(); // up to ~+/-25 cents of broken tuning
        let hz = midi_hz((60 + semitone + 12 * self.octave) as f32) * detune;
        (hz, a, d, s, r, drift, noise, self.next_seed())
    }

    fn adjust(&mut self, d: i32) {
        let d = d as f32;
        match self.selected {
            Param::Attack => self.attack = (self.attack + d * 0.02).clamp(0.001, 2.0),
            Param::Decay => self.decay = (self.decay + d * 0.02).clamp(0.001, 2.0),
            Param::Sustain => self.sustain = (self.sustain + d * 0.05).clamp(0.0, 1.0),
            Param::Release => self.release = (self.release + d * 0.05).clamp(0.001, 3.0),
            Param::Drift => self.drift = (self.drift + d * 0.05).clamp(0.0, 1.0),
            Param::Noise => self.noise = (self.noise + d * 0.05).clamp(0.0, 1.0),
            Param::Hiss => self.hiss = (self.hiss + d * 0.05).clamp(0.0, 1.0),
            Param::Cutoff => {
                let factor = if d > 0.0 { 1.25 } else { 1.0 / 1.25 };
                self.cutoff = (self.cutoff * factor).clamp(20.0, 20000.0);
            }
            Param::Resonance => self.resonance = (self.resonance + d * 0.05).clamp(0.0, 0.98),
            Param::Drive => self.drive = (self.drive + d * 0.05).clamp(0.0, 1.0),
            Param::RevAmount => self.reverb_amt = (self.reverb_amt + d * 0.05).clamp(0.0, 1.0),
            Param::RevRoom => {
                self.room = (self.room + d).clamp(10.0, 30.0);
                self.rebuild_reverb();
            }
            Param::RevTime => {
                self.time = (self.time + d * 0.25).clamp(0.5, 10.0);
                self.rebuild_reverb();
            }
            Param::RevDiffusion => {
                self.diffusion = (self.diffusion + d * 0.05).clamp(0.0, 1.0);
                self.rebuild_reverb();
            }
            Param::Volume => self.volume = (self.volume + d * 0.05).clamp(0.0, 1.0),
            Param::Comp => self.comp = (self.comp + d * 0.05).clamp(0.0, 1.0),
            Param::Bits => {
                let n = BIT_OPTIONS.len() as i32;
                self.bits_idx = ((self.bits_idx as i32 + d as i32).rem_euclid(n)) as usize;
                self.quant.set_value(BIT_OPTIONS[self.bits_idx].1);
            }
            Param::Chaos => self.chaos = (self.chaos + d * 0.05).clamp(0.0, 1.0),
        }
    }

    /// Push base params (plus continuous chaos wander) to the audio-thread shareds. Called every
    /// UI frame. Each live param wanders on its own slow, seeded noise so they drift independently.
    fn tick_chaos(&mut self, t: f32) {
        let c = self.chaos;
        let w = |seed: u64, rate: f32| spline_noise::<f32>(seed, t * rate);
        let cutoff = (self.cutoff * (1.0 + 0.35 * c * w(0x01, 0.8))).clamp(20.0, 20000.0);
        let reso = (self.resonance + 0.15 * c * w(0x02, 0.5)).clamp(0.0, 0.98);
        let reverb = (self.reverb_amt + 0.15 * c * w(0x03, 0.3)).clamp(0.0, 1.0);
        let vol = (self.volume + 0.12 * c * w(0x04, 1.1)).clamp(0.0, 1.0);
        let drive = (self.drive + 0.1 * c * w(0x07, 0.6)).clamp(0.0, 1.0);
        // Pitch wander: a couple of detuned noise layers so held notes warble like sick tape.
        let pitch_mod = 1.0 + 0.02 * c * (0.7 * w(0x05, 1.3) + 0.3 * w(0x06, 4.0));
        self.cutoff_sh.set_value(cutoff);
        self.reso_sh.set_value(reso);
        self.drive_sh.set_value(drive);
        self.reverb_sh.set_value(reverb);
        self.vol_sh.set_value(vol);
        self.comp_sh.set_value(1.0 + 3.0 * self.comp); // pre-gain into the limiter
        self.hiss_sh.set_value(self.hiss * 0.03); // subtle noise floor
        self.pitch_mod.set_value(pitch_mod);
    }

    /// Crossfade in a freshly built reverb node (room/time/diffusion changed the delay structure).
    fn rebuild_reverb(&mut self) {
        let unit = create_reverb(self.room, self.time, self.diffusion);
        self.net.crossfade(self.reverb_id, Fade::Smooth, 0.5, unit);
        self.net.commit();
    }

    fn next_seed(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.seed
    }

    /// xorshift64 -> uniform in [-1, 1).
    fn rnd(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn capture(&self) -> Preset {
        Preset {
            wave: Wave::ALL.iter().position(|w| *w == self.wave).unwrap_or(0),
            octave: self.octave,
            attack: self.attack,
            decay: self.decay,
            sustain: self.sustain,
            release: self.release,
            drift: self.drift,
            noise: self.noise,
            cutoff: self.cutoff,
            resonance: self.resonance,
            reverb_amt: self.reverb_amt,
            volume: self.volume,
            room: self.room,
            time: self.time,
            diffusion: self.diffusion,
            chaos: self.chaos,
            drive: self.drive,
            comp: self.comp,
            hiss: self.hiss,
            bits_idx: self.bits_idx,
        }
    }

    fn apply(&mut self, p: Preset) {
        self.wave = Wave::ALL[std::cmp::min(p.wave, Wave::ALL.len() - 1)];
        self.octave = p.octave;
        self.attack = p.attack;
        self.decay = p.decay;
        self.sustain = p.sustain;
        self.release = p.release;
        self.drift = p.drift;
        self.noise = p.noise;
        self.cutoff = p.cutoff;
        self.resonance = p.resonance;
        self.reverb_amt = p.reverb_amt;
        self.volume = p.volume;
        self.room = p.room;
        self.time = p.time;
        self.diffusion = p.diffusion;
        self.chaos = p.chaos;
        self.drive = p.drive;
        self.comp = p.comp;
        self.hiss = p.hiss;
        self.bits_idx = std::cmp::min(p.bits_idx, BIT_OPTIONS.len() - 1);
        self.quant.set_value(BIT_OPTIONS[self.bits_idx].1);
        self.rebuild_reverb(); // room/time/diffusion may have changed
    }

    fn save_preset(&mut self, slot: usize) {
        self.presets[slot] = Some(self.capture());
        save_presets(&self.presets);
        self.set_toast(format!("✓ saved preset {}", slot + 1));
    }

    fn load_preset(&mut self, slot: usize) {
        match self.presets[slot] {
            Some(p) => {
                self.apply(p);
                self.set_toast(format!("→ loaded preset {}", slot + 1));
            }
            None => self.set_toast(format!("preset {} empty", slot + 1)),
        }
    }

    fn set_toast(&mut self, msg: String) {
        self.toast = Some(msg);
        self.toast_until = self.clock.elapsed().as_secs_f32() + 1.6;
    }
}

/// Returns true when the app should quit.
fn handle_key(app: &mut App, k: KeyEvent) -> bool {
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    match k.kind {
        KeyEventKind::Press => match k.code {
            KeyCode::Esc => return true,
            KeyCode::Char('c') if ctrl => return true,
            KeyCode::Tab => app.selected = app.selected.step(1),
            KeyCode::BackTab => app.selected = app.selected.step(-1),
            KeyCode::Up => app.adjust(1),
            KeyCode::Down => app.adjust(-1),
            KeyCode::Left => app.wave = app.wave.step(-1),
            KeyCode::Right => app.wave = app.wave.step(1),
            KeyCode::Char(',') => app.octave = std::cmp::max(app.octave - 1, -3),
            KeyCode::Char('.') => app.octave = std::cmp::min(app.octave + 1, 3),
            KeyCode::Char(c @ '1'..='9') => {
                let slot = c as usize - '1' as usize;
                if ctrl {
                    app.load_preset(slot);
                } else {
                    app.save_preset(slot);
                }
            }
            KeyCode::Char(c) => {
                if let Some(st) = key_to_semitone(c) {
                    if app.supports_release {
                        app.note_on(c, st);
                    } else {
                        app.play_fixed(st);
                    }
                }
            }
            _ => {}
        },
        KeyEventKind::Repeat => match k.code {
            KeyCode::Up => app.adjust(1),
            KeyCode::Down => app.adjust(-1),
            _ => {}
        },
        KeyEventKind::Release => {
            if let KeyCode::Char(c) = k.code {
                if key_to_semitone(c).is_some() {
                    app.note_off(c);
                }
            }
        }
    }
    false
}

// ---------- rendering ----------

fn bar(ratio: f32, width: usize) -> String {
    let filled = (ratio.clamp(0.0, 1.0) * width as f32).round() as usize;
    let mut s = String::with_capacity(width);
    for _ in 0..filled {
        s.push('█');
    }
    for _ in filled..width {
        s.push('░');
    }
    s
}

fn param_row(name: &str, value: String, ratio: f32, selected: bool) -> Line<'static> {
    let (label_style, val_style) = if selected {
        (
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            Style::default().fg(ACCENT),
        )
    } else {
        (Style::default().fg(IDLE), Style::default().fg(DIM))
    };
    let bar_style = Style::default().fg(if selected { ACCENT } else { DIM });
    Line::from(vec![
        Span::styled(format!(" {:<7}", name), label_style),
        Span::styled(bar(ratio, 9), bar_style),
        Span::styled(format!(" {}", value), val_style),
    ])
}

/// Build one styled keyboard row from a grid of (column, char, active) cells.
fn key_line(slots: &[(usize, char, bool)], width: usize) -> Line<'static> {
    let mut grid = vec![(' ', false); width];
    for &(col, ch, active) in slots {
        if col < width {
            grid[col] = (ch, active);
        }
    }
    let spans = grid
        .into_iter()
        .map(|(ch, active)| {
            let style = if ch == ' ' {
                Style::default()
            } else if active {
                Style::default()
                    .fg(BG)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(IDLE)
            };
            Span::styled(ch.to_string(), style)
        })
        .collect::<Vec<_>>();
    Line::from(spans)
}

fn ui(f: &mut Frame, app: &App) {
    let area = f.area();
    let mut title_spans = vec![
        Span::styled(
            " ♪ tui-synth ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("· {} voices ", app.active.len()),
            Style::default().fg(DIM),
        ),
    ];
    if let Some(t) = &app.toast {
        title_spans.push(Span::styled(
            format!("· {} ", t),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    }
    let title = Line::from(title_spans);
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(title)
        .title_alignment(Alignment::Center)
        .style(Style::default().bg(BG));
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // [0] waveform + octave
            Constraint::Length(1), // [1] spacer
            Constraint::Length(9), // [2] envelope curve + parameter columns
            Constraint::Length(1), // [3] spacer
            Constraint::Length(3), // [4] piano
            Constraint::Min(1),    // [5] footer
        ])
        .split(inner);

    // --- row 0: waveform selector + octave ---
    let head = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(10)])
        .split(rows[0]);
    let mut wave_spans = vec![Span::styled(" WAVE ", Style::default().fg(DIM))];
    for w in Wave::ALL {
        let st = if w == app.wave {
            Style::default()
                .fg(BG)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(IDLE)
        };
        wave_spans.push(Span::styled(format!(" {} ", w.label()), st));
        wave_spans.push(Span::raw(" "));
    }
    f.render_widget(Paragraph::new(Line::from(wave_spans)), head[0]);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("OCT {:+} ", app.octave),
            Style::default().fg(IDLE),
        )))
        .alignment(Alignment::Right),
        head[1],
    );

    // --- row 1: preset slots ---
    let mut preset_spans = vec![Span::styled(" PRESETS ", Style::default().fg(DIM))];
    for i in 0..9 {
        let st = if app.presets[i].is_some() {
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(DIM)
        };
        preset_spans.push(Span::styled(format!("{} ", i + 1), st));
    }
    preset_spans.push(Span::styled("  tap save · Ctrl+# load", Style::default().fg(DIM)));
    f.render_widget(Paragraph::new(Line::from(preset_spans)), rows[1]);

    // --- row 2: envelope curve | osc+filter column | fx column ---
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(28),
            Constraint::Percentage(38),
            Constraint::Percentage(34),
        ])
        .split(rows[2]);

    // envelope curve
    let (a, d, s, r) = (app.attack, app.decay, app.sustain, app.release);
    let hold = 0.35f32;
    let total = (a + d + hold + r).max(0.0001);
    let x1 = (a / total) as f64;
    let x2 = ((a + d) / total) as f64;
    let x3 = ((a + d + hold) / total) as f64;
    let sy = s as f64;
    let env = Canvas::default()
        .block(Block::default().title(Span::styled(" envelope", Style::default().fg(DIM))))
        .background_color(BG)
        .marker(Marker::Braille)
        .x_bounds([0.0, 1.0])
        .y_bounds([0.0, 1.05])
        .paint(move |ctx| {
            let seg = |x1, y1, x2, y2| CanvasLine {
                x1,
                y1,
                x2,
                y2,
                color: ACCENT,
            };
            ctx.draw(&seg(0.0, 0.0, x1, 1.0)); // attack
            ctx.draw(&seg(x1, 1.0, x2, sy)); // decay
            ctx.draw(&seg(x2, sy, x3, sy)); // sustain
            ctx.draw(&seg(x3, sy, 1.0, 0.0)); // release
        });
    f.render_widget(env, mid[0]);

    // column A — envelope + oscillator + filter
    let secs = |v: f32| format!("{:.0}ms", v * 1000.0);
    let sel = |p: Param| app.selected == p;
    let cutoff = app.cutoff;
    let res = app.resonance;
    let cutoff_ratio = (cutoff.max(20.0).ln() - 20f32.ln()) / (20000f32.ln() - 20f32.ln());
    let col_a = vec![
        param_row("Attack", secs(a), a / 2.0, sel(Param::Attack)),
        param_row("Decay", secs(d), d / 2.0, sel(Param::Decay)),
        param_row("Sustain", format!("{:.0}%", s * 100.0), s, sel(Param::Sustain)),
        param_row("Release", secs(r), r / 3.0, sel(Param::Release)),
        param_row("Drift", format!("{:.0}%", app.drift * 100.0), app.drift, sel(Param::Drift)),
        param_row("Noise", format!("{:.0}%", app.noise * 100.0), app.noise, sel(Param::Noise)),
        param_row("Hiss", format!("{:.0}%", app.hiss * 100.0), app.hiss, sel(Param::Hiss)),
        param_row("Cutoff", format!("{:.0}Hz", cutoff), cutoff_ratio, sel(Param::Cutoff)),
        param_row("Reso", format!("{:.0}%", res / 0.98 * 100.0), res / 0.98, sel(Param::Resonance)),
    ];
    f.render_widget(Paragraph::new(col_a), mid[1]);

    // column B — saturator + reverb + output + chaos
    let rv = app.reverb_amt;
    let vol = app.volume;
    let col_b = vec![
        param_row("Drive", format!("{:.0}%", app.drive * 100.0), app.drive, sel(Param::Drive)),
        param_row("Reverb", format!("{:.0}%", rv * 100.0), rv, sel(Param::RevAmount)),
        param_row("Room", format!("{:.0}m", app.room), (app.room - 10.0) / 20.0, sel(Param::RevRoom)),
        param_row("Time", format!("{:.2}s", app.time), app.time / 10.0, sel(Param::RevTime)),
        param_row("Diffuse", format!("{:.0}%", app.diffusion * 100.0), app.diffusion, sel(Param::RevDiffusion)),
        param_row("Volume", format!("{:.0}%", vol * 100.0), vol, sel(Param::Volume)),
        param_row("Comp", format!("{:.0}%", app.comp * 100.0), app.comp, sel(Param::Comp)),
        param_row(
            "Bits",
            BIT_OPTIONS[app.bits_idx].0.to_string(),
            app.bits_idx as f32 / (BIT_OPTIONS.len() - 1) as f32,
            sel(Param::Bits),
        ),
        param_row("Chaos", format!("{:.0}%", app.chaos * 100.0), app.chaos, sel(Param::Chaos)),
    ];
    f.render_widget(Paragraph::new(col_b), mid[2]);

    // --- piano ---
    let act = |c: char| app.active.contains_key(&c);
    let whites = [
        (2usize, 'a', 'C'),
        (6, 's', 'D'),
        (10, 'd', 'E'),
        (14, 'f', 'F'),
        (18, 'g', 'G'),
        (22, 'h', 'A'),
        (26, 'j', 'B'),
        (30, 'k', 'C'),
    ];
    let blacks = [(4usize, 'w'), (8, 'e'), (16, 't'), (20, 'y'), (24, 'u')];
    let width = 33;
    let black_slots: Vec<_> = blacks.iter().map(|&(c, ch)| (c, ch, act(ch))).collect();
    let white_slots: Vec<_> = whites.iter().map(|&(c, ch, _)| (c, ch, act(ch))).collect();
    let note_slots: Vec<_> = whites.iter().map(|&(c, _, n)| (c, n, false)).collect();
    let piano = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(rows[4]);
    f.render_widget(Paragraph::new(key_line(&black_slots, width)), piano[0]);
    f.render_widget(Paragraph::new(key_line(&white_slots, width)), piano[1]);
    f.render_widget(Paragraph::new(key_line(&note_slots, width)), piano[2]);

    // --- footer ---
    let mut hint = String::from(
        "notes a–k  ·  Tab param  ·  ↑↓ adjust  ·  ←→ wave  ·  , . octave  ·  Esc quit",
    );
    if !app.supports_release {
        hint.push_str("   (no key-release: fixed-length notes)");
    }
    f.render_widget(
        Paragraph::new(Span::styled(hint, Style::default().fg(DIM))).alignment(Alignment::Center),
        rows[5],
    );
}

// ---------- audio plumbing ----------

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut backend: BlockRateAdapter,
    quant: Shared,
) -> Result<cpal::Stream, Box<dyn std::error::Error>>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let channels = config.channels as usize;
    let mut next = move || backend.get_stereo();
    // Bitcrush: round each sample to `levels` steps (0 = off). Cheap atomic read per frame.
    let crush = move |x: f32, levels: f32| {
        if levels >= 1.0 {
            (x * levels).round() / levels
        } else {
            x
        }
    };
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let levels = quant.value();
            for frame in data.chunks_mut(channels) {
                let (l, r) = next();
                let (l, r) = (crush(l, levels), crush(r, levels));
                for (i, sample) in frame.iter_mut().enumerate() {
                    *sample = if i & 1 == 0 {
                        T::from_sample(l)
                    } else {
                        T::from_sample(r)
                    };
                }
            }
        },
        |err| eprintln!("stream error: {err}"),
        None,
    )?;
    Ok(stream)
}

fn restore_terminal(supports: bool) {
    let mut out = io::stdout();
    if supports {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    let _ = execute!(out, LeaveAlternateScreen);
    let _ = disable_raw_mode();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- audio setup ---
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = device.default_output_config()?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let sample_rate = config.sample_rate as f64;

    let mut sequencer = Sequencer::new(0, 1, ReplayMode::None);
    sequencer.set_sample_rate(sample_rate); // tune pushed voices to the device rate
    let seq_backend = sequencer.backend();

    let cutoff_sh = shared(8000.0);
    let reso_sh = shared(0.2);
    let drive_sh = shared(0.0);
    let reverb_sh = shared(0.25);
    let vol_sh = shared(0.5);
    let comp_sh = shared(1.0);
    let hiss_sh = shared(0.0);
    let pitch_mod = shared(1.0);
    let quant = shared(0.0); // bit-depth off by default
    let (room, time, diffusion) = (12.0_f32, 2.0_f32, 0.5_f32);
    let (mut net, reverb_id) = build_net(
        Box::new(seq_backend),
        &cutoff_sh,
        &reso_sh,
        &drive_sh,
        &reverb_sh,
        &vol_sh,
        &comp_sh,
        &hiss_sh,
        room,
        time,
        diffusion,
        sample_rate,
    );
    let backend = BlockRateAdapter::new(Box::new(net.backend()));

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, backend, quant.clone())?,
        cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, backend, quant.clone())?,
        cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config, backend, quant.clone())?,
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    };
    stream.play()?;

    // --- terminal setup (Kitty keyboard protocol for true note-off) ---
    let supports = supports_keyboard_enhancement().unwrap_or(false);
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    if supports {
        execute!(
            stdout,
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                    | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
            )
        )?;
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal(supports);
        default_hook(info);
    }));

    let mut terminal: Terminal<CrosstermBackend<Stdout>> =
        Terminal::new(CrosstermBackend::new(stdout))?;

    let mut app = App {
        sequencer,
        net,
        reverb_id,
        cutoff_sh,
        reso_sh,
        drive_sh,
        reverb_sh,
        vol_sh,
        comp_sh,
        hiss_sh,
        pitch_mod,
        quant,
        bits_idx: 0,
        cutoff: 8000.0,
        resonance: 0.2,
        reverb_amt: 0.25,
        volume: 0.5,
        room,
        time,
        diffusion,
        wave: Wave::Saw,
        attack: 0.02,
        decay: 0.15,
        sustain: 0.6,
        release: 0.3,
        drift: 0.3,
        noise: 0.0,
        hiss: 0.0,
        drive: 0.0,
        comp: 0.0,
        chaos: 0.0,
        octave: 0,
        selected: Param::Attack,
        active: HashMap::new(),
        seed: 0,
        rng: 0x853c_49e6_748f_ea9b,
        clock: Instant::now(),
        presets: load_presets(),
        toast: None,
        toast_until: 0.0,
        supports_release: supports,
    };

    // --- event loop ---
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let t = app.clock.elapsed().as_secs_f32();
            app.tick_chaos(t);
            if app.toast.is_some() && t > app.toast_until {
                app.toast = None;
            }
            terminal.draw(|f| ui(f, &app))?;
            if event::poll(Duration::from_millis(16))? {
                if let Event::Key(k) = event::read()? {
                    if handle_key(&mut app, k) {
                        break;
                    }
                }
            }
        }
        Ok(())
    })();

    restore_terminal(supports);
    result
}
