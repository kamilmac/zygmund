//! zygmunt — a compact polyphonic ADSR synthesizer that plays from the terminal.
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
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use fundsp::prelude64::*;

use midir::{Ignore, MidiInput, MidiInputConnection};

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
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame, Terminal,
};

// ---------- palette ----------
const ACCENT: Color = Color::Rgb(125, 207, 255); // soft cyan-blue
const IDLE: Color = Color::Rgb(165, 170, 185);
const DIM: Color = Color::Rgb(95, 98, 120);
const BG: Color = Color::Rgb(18, 18, 26);

/// Per-voice gain. Voices sum in the sequencer, so this is kept low to leave headroom for chords
/// (the master limiter then only needs to catch the occasional transient, not constant overshoot).
const VEL: f32 = 0.22;

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
    Detune,
    Noise,
    Hiss,
    Cutoff,
    Resonance,
    FEnv,
    FDecay,
    Drive,
    Delay,
    DFeed,
    RevAmount,
    RevRoom,
    RevTime,
    RevDiffusion,
    Eq1k,
    Volume,
    Comp,
    Bits,
    Chaos,
}
impl Param {
    const ALL: [Param; 24] = [
        Param::Attack,
        Param::Decay,
        Param::Sustain,
        Param::Release,
        Param::Drift,
        Param::Detune,
        Param::Noise,
        Param::Hiss,
        Param::Cutoff,
        Param::Resonance,
        Param::FEnv,
        Param::FDecay,
        Param::Drive,
        Param::Delay,
        Param::DFeed,
        Param::RevAmount,
        Param::RevRoom,
        Param::RevTime,
        Param::RevDiffusion,
        Param::Eq1k,
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
    fn name(self) -> &'static str {
        match self {
            Param::Attack => "Attack",
            Param::Decay => "Decay",
            Param::Sustain => "Sustain",
            Param::Release => "Release",
            Param::Drift => "Drift",
            Param::Detune => "Detune",
            Param::Noise => "Noise",
            Param::Hiss => "Hiss",
            Param::Cutoff => "Cutoff",
            Param::Resonance => "Reso",
            Param::FEnv => "F.Env",
            Param::FDecay => "F.Decay",
            Param::Drive => "Drive",
            Param::Delay => "Delay",
            Param::DFeed => "D.Feed",
            Param::RevAmount => "Reverb",
            Param::RevRoom => "Room",
            Param::RevTime => "Time",
            Param::RevDiffusion => "Diffuse",
            Param::Eq1k => "EQ 1k",
            Param::Volume => "Volume",
            Param::Comp => "Comp",
            Param::Bits => "Bits",
            Param::Chaos => "Chaos",
        }
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
    detune: f32,
    noise_amt: f32,
    pitch_mod: &Shared,
    seed: u64,
) -> Box<dyn AudioUnit> {
    // Three detuned oscillators per voice, each wandering on its own drift noise (analog-style
    // unison: the copies beat against each other so the tone moves instead of sitting still).
    // Pitch = drift wander * global pitch-mod (chaos bends held notes live). prelude64 lfo time is f64.
    let (hz, drift, det) = (hz as f64, drift as f64, detune as f64 * 0.02);
    let (pm0, pm1, pm2) = (pitch_mod.clone(), pitch_mod.clone(), pitch_mod.clone());
    // All three share the same drift wander (same seed/rate) so Detune is the ONLY thing that
    // spreads them. At Detune 0 they collapse onto one frequency -> a single oscillator.
    let p0 = lfo(move |t| hz * (1.0 + drift * spline_noise(seed, t * 3.0)) * pm0.value() as f64);
    let p1 = lfo(move |t| {
        hz * (1.0 + det) * (1.0 + drift * spline_noise(seed, t * 3.0)) * pm1.value() as f64
    });
    let p2 = lfo(move |t| {
        hz * (1.0 - det) * (1.0 + drift * spline_noise(seed, t * 3.0)) * pm2.value() as f64
    });
    let env = var(gate) >> adsr_live(a, d, s, r);
    // unison osc sum (normalised) + white noise, shaped by the envelope and velocity
    match wave {
        Wave::Sine => Box::new(
            (((p0 >> sine()) + (p1 >> sine()) + (p2 >> sine())) * 0.33 + noise() * noise_amt)
                * env
                * vel,
        ),
        Wave::Saw => Box::new(
            (((p0 >> saw()) + (p1 >> saw()) + (p2 >> saw())) * 0.33 + noise() * noise_amt)
                * env
                * vel,
        ),
        Wave::Square => Box::new(
            (((p0 >> square()) + (p1 >> square()) + (p2 >> square())) * 0.33 + noise() * noise_amt)
                * env
                * vel,
        ),
        Wave::Triangle => Box::new(
            (((p0 >> triangle()) + (p1 >> triangle()) + (p2 >> triangle())) * 0.33
                + noise() * noise_amt)
                * env
                * vel,
        ),
    }
}

/// Build a ready-to-trigger voice. `adsr_live` only attacks after it has seen the gate at <=0
/// once, so we tick it a single time at gate=0 to arm it, then raise the gate to 1.0 — the
/// attack then fires on the first tick inside the sequencer (no cross-thread race).
#[allow(clippy::too_many_arguments)]
fn make_voice(
    wave: Wave,
    hz: f32,
    vel: f32,
    a: f32,
    d: f32,
    s: f32,
    r: f32,
    drift: f32,
    detune: f32,
    noise_amt: f32,
    pitch_mod: &Shared,
    seed: u64,
) -> (Box<dyn AudioUnit>, Shared) {
    let gate = shared(0.0);
    let mut voice =
        build_voice(wave, hz, vel, &gate, a, d, s, r, drift, detune, noise_amt, pitch_mod, seed);
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
/// A deliberately strange stereo delay. Each channel's delay time warbles on its own slow random
/// LFO (echoes drift in pitch), the two channels run at an odd time ratio (0.33 s vs 0.49 s — a
/// lopsided, non-rhythmic ping-pong), and the feedback path swaps L/R and darkens each repeat.
/// `dfeed` sets the feedback (number of repeats).
fn strange_delay(dfeed: &Shared) -> An<impl AudioNode<Inputs = U2, Outputs = U2>> {
    let delay_l = (pass()
        | lfo(|t: f64| 0.33 * (1.0 + 0.06 * spline_noise::<f64>(11, t * 0.7))))
        >> tap_linear(0.02, 1.2);
    let delay_r = (pass()
        | lfo(|t: f64| 0.49 * (1.0 + 0.06 * spline_noise::<f64>(22, t * 0.5))))
        >> tap_linear(0.02, 1.2);
    feedback(
        reverse::<U2>()
            >> (delay_l | delay_r)
            >> (lowpass_hz(2400.0, 1.0) | lowpass_hz(2400.0, 1.0))
            >> ((var(dfeed) >> follow(0.05) >> split::<U2>()) * multipass::<U2>()),
    )
}

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
    delay_mix: &Shared,
    dfeed: &Shared,
    eq1k: &Shared,
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
    // Strange delay (before the reverb), blended dry/wet by delay_mix.
    net = net
        >> ((1.0 - var(delay_mix) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(delay_mix) >> follow(0.01) >> split::<U2>()) * strange_delay(dfeed));
    let (reverb, reverb_id) = Net::wrap_id(create_reverb(room, time, diffusion));
    net = net
        >> ((1.0 - var(reverb_amt) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(reverb_amt) >> follow(0.01) >> split::<U2>()) * reverb);
    // 1 kHz EQ dip: subtract a band from the dry signal (eq1k = depth). More bands can be added later.
    net = net
        >> (multipass::<U2>()
            & ((-1.5 * var(eq1k) >> follow(0.02) >> split::<U2>())
                * (bandpass_hz(1000.0, 1.0) | bandpass_hz(1000.0, 1.0))));
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
    fenv: f32,
    fdecay: f32,
    delay: f32,
    dfeed: f32,
    detune: f32,
    eq1k: f32,
    bits_idx: usize,
}

impl Preset {
    fn to_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.wave, self.octave, self.attack, self.decay, self.sustain, self.release,
            self.drift, self.noise, self.cutoff, self.resonance, self.reverb_amt, self.volume,
            self.room, self.time, self.diffusion, self.chaos, self.bits_idx, self.drive, self.comp,
            self.hiss, self.fenv, self.fdecay, self.delay, self.dfeed, self.detune, self.eq1k,
        )
    }

    fn from_line(s: &str) -> Option<Preset> {
        let p: Vec<&str> = s.split(',').collect();
        if p.len() != 26 {
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
            fenv: p[20].parse().ok()?,
            fdecay: p[21].parse().ok()?,
            delay: p[22].parse().ok()?,
            dfeed: p[23].parse().ok()?,
            detune: p[24].parse().ok()?,
            eq1k: p[25].parse().ok()?,
        })
    }
}

fn presets_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".zygmunt-presets.txt")
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

// ---------- MIDI ----------

enum MidiMsg {
    NoteOn { ch: u8, note: u8, vel: u8 },
    NoteOff { ch: u8, note: u8 },
    Cc { ch: u8, cc: u8, value: u8 },
}

fn parse_midi(bytes: &[u8]) -> Option<MidiMsg> {
    if bytes.len() < 3 {
        return None;
    }
    let ch = bytes[0] & 0x0F;
    match bytes[0] & 0xF0 {
        0x90 => Some(MidiMsg::NoteOn { ch, note: bytes[1], vel: bytes[2] }),
        0x80 => Some(MidiMsg::NoteOff { ch, note: bytes[1] }),
        0xB0 => Some(MidiMsg::Cc { ch, cc: bytes[1], value: bytes[2] }),
        _ => None,
    }
}

/// Connect to the first available MIDI input (preferring a Digitakt/Elektron port). The returned
/// connection must be kept alive to keep receiving. Each message is parsed and sent over `tx`.
fn setup_midi(tx: Sender<MidiMsg>) -> Option<(MidiInputConnection<()>, String)> {
    let mut input = MidiInput::new("zygmunt").ok()?;
    input.ignore(Ignore::None);
    let ports = input.ports();
    if ports.is_empty() {
        return None;
    }
    let port = ports
        .iter()
        .find(|p| {
            input
                .port_name(p)
                .map(|n| {
                    let n = n.to_lowercase();
                    n.contains("digitakt") || n.contains("elektron")
                })
                .unwrap_or(false)
        })
        .cloned()
        .unwrap_or_else(|| ports[0].clone());
    let name = input.port_name(&port).unwrap_or_else(|_| "midi".into());
    let conn = input
        .connect(
            &port,
            "zygmunt-in",
            move |_stamp, bytes, _| {
                if let Some(msg) = parse_midi(bytes) {
                    let _ = tx.send(msg);
                }
            },
            (),
        )
        .ok()?;
    Some((conn, name))
}

fn midi_map_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".zygmunt-midi.txt")
}

fn load_cc_map() -> HashMap<u8, Param> {
    let mut m = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(midi_map_path()) {
        for line in text.lines() {
            let mut it = line.split(',');
            if let (Some(cc), Some(idx)) = (it.next(), it.next()) {
                if let (Ok(cc), Ok(idx)) = (cc.parse::<u8>(), idx.parse::<usize>()) {
                    if idx < Param::ALL.len() {
                        m.insert(cc, Param::ALL[idx]);
                    }
                }
            }
        }
    }
    m
}

fn save_cc_map(m: &HashMap<u8, Param>) {
    let mut s = String::new();
    for (cc, p) in m {
        let idx = Param::ALL.iter().position(|x| x == p).unwrap_or(0);
        s.push_str(&format!("{},{}\n", cc, idx));
    }
    let _ = std::fs::write(midi_map_path(), s);
}

/// A held note's identity — keyboard notes key by char (survives octave changes), MIDI by note number.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum NoteId {
    Kbd(char),
    Midi(u8),
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
    delay_sh: Shared,
    dfeed_sh: Shared,
    eq1k_sh: Shared,
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
    detune: f32,  // unison detune spread (analog thickness)
    noise: f32,
    hiss: f32,    // constant background noise floor
    drive: f32,   // mid saturator dry/wet (0 = clean)
    delay: f32,   // strange delay dry/wet (0 = clean)
    dfeed: f32,   // strange delay feedback (repeats)
    eq1k: f32,    // 1 kHz EQ dip depth (0 = flat)
    comp: f32,    // output compression amount (drive into the limiter)
    fenv: f32,    // filter envelope amount (cutoff sweep on each note)
    fdecay: f32,  // filter envelope decay time (s)
    last_note: f32, // time of the most recent note-on, for the filter envelope
    chaos: f32,   // global instability: continuous wander + per-note variation
    octave: i32,
    selected: Param,
    active: HashMap<NoteId, (EventId, Shared)>, // held notes -> (sequencer event, gate)
    seed: u64,                                  // per-voice drift seed source
    rng: u64,                                   // per-note chaos RNG state
    clock: Instant,                             // for continuous wander
    presets: [Option<Preset>; 9],
    toast: Option<String>, // transient "saved/loaded preset N" notification
    toast_until: f32,
    hold: Option<(usize, f32)>, // (preset slot, press time) — release before threshold = load, hold = save
    midi_channel: u8, // 0 = Omni, 1..=16
    midi_port: String,
    cc_map: HashMap<u8, Param>, // CC number -> bound param (MIDI learn)
    learn_armed: bool,          // next CC binds to the selected param
    last_midi: Option<String>,  // last received MIDI message, for the UI
    supports_release: bool,
}

impl App {
    fn start_note(&mut self, id: NoteId, base_hz: f32, vel: f32) {
        if self.active.contains_key(&id) {
            return;
        }
        let (hz, a, d, s, r, drift, noise, seed) = self.voice_params(base_hz);
        let (voice, gate) = make_voice(
            self.wave, hz, vel, a, d, s, r, drift, self.detune, noise, &self.pitch_mod, seed,
        );
        let evid = self
            .sequencer
            .push_relative(0.0, f64::INFINITY, Fade::Smooth, 0.004, 0.01, voice);
        self.active.insert(id, (evid, gate));
    }

    fn stop_note(&mut self, id: NoteId) {
        if let Some((evid, gate)) = self.active.remove(&id) {
            gate.set_value(-1.0); // start the ADSR release
            let tail = self.release as f64 + 0.1;
            self.sequencer.edit_relative(evid, tail, 0.05); // remove voice after the tail
        }
    }

    /// Fallback for terminals without key-release reporting: a fixed-length note.
    fn play_fixed(&mut self, base_hz: f32, vel: f32) {
        let (hz, a, d, s, r, drift, noise, seed) = self.voice_params(base_hz);
        let (voice, _gate) = make_voice(
            self.wave, hz, vel, a, d, s, r, drift, self.detune, noise, &self.pitch_mod, seed,
        );
        let len = (a + d + 0.4 + r) as f64;
        self.sequencer
            .push_relative(0.0, len, Fade::Smooth, 0.004, r as f64, voice);
    }

    /// Per-note voice parameters, with chaos applied as random per-note variation.
    /// `base_hz` is the note's nominal frequency; chaos detune is applied here.
    /// Returns (hz, attack, decay, sustain, release, drift_amt, noise_amt, seed).
    #[allow(clippy::type_complexity)]
    fn voice_params(&mut self, base_hz: f32) -> (f32, f32, f32, f32, f32, f32, f32, u64) {
        self.last_note = self.clock.elapsed().as_secs_f32(); // retrigger the filter envelope
        let c = self.chaos;
        let a = (self.attack * (1.0 + 0.3 * c * self.rnd())).clamp(0.001, 2.0);
        let d = (self.decay * (1.0 + 0.3 * c * self.rnd())).clamp(0.001, 2.0);
        let s = (self.sustain + 0.1 * c * self.rnd()).clamp(0.0, 1.0);
        let r = (self.release * (1.0 + 0.3 * c * self.rnd())).clamp(0.001, 3.0);
        let drift = (self.drift + 0.3 * c * self.rnd().abs()).clamp(0.0, 1.0) * 0.015;
        let noise = (self.noise + 0.2 * c * self.rnd().abs()).clamp(0.0, 1.0) * 0.5;
        let detune = 1.0 + 0.015 * c * self.rnd(); // up to ~+/-25 cents of broken tuning
        let hz = base_hz * detune;
        (hz, a, d, s, r, drift, noise, self.next_seed())
    }

    /// Apply an incoming MIDI message (channel-filtered): notes, or CC -> param (with learn).
    fn handle_midi(&mut self, msg: MidiMsg) {
        let ch_ok = |ch: u8| self.midi_channel == 0 || self.midi_channel == ch + 1;
        match msg {
            MidiMsg::NoteOn { ch, note, vel } if ch_ok(ch) => {
                self.last_midi = Some(format!("note {} v{}", note, vel));
                if vel == 0 {
                    self.stop_note(NoteId::Midi(note));
                } else {
                    let g = vel as f32 / 127.0 * 0.3; // map velocity to per-voice gain
                    self.start_note(NoteId::Midi(note), midi_hz(note as f32), g);
                }
            }
            MidiMsg::NoteOff { ch, note } if ch_ok(ch) => {
                self.last_midi = Some(format!("off {}", note));
                self.stop_note(NoteId::Midi(note));
            }
            MidiMsg::Cc { ch, cc, value } if ch_ok(ch) => {
                self.last_midi = Some(format!("CC{} {}", cc, value));
                if self.learn_armed {
                    self.cc_map.insert(cc, self.selected);
                    self.learn_armed = false;
                    save_cc_map(&self.cc_map);
                    self.set_toast(format!("CC{} → {}", cc, self.selected.name()));
                } else if let Some(&p) = self.cc_map.get(&cc) {
                    self.set_param_norm(p, value as f32 / 127.0);
                }
            }
            _ => {}
        }
    }

    /// Set a param from a normalised 0..1 value (used by MIDI CC, absolute control).
    fn set_param_norm(&mut self, p: Param, norm: f32) {
        let n = norm.clamp(0.0, 1.0);
        match p {
            Param::Attack => self.attack = 0.001 + n * 1.999,
            Param::Decay => self.decay = 0.001 + n * 1.999,
            Param::Sustain => self.sustain = n,
            Param::Release => self.release = 0.001 + n * 2.999,
            Param::Drift => self.drift = n,
            Param::Detune => self.detune = n,
            Param::Noise => self.noise = n,
            Param::Hiss => self.hiss = n,
            Param::Cutoff => self.cutoff = 20.0 * 1000.0_f32.powf(n), // log 20..20000
            Param::Resonance => self.resonance = n * 0.98,
            Param::FEnv => self.fenv = n,
            Param::FDecay => self.fdecay = 0.02 + n * 1.98,
            Param::Drive => self.drive = n,
            Param::Delay => self.delay = n,
            Param::DFeed => self.dfeed = n * 0.9,
            Param::RevAmount => self.reverb_amt = n,
            Param::RevRoom => {
                let v = 10.0 + n * 20.0;
                if (v - self.room).abs() > 1.0 {
                    self.room = v;
                    self.rebuild_reverb();
                }
            }
            Param::RevTime => {
                let v = 0.5 + n * 9.5;
                if (v - self.time).abs() > 0.5 {
                    self.time = v;
                    self.rebuild_reverb();
                }
            }
            Param::RevDiffusion => {
                let v = n;
                if (v - self.diffusion).abs() > 0.05 {
                    self.diffusion = v;
                    self.rebuild_reverb();
                }
            }
            Param::Eq1k => self.eq1k = n,
            Param::Volume => self.volume = n,
            Param::Comp => self.comp = n,
            Param::Bits => {
                self.bits_idx = (n * (BIT_OPTIONS.len() - 1) as f32).round() as usize;
                self.quant.set_value(BIT_OPTIONS[self.bits_idx].1);
            }
            Param::Chaos => self.chaos = n,
        }
    }

    fn cycle_midi_channel(&mut self, d: i32) {
        self.midi_channel = (self.midi_channel as i32 + d).rem_euclid(17) as u8;
    }

    fn adjust(&mut self, d: i32) {
        let d = d as f32;
        match self.selected {
            Param::Attack => self.attack = (self.attack + d * 0.02).clamp(0.001, 2.0),
            Param::Decay => self.decay = (self.decay + d * 0.02).clamp(0.001, 2.0),
            Param::Sustain => self.sustain = (self.sustain + d * 0.05).clamp(0.0, 1.0),
            Param::Release => self.release = (self.release + d * 0.05).clamp(0.001, 3.0),
            Param::Drift => self.drift = (self.drift + d * 0.05).clamp(0.0, 1.0),
            Param::Detune => self.detune = (self.detune + d * 0.05).clamp(0.0, 1.0),
            Param::Noise => self.noise = (self.noise + d * 0.05).clamp(0.0, 1.0),
            Param::Hiss => self.hiss = (self.hiss + d * 0.05).clamp(0.0, 1.0),
            Param::Cutoff => {
                let factor = if d > 0.0 { 1.25 } else { 1.0 / 1.25 };
                self.cutoff = (self.cutoff * factor).clamp(20.0, 20000.0);
            }
            Param::Resonance => self.resonance = (self.resonance + d * 0.05).clamp(0.0, 0.98),
            Param::FEnv => self.fenv = (self.fenv + d * 0.05).clamp(0.0, 1.0),
            Param::FDecay => self.fdecay = (self.fdecay + d * 0.05).clamp(0.02, 2.0),
            Param::Drive => self.drive = (self.drive + d * 0.05).clamp(0.0, 1.0),
            Param::Delay => self.delay = (self.delay + d * 0.05).clamp(0.0, 1.0),
            Param::DFeed => self.dfeed = (self.dfeed + d * 0.05).clamp(0.0, 0.9),
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
            Param::Eq1k => self.eq1k = (self.eq1k + d * 0.05).clamp(0.0, 1.0),
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
        // Filter envelope: fast attack on note-on, exponential decay back to the base cutoff.
        let fenv = if self.fenv > 0.0 {
            let dt = (t - self.last_note).max(0.0);
            self.fenv * 9000.0 * (-dt / self.fdecay.max(0.01)).exp()
        } else {
            0.0
        };
        let cutoff = ((self.cutoff + fenv) * (1.0 + 0.35 * c * w(0x01, 0.8))).clamp(20.0, 20000.0);
        let reso = (self.resonance + 0.15 * c * w(0x02, 0.5)).clamp(0.0, 0.98);
        let reverb = (self.reverb_amt + 0.15 * c * w(0x03, 0.3)).clamp(0.0, 1.0);
        let vol = (self.volume + 0.12 * c * w(0x04, 1.1)).clamp(0.0, 1.0);
        let drive = (self.drive + 0.1 * c * w(0x07, 0.6)).clamp(0.0, 1.0);
        // Pitch wander: a couple of detuned noise layers so held notes warble like sick tape.
        let pitch_mod = 1.0 + 0.02 * c * (0.7 * w(0x05, 1.3) + 0.3 * w(0x06, 4.0));
        self.cutoff_sh.set_value(cutoff);
        self.reso_sh.set_value(reso);
        self.drive_sh.set_value(drive);
        self.delay_sh.set_value(self.delay);
        self.dfeed_sh.set_value(self.dfeed);
        self.eq1k_sh.set_value(self.eq1k);
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
            fenv: self.fenv,
            fdecay: self.fdecay,
            delay: self.delay,
            dfeed: self.dfeed,
            detune: self.detune,
            eq1k: self.eq1k,
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
        self.fenv = p.fenv;
        self.fdecay = p.fdecay;
        self.delay = p.delay;
        self.dfeed = p.dfeed;
        self.detune = p.detune;
        self.eq1k = p.eq1k;
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
            KeyCode::Char('m') => app.cycle_midi_channel(1),
            KeyCode::Char('M') => app.cycle_midi_channel(-1),
            KeyCode::Enter => {
                app.learn_armed = !app.learn_armed;
                if app.learn_armed {
                    app.set_toast(format!("MIDI learn: send a CC for {}", app.selected.name()));
                }
            }
            KeyCode::Char(c @ '1'..='9') => {
                let slot = c as usize - '1' as usize;
                if app.supports_release {
                    // tap (release before threshold) = load; hold = save (handled in the loop)
                    app.hold = Some((slot, app.clock.elapsed().as_secs_f32()));
                } else if ctrl {
                    app.save_preset(slot); // fallback: no key-release -> Ctrl saves
                } else {
                    app.load_preset(slot); // fallback: tap loads
                }
            }
            KeyCode::Char(c) => {
                if let Some(st) = key_to_semitone(c) {
                    let hz = midi_hz((60 + st + 12 * app.octave) as f32);
                    if app.supports_release {
                        app.start_note(NoteId::Kbd(c), hz, VEL);
                    } else {
                        app.play_fixed(hz, VEL);
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
                if ('1'..='9').contains(&c) {
                    let slot = c as usize - '1' as usize;
                    if let Some((s, _)) = app.hold {
                        if s == slot {
                            app.hold = None;
                            app.load_preset(slot); // released before threshold = short tap = load
                        }
                    }
                } else if key_to_semitone(c).is_some() {
                    app.stop_note(NoteId::Kbd(c));
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
        Span::styled(bar(ratio, 16), bar_style),
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
            " ♪ zygmunt ",
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
            Constraint::Length(1),  // [0] waveform + octave
            Constraint::Length(1),  // [1] presets
            Constraint::Length(1),  // [2] MIDI status
            Constraint::Length(12), // [3] parameter columns
            Constraint::Length(1),  // [4] spacer
            Constraint::Length(3),  // [5] piano
            Constraint::Min(1),     // [6] footer
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
    let preset_hint = match app.hold {
        Some((slot, _)) => format!("  ◉ hold {} to save…", slot + 1),
        None => "  tap load · hold to save".to_string(),
    };
    let hint_style = if app.hold.is_some() {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(DIM)
    };
    preset_spans.push(Span::styled(preset_hint, hint_style));
    f.render_widget(Paragraph::new(Line::from(preset_spans)), rows[1]);

    // --- row 2: MIDI status ---
    let ch_label = if app.midi_channel == 0 {
        "Omni".to_string()
    } else {
        app.midi_channel.to_string()
    };
    let mut midi_spans = vec![
        Span::styled(" MIDI ", Style::default().fg(DIM)),
        Span::styled(format!("ch:{ch_label} "), Style::default().fg(IDLE)),
        Span::styled(format!("· {} ", app.midi_port), Style::default().fg(DIM)),
    ];
    if app.learn_armed {
        midi_spans.push(Span::styled(
            format!("· ◉ LEARN {} (send a CC) ", app.selected.name()),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    } else {
        midi_spans.push(Span::styled(
            format!("· {} maps ", app.cc_map.len()),
            Style::default().fg(DIM),
        ));
        if let Some(m) = &app.last_midi {
            midi_spans.push(Span::styled(format!("· {m}"), Style::default().fg(DIM)));
        }
    }
    f.render_widget(Paragraph::new(Line::from(midi_spans)), rows[2]);

    // --- row 3: two parameter columns ---
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[3]);

    let (a, d, s, r) = (app.attack, app.decay, app.sustain, app.release);
    // column A — oscillator + filter
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
        param_row("Detune", format!("{:.0}%", app.detune * 100.0), app.detune, sel(Param::Detune)),
        param_row("Noise", format!("{:.0}%", app.noise * 100.0), app.noise, sel(Param::Noise)),
        param_row("Hiss", format!("{:.0}%", app.hiss * 100.0), app.hiss, sel(Param::Hiss)),
        param_row("Cutoff", format!("{:.0}Hz", cutoff), cutoff_ratio, sel(Param::Cutoff)),
        param_row("Reso", format!("{:.0}%", res / 0.98 * 100.0), res / 0.98, sel(Param::Resonance)),
        param_row("F.Env", format!("{:.0}%", app.fenv * 100.0), app.fenv, sel(Param::FEnv)),
        param_row("F.Decay", format!("{:.2}s", app.fdecay), app.fdecay / 2.0, sel(Param::FDecay)),
    ];
    f.render_widget(Paragraph::new(col_a), mid[0]);

    // column B — saturator + reverb + output + chaos
    let rv = app.reverb_amt;
    let vol = app.volume;
    let col_b = vec![
        param_row("Drive", format!("{:.0}%", app.drive * 100.0), app.drive, sel(Param::Drive)),
        param_row("Delay", format!("{:.0}%", app.delay * 100.0), app.delay, sel(Param::Delay)),
        param_row("D.Feed", format!("{:.0}%", app.dfeed / 0.9 * 100.0), app.dfeed / 0.9, sel(Param::DFeed)),
        param_row("Reverb", format!("{:.0}%", rv * 100.0), rv, sel(Param::RevAmount)),
        param_row("Room", format!("{:.0}m", app.room), (app.room - 10.0) / 20.0, sel(Param::RevRoom)),
        param_row("Time", format!("{:.2}s", app.time), app.time / 10.0, sel(Param::RevTime)),
        param_row("Diffuse", format!("{:.0}%", app.diffusion * 100.0), app.diffusion, sel(Param::RevDiffusion)),
        param_row("EQ 1k", format!("{:.0}%", app.eq1k * 100.0), app.eq1k, sel(Param::Eq1k)),
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
    f.render_widget(Paragraph::new(col_b), mid[1]);

    // --- piano ---
    let act = |c: char| app.active.contains_key(&NoteId::Kbd(c));
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
        .split(rows[5]);
    f.render_widget(Paragraph::new(key_line(&black_slots, width)), piano[0]);
    f.render_widget(Paragraph::new(key_line(&white_slots, width)), piano[1]);
    f.render_widget(Paragraph::new(key_line(&note_slots, width)), piano[2]);

    // --- footer ---
    let mut hint = String::from(
        "Tab param · ↑↓ adjust · ←→ wave · ,. octave · m MIDI · Enter learn · 1-9 tap=load/hold=save · Esc quit",
    );
    if !app.supports_release {
        hint.push_str("  (no key-release)");
    }
    f.render_widget(
        Paragraph::new(Span::styled(hint, Style::default().fg(DIM))).alignment(Alignment::Center),
        rows[6],
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
    let delay_sh = shared(0.0);
    let dfeed_sh = shared(0.35);
    let eq1k_sh = shared(0.0);
    let reverb_sh = shared(0.25);
    let vol_sh = shared(0.7);
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
        &delay_sh,
        &dfeed_sh,
        &eq1k_sh,
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

    // --- MIDI input (kept alive for the program's lifetime) ---
    let (midi_tx, midi_rx) = std::sync::mpsc::channel::<MidiMsg>();
    let _midi_conn = setup_midi(midi_tx);
    let midi_port = _midi_conn
        .as_ref()
        .map(|(_, name)| name.clone())
        .unwrap_or_else(|| "no device".into());

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
        delay_sh,
        dfeed_sh,
        eq1k_sh,
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
        volume: 0.7,
        room,
        time,
        diffusion,
        wave: Wave::Saw,
        attack: 0.02,
        decay: 0.15,
        sustain: 0.6,
        release: 0.3,
        drift: 0.3,
        detune: 0.25,
        noise: 0.0,
        hiss: 0.0,
        drive: 0.0,
        delay: 0.0,
        dfeed: 0.35,
        eq1k: 0.0,
        comp: 0.0,
        fenv: 0.0,
        fdecay: 0.3,
        last_note: -1000.0,
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
        hold: None,
        midi_channel: 0,
        midi_port,
        cc_map: load_cc_map(),
        learn_armed: false,
        last_midi: None,
        supports_release: supports,
    };

    // --- event loop ---
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut last_draw = -1.0f32;
        loop {
            // drain MIDI promptly (low note latency); apply on the UI thread
            while let Ok(msg) = midi_rx.try_recv() {
                app.handle_midi(msg);
            }
            let t = app.clock.elapsed().as_secs_f32();
            app.tick_chaos(t);
            if app.toast.is_some() && t > app.toast_until {
                app.toast = None;
            }
            // hold a preset key past the threshold -> save (release before = load)
            if let Some((slot, t0)) = app.hold {
                if t - t0 >= 0.4 {
                    app.save_preset(slot);
                    app.hold = None;
                }
            }
            if t - last_draw >= 0.033 {
                terminal.draw(|f| ui(f, &app))?;
                last_draw = t;
            }
            if event::poll(Duration::from_millis(3))? {
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
