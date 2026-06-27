//! zygdrum — a 3-voice FM drum synth for glitch / IDM / Aphex-flavoured percussion.
//!
//! Three one-shot FM percussion voices, each triggered by a note pitch-class:
//!   C -> kick   ·   D -> snare/clap   ·   E -> rimshot/hihat
//! (computer keys a / s / d, or MIDI notes by pitch class, velocity-sensitive).
//! Each voice is built fresh per hit with the current knobs baked in, pushed to the fundsp
//! Sequencer as a finite one-shot. Master glitch chain: drive (tanh) -> volume -> limiter, with a
//! bit-crusher in the audio callback. Architecture mirrors zygmunt: UI thread owns state and pushes
//! hits; the cpal callback owns DSP and never allocates.

use std::io::{self, Stdout};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use fundsp::prelude64::*;

use midir::{Ignore, MidiInput, MidiInputConnection};

use ratatui::crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
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
const ACCENT: Color = Color::Rgb(255, 150, 90); // warm orange (drums)
const IDLE: Color = Color::Rgb(165, 170, 185);
const DIM: Color = Color::Rgb(95, 98, 120);
const BG: Color = Color::Rgb(16, 14, 20);

const DRUMS: [&str; 3] = ["KICK", "SNARE", "HIHAT"];

/// Bit-crush options (label, quantization levels). 0 = off. Aggressive end for IDM.
const BIT_OPTIONS: [(&str, f32); 6] = [
    ("off", 0.0),
    ("12-bit", 2048.0),
    ("8-bit", 128.0),
    ("6-bit", 32.0),
    ("4-bit", 8.0),
    ("3-bit", 4.0),
];

// ---------- drum voice ----------

#[derive(Clone, Copy)]
struct DrumParams {
    tune: f32,
    decay: f32,
    fm: f32,
    snap: f32,
}

/// FM kick: sine carrier with a fast pitch drop, FM body, and a short noise click.
fn build_kick(p: DrumParams, vel: f32) -> Box<dyn AudioUnit> {
    let base = (35.0 + 55.0 * p.tune) as f64; // 35..90 Hz floor
    let drop = 150.0_f64;
    let amp_rate = (16.0 - 12.0 * p.decay) as f64; // fast..slow
    let fm_amt = (p.fm * 350.0) as f64;
    let snap = p.snap as f64;
    let amp = (vel * 0.9) as f64;
    let carrier = lfo(move |t| base + drop * (-t * 40.0).exp());
    let modf = lfo(move |t| base + drop * (-t * 40.0).exp());
    let mi = lfo(move |t| fm_amt * (-t * 26.0).exp());
    let env = lfo(move |t| amp * (-t * amp_rate).exp());
    let click = lfo(move |t| snap * (-t * 350.0).exp());
    let body = ((carrier + (modf >> sine()) * mi) >> sine()) * env;
    Box::new(body + noise() * click)
}

/// FM snare: short inharmonic FM body plus a filtered noise crack.
fn build_snare(p: DrumParams, vel: f32) -> Box<dyn AudioUnit> {
    let base = (150.0 + 150.0 * p.tune) as f64; // 150..300
    let amp_rate = (28.0 - 16.0 * p.decay) as f64;
    let fm_amt = (p.fm * 250.0) as f64;
    let noise_amt = (0.5 + p.snap) as f64;
    let amp = (vel * 0.8) as f64;
    let cf = lfo(move |t| base + 90.0 * (-t * 55.0).exp());
    let mf = lfo(move |t| (base + 90.0 * (-t * 55.0).exp()) * 1.6); // inharmonic ratio
    let mi = lfo(move |t| fm_amt * (-t * 45.0).exp());
    let benv = lfo(move |t| amp * 0.7 * (-t * (amp_rate + 10.0)).exp());
    let nenv = lfo(move |t| amp * noise_amt * (-t * amp_rate).exp());
    let body = ((cf + (mf >> sine()) * mi) >> sine()) * benv;
    let noise_part = (noise() >> highpass_hz(1400.0, 1.0)) * nenv;
    Box::new(body + noise_part)
}

/// FM hihat/rimshot: bright inharmonic metallic FM plus high-passed noise, very fast decay.
fn build_hihat(p: DrumParams, vel: f32) -> Box<dyn AudioUnit> {
    let base = 3500.0 + 5000.0 * p.tune; // metallic base (f32 for `constant`)
    let amp_rate = (80.0 - 55.0 * p.decay) as f64; // very fast..fast
    let fm_amt = p.fm * 4000.0; // f32: scalar mult on An needs f32
    let noise_amt = 0.4 + p.snap;
    let amp = (vel * 0.6) as f64;
    let metallic = (constant(base) + (constant(base * 1.43) >> sine()) * fm_amt) >> sine();
    let env = lfo(move |t| amp * (-t * amp_rate).exp());
    let hat = (metallic * 0.45 + (noise() >> highpass_hz(6500.0, 1.0)) * noise_amt) * env;
    Box::new(hat)
}

fn build_drum(drum: usize, p: DrumParams, vel: f32) -> Box<dyn AudioUnit> {
    match drum {
        0 => build_kick(p, vel),
        1 => build_snare(p, vel),
        _ => build_hihat(p, vel),
    }
}

/// One-shot length in seconds (the sequencer removes the voice after this).
fn drum_length(drum: usize, p: &DrumParams) -> f64 {
    match drum {
        0 => (0.15 + 1.0 * p.decay) as f64,
        1 => (0.08 + 0.4 * p.decay) as f64,
        _ => (0.03 + 0.25 * p.decay) as f64,
    }
}

/// C/D/E pitch classes -> kick/snare/hihat.
fn drum_for_semitone(st: i32) -> Option<usize> {
    match st.rem_euclid(12) {
        0 => Some(0), // C
        2 => Some(1), // D
        4 => Some(2), // E
        _ => None,
    }
}

fn key_to_semitone(c: char) -> Option<i32> {
    Some(match c {
        'a' => 0,
        'w' => 1,
        's' => 2,
        'e' => 3,
        'd' => 4,
        'f' => 5,
        't' => 6,
        'g' => 7,
        'y' => 8,
        'h' => 9,
        'u' => 10,
        'j' => 11,
        'k' => 12,
        _ => return None,
    })
}

// ---------- master net ----------

/// Master glitch chain: pan -> drive (dry/wet hard tanh) -> volume -> limiter.
fn build_net(seq_backend: Box<dyn AudioUnit>, drive: &Shared, volume: &Shared, sr: f64) -> Net {
    let mut net = Net::wrap(seq_backend);
    net = net >> pan(0.0); // mono -> stereo
    let dist = (pass() * 3.0 >> shape(Tanh(2.0))) | (pass() * 3.0 >> shape(Tanh(2.0)));
    net = net
        >> ((1.0 - var(drive) >> follow(0.01) >> split::<U2>()) * multipass::<U2>()
            & (var(drive) >> follow(0.01) >> split::<U2>()) * dist);
    net = net >> ((var(volume) >> follow(0.02) >> split::<U2>()) * multipass::<U2>());
    net = net >> limiter_stereo(0.003, 0.1);
    net.set_sample_rate(sr);
    net
}

// ---------- MIDI ----------

enum MidiMsg {
    NoteOn { ch: u8, note: u8, vel: u8 },
}

fn parse_midi(bytes: &[u8]) -> Option<MidiMsg> {
    if bytes.len() < 3 {
        return None;
    }
    let ch = bytes[0] & 0x0F;
    if bytes[0] & 0xF0 == 0x90 {
        Some(MidiMsg::NoteOn {
            ch,
            note: bytes[1],
            vel: bytes[2],
        })
    } else {
        None
    }
}

fn setup_midi(tx: Sender<MidiMsg>) -> Option<(MidiInputConnection<()>, String)> {
    let mut input = MidiInput::new("zygdrum").ok()?;
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
            "zygdrum-in",
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

// ---------- app state ----------

struct App {
    sequencer: Sequencer,
    _net: Net,
    drive_sh: Shared,
    vol_sh: Shared,
    quant: Shared,
    bits_idx: usize,
    drums: [DrumParams; 3],
    selected_drum: usize,
    selected_param: usize, // 0..3 drum params, 4 drive, 5 bits, 6 volume
    flash: [f32; 3],       // time of last hit per drum, for UI feedback
    midi_channel: u8,
    midi_port: String,
    last_midi: Option<String>,
    clock: Instant,
}

const NUM_PARAMS: usize = 7;

impl App {
    fn trigger(&mut self, drum: usize, vel: f32) {
        self.selected_drum = drum;
        self.flash[drum] = self.clock.elapsed().as_secs_f32();
        let p = self.drums[drum];
        let voice = build_drum(drum, p, vel);
        let len = drum_length(drum, &p);
        self.sequencer
            .push_relative(0.0, len, Fade::Smooth, 0.001, 0.02, voice);
    }

    fn adjust(&mut self, d: i32) {
        let step = d as f32 * 0.05;
        let i = self.selected_drum;
        match self.selected_param {
            0 => self.drums[i].tune = (self.drums[i].tune + step).clamp(0.0, 1.0),
            1 => self.drums[i].decay = (self.drums[i].decay + step).clamp(0.0, 1.0),
            2 => self.drums[i].fm = (self.drums[i].fm + step).clamp(0.0, 1.0),
            3 => self.drums[i].snap = (self.drums[i].snap + step).clamp(0.0, 1.0),
            4 => {
                let v = (self.drive_sh.value() + step).clamp(0.0, 1.0);
                self.drive_sh.set_value(v);
            }
            5 => {
                let n = BIT_OPTIONS.len() as i32;
                self.bits_idx = ((self.bits_idx as i32 + d).rem_euclid(n)) as usize;
                self.quant.set_value(BIT_OPTIONS[self.bits_idx].1);
            }
            _ => {
                let v = (self.vol_sh.value() + step).clamp(0.0, 1.0);
                self.vol_sh.set_value(v);
            }
        }
    }

    fn handle_midi(&mut self, msg: MidiMsg) {
        let MidiMsg::NoteOn { ch, note, vel } = msg;
        if self.midi_channel != 0 && self.midi_channel != ch + 1 {
            return;
        }
        self.last_midi = Some(format!("note {note} v{vel}"));
        if vel == 0 {
            return;
        }
        if let Some(drum) = drum_for_semitone(note as i32) {
            self.trigger(drum, vel as f32 / 127.0);
        }
    }
}

/// Returns true to quit.
fn handle_key(app: &mut App, k: KeyEvent) -> bool {
    if k.kind != KeyEventKind::Press {
        return false;
    }
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    match k.code {
        KeyCode::Esc => return true,
        KeyCode::Char('c') if ctrl => return true,
        KeyCode::Tab => app.selected_param = (app.selected_param + 1) % NUM_PARAMS,
        KeyCode::BackTab => app.selected_param = (app.selected_param + NUM_PARAMS - 1) % NUM_PARAMS,
        KeyCode::Up => app.adjust(1),
        KeyCode::Down => app.adjust(-1),
        KeyCode::Char('m') => app.midi_channel = (app.midi_channel as i32 + 1).rem_euclid(17) as u8,
        KeyCode::Char(c @ '1'..='3') => app.selected_drum = c as usize - '1' as usize,
        KeyCode::Char(c) => {
            if let Some(st) = key_to_semitone(c) {
                if let Some(drum) = drum_for_semitone(st) {
                    app.trigger(drum, 0.9);
                }
            }
        }
        _ => {}
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

fn slider(name: &str, value: String, ratio: f32, selected: bool) -> Line<'static> {
    let (label, val) = if selected {
        (
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            Style::default().fg(ACCENT),
        )
    } else {
        (Style::default().fg(IDLE), Style::default().fg(DIM))
    };
    Line::from(vec![
        Span::styled(format!(" {:<8}", name), label),
        Span::styled(bar(ratio, 20), Style::default().fg(if selected { ACCENT } else { DIM })),
        Span::styled(format!(" {}", value), val),
    ])
}

fn ui(f: &mut Frame, app: &App) {
    let area = f.area();
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(
            " ◆ zygdrum · FM drums ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))
        .title_alignment(Alignment::Center)
        .style(Style::default().bg(BG));
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // drum tabs
            Constraint::Length(1), // midi
            Constraint::Length(1), // separator
            Constraint::Length(4), // drum params
            Constraint::Length(1), // gap
            Constraint::Length(3), // master params
            Constraint::Length(1), // separator
            Constraint::Min(1),    // footer
        ])
        .split(inner);

    // drum tabs
    let now = app.clock.elapsed().as_secs_f32();
    let mut tabs = vec![Span::styled(" DRUM ", Style::default().fg(DIM))];
    for (i, name) in DRUMS.iter().enumerate() {
        let hit = now - app.flash[i] < 0.12;
        let st = if i == app.selected_drum {
            Style::default().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)
        } else if hit {
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(IDLE)
        };
        let key = ['C', 'D', 'E'][i];
        tabs.push(Span::styled(format!(" {name} ({key}) "), st));
        tabs.push(Span::raw(" "));
    }
    f.render_widget(Paragraph::new(Line::from(tabs)), rows[0]);

    // midi line
    let ch = if app.midi_channel == 0 {
        "Omni".to_string()
    } else {
        app.midi_channel.to_string()
    };
    let mut midi = vec![
        Span::styled(" MIDI ", Style::default().fg(DIM)),
        Span::styled(format!("ch:{ch} "), Style::default().fg(IDLE)),
        Span::styled(format!("· {} ", app.midi_port), Style::default().fg(DIM)),
    ];
    if let Some(m) = &app.last_midi {
        midi.push(Span::styled(format!("· {m}"), Style::default().fg(DIM)));
    }
    f.render_widget(Paragraph::new(Line::from(midi)), rows[1]);

    let rule = || {
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(DIM))
    };
    f.render_widget(rule(), rows[2]);
    f.render_widget(rule(), rows[6]);

    // selected drum params
    let dp = app.drums[app.selected_drum];
    let sp = app.selected_param;
    let pct = |v: f32| format!("{:.0}%", v * 100.0);
    let drum_lines = vec![
        slider("Tune", pct(dp.tune), dp.tune, sp == 0),
        slider("Decay", pct(dp.decay), dp.decay, sp == 1),
        slider("FM", pct(dp.fm), dp.fm, sp == 2),
        slider("Snap", pct(dp.snap), dp.snap, sp == 3),
    ];
    f.render_widget(Paragraph::new(drum_lines), rows[3]);

    // master params
    let drive = app.drive_sh.value();
    let vol = app.vol_sh.value();
    let master = vec![
        slider("Drive", pct(drive), drive, sp == 4),
        slider(
            "Bits",
            BIT_OPTIONS[app.bits_idx].0.to_string(),
            app.bits_idx as f32 / (BIT_OPTIONS.len() - 1) as f32,
            sp == 5,
        ),
        slider("Volume", pct(vol), vol, sp == 6),
    ];
    f.render_widget(Paragraph::new(master), rows[5]);

    let hint = "a/s/d (or any C/D/E) trigger · 1/2/3 select drum · Tab param · ↑↓ adjust · m MIDI · Esc quit";
    f.render_widget(
        Paragraph::new(Span::styled(hint, Style::default().fg(DIM))).alignment(Alignment::Center),
        rows[7],
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

fn restore_terminal() {
    let mut out = io::stdout();
    let _ = execute!(out, LeaveAlternateScreen);
    let _ = disable_raw_mode();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = device.default_output_config()?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let sample_rate = config.sample_rate as f64;

    let mut sequencer = Sequencer::new(0, 1, ReplayMode::None);
    sequencer.set_sample_rate(sample_rate);
    let seq_backend = sequencer.backend();

    let drive_sh = shared(0.0);
    let vol_sh = shared(0.7);
    let quant = shared(0.0);
    let mut net = build_net(Box::new(seq_backend), &drive_sh, &vol_sh, sample_rate);
    let backend = BlockRateAdapter::new(Box::new(net.backend()));

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, backend, quant.clone())?,
        cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, backend, quant.clone())?,
        cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config, backend, quant.clone())?,
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    };
    stream.play()?;

    let (midi_tx, midi_rx) = std::sync::mpsc::channel::<MidiMsg>();
    let _midi_conn = setup_midi(midi_tx);
    let midi_port = _midi_conn
        .as_ref()
        .map(|(_, n)| n.clone())
        .unwrap_or_else(|| "no device".into());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        default_hook(info);
    }));
    let mut terminal: Terminal<CrosstermBackend<Stdout>> =
        Terminal::new(CrosstermBackend::new(stdout))?;

    let mut app = App {
        sequencer,
        _net: net,
        drive_sh,
        vol_sh,
        quant,
        bits_idx: 0,
        drums: [
            DrumParams { tune: 0.3, decay: 0.5, fm: 0.3, snap: 0.4 },
            DrumParams { tune: 0.4, decay: 0.4, fm: 0.5, snap: 0.6 },
            DrumParams { tune: 0.5, decay: 0.2, fm: 0.6, snap: 0.5 },
        ],
        selected_drum: 0,
        selected_param: 0,
        flash: [-1.0; 3],
        midi_channel: 0,
        midi_port,
        last_midi: None,
        clock: Instant::now(),
    };

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut last_draw = -1.0f32;
        loop {
            while let Ok(msg) = midi_rx.try_recv() {
                app.handle_midi(msg);
            }
            let t = app.clock.elapsed().as_secs_f32();
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

    restore_terminal();
    result
}
