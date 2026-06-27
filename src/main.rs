//! zygdrum — a 3-voice FM drum synth for glitch / IDM / Aphex-flavoured percussion.
//!
//! Three one-shot FM percussion voices (kick / snare / hihat), each a generalized FM percussion
//! engine with 9 params, triggered by a note pitch-class: C->kick, D->snare, E->hihat (keys a/s/d
//! or any C/D/E, or MIDI, velocity-sensitive). All voices are visible at once as a matrix
//! (param rows x drum columns). CC->param MIDI learn binds the selected cell. Master glitch chain:
//! drive (tanh) -> volume -> limiter, with a bit-crusher in the audio callback.

use std::cell::RefCell;
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
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame, Terminal,
};

// ---------- palette ----------
const ACCENT: Color = Color::Rgb(255, 150, 90);
const IDLE: Color = Color::Rgb(165, 170, 185);
const DIM: Color = Color::Rgb(95, 98, 120);
const BG: Color = Color::Rgb(16, 14, 20);

const DRUMS: [&str; 3] = ["KICK", "SNARE", "HIHAT"];
const PARAMS: [&str; 9] = [
    "Tune", "Ratio", "FM", "FMDec", "PEnv", "PDec", "Decay", "Snap", "Tone",
];
const MASTER: [&str; 3] = ["Drive", "Bits", "Volume"];
const NP: usize = 9; // params per drum
const NROWS: usize = NP + 3; // selectable rows: 9 drum params + 3 master

const BIT_OPTIONS: [(&str, f32); 6] = [
    ("off", 0.0),
    ("12-bit", 2048.0),
    ("8-bit", 128.0),
    ("6-bit", 32.0),
    ("4-bit", 8.0),
    ("3-bit", 4.0),
];

// ---------- drum voice (generalized FM percussion) ----------

#[derive(Clone, Copy)]
struct DrumParams {
    p: [f32; NP], // tune, ratio, fm, fmdec, penv, pdec, decay, snap, tone (all 0..1)
}

fn build_drum(d: DrumParams, vel: f32) -> Box<dyn AudioUnit> {
    let v = d.p;
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
    Box::new(osc + noise_part)
}

fn drum_length(d: &DrumParams) -> f64 {
    (0.08 + 1.8 * d.p[6]) as f64
}

fn drum_for_semitone(st: i32) -> Option<usize> {
    match st.rem_euclid(12) {
        0 => Some(0),
        2 => Some(1),
        4 => Some(2),
        _ => None,
    }
}

fn key_to_semitone(c: char) -> Option<i32> {
    Some(match c {
        'a' => 0, 'w' => 1, 's' => 2, 'e' => 3, 'd' => 4, 'f' => 5,
        't' => 6, 'g' => 7, 'y' => 8, 'h' => 9, 'u' => 10, 'j' => 11, 'k' => 12,
        _ => return None,
    })
}

// ---------- master net ----------

fn build_net(seq_backend: Box<dyn AudioUnit>, drive: &Shared, volume: &Shared, sr: f64) -> Net {
    let mut net = Net::wrap(seq_backend);
    net = net >> pan(0.0);
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
    Cc { ch: u8, cc: u8, value: u8 },
}

fn parse_midi(bytes: &[u8]) -> Option<MidiMsg> {
    if bytes.len() < 3 {
        return None;
    }
    let ch = bytes[0] & 0x0F;
    match bytes[0] & 0xF0 {
        0x90 => Some(MidiMsg::NoteOn { ch, note: bytes[1], vel: bytes[2] }),
        0xB0 => Some(MidiMsg::Cc { ch, cc: bytes[1], value: bytes[2] }),
        _ => None,
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
            input.port_name(p).map(|n| {
                let n = n.to_lowercase();
                n.contains("digitakt") || n.contains("elektron")
            }).unwrap_or(false)
        })
        .cloned()
        .unwrap_or_else(|| ports[0].clone());
    let name = input.port_name(&port).unwrap_or_else(|_| "midi".into());
    let conn = input
        .connect(&port, "zygdrum-in", move |_s, bytes, _| {
            if let Some(m) = parse_midi(bytes) {
                let _ = tx.send(m);
            }
        }, ())
        .ok()?;
    Some((conn, name))
}

fn cc_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".zygdrum-midi.txt")
}

fn load_cc_map() -> HashMap<u8, usize> {
    let mut m = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(cc_path()) {
        for line in text.lines() {
            let mut it = line.split(',');
            if let (Some(cc), Some(id)) = (it.next(), it.next()) {
                if let (Ok(cc), Ok(id)) = (cc.parse::<u8>(), id.parse::<usize>()) {
                    if id < 30 {
                        m.insert(cc, id);
                    }
                }
            }
        }
    }
    m
}

fn save_cc_map(m: &HashMap<u8, usize>) {
    let mut s = String::new();
    for (cc, id) in m {
        s.push_str(&format!("{cc},{id}\n"));
    }
    let _ = std::fs::write(cc_path(), s);
}

// ---------- app state ----------

#[derive(Default)]
struct Hits {
    cells: Vec<(usize, Rect)>, // (control id, bar rect)
}

struct App {
    sequencer: Sequencer,
    _net: Net,
    drive_sh: Shared,
    vol_sh: Shared,
    quant: Shared,
    bits_idx: usize,
    drums: [DrumParams; 3],
    sel_drum: usize,
    sel_row: usize, // 0..NROWS (0..9 drum params, 9..12 master)
    flash: [f32; 3],
    cc_map: HashMap<u8, usize>,
    learn: bool,
    midi_channel: u8,
    midi_port: String,
    last_midi: Option<String>,
    toast: Option<String>,
    toast_until: f32,
    hits: RefCell<Hits>,
    clock: Instant,
}

impl App {
    /// Control id for the currently selected cell.
    fn cur_id(&self) -> usize {
        if self.sel_row < NP {
            self.sel_drum * NP + self.sel_row
        } else {
            27 + (self.sel_row - NP)
        }
    }

    fn control_norm(&self, id: usize) -> f32 {
        if id < 27 {
            self.drums[id / NP].p[id % NP]
        } else {
            match id - 27 {
                0 => self.drive_sh.value(),
                1 => self.bits_idx as f32 / (BIT_OPTIONS.len() - 1) as f32,
                _ => self.vol_sh.value(),
            }
        }
    }

    fn set_norm(&mut self, id: usize, n: f32) {
        let n = n.clamp(0.0, 1.0);
        if id < 27 {
            self.drums[id / NP].p[id % NP] = n;
        } else {
            match id - 27 {
                0 => self.drive_sh.set_value(n),
                1 => {
                    self.bits_idx = (n * (BIT_OPTIONS.len() - 1) as f32).round() as usize;
                    self.quant.set_value(BIT_OPTIONS[self.bits_idx].1);
                }
                _ => self.vol_sh.set_value(n),
            }
        }
    }

    fn adjust(&mut self, d: i32) {
        let id = self.cur_id();
        if id == 28 {
            let n = BIT_OPTIONS.len() as i32;
            self.bits_idx = ((self.bits_idx as i32 + d).rem_euclid(n)) as usize;
            self.quant.set_value(BIT_OPTIONS[self.bits_idx].1);
        } else {
            let cur = self.control_norm(id);
            self.set_norm(id, cur + d as f32 * 0.05);
        }
    }

    fn trigger(&mut self, drum: usize, vel: f32) {
        self.sel_drum = drum;
        self.flash[drum] = self.clock.elapsed().as_secs_f32();
        let p = self.drums[drum];
        let voice = build_drum(p, vel);
        let len = drum_length(&p);
        self.sequencer.push_relative(0.0, len, Fade::Smooth, 0.001, 0.02, voice);
    }

    fn set_toast(&mut self, m: String) {
        self.toast = Some(m);
        self.toast_until = self.clock.elapsed().as_secs_f32() + 1.6;
    }

    fn handle_midi(&mut self, msg: MidiMsg) {
        let ch_ok = |ch: u8| self.midi_channel == 0 || self.midi_channel == ch + 1;
        match msg {
            MidiMsg::NoteOn { ch, note, vel } if ch_ok(ch) => {
                self.last_midi = Some(format!("note {note} v{vel}"));
                if vel > 0 {
                    if let Some(drum) = drum_for_semitone(note as i32) {
                        self.trigger(drum, vel as f32 / 127.0);
                    }
                }
            }
            MidiMsg::Cc { ch, cc, value } if ch_ok(ch) => {
                self.last_midi = Some(format!("CC{cc} {value}"));
                if self.learn {
                    let id = self.cur_id();
                    self.cc_map.insert(cc, id);
                    self.learn = false;
                    save_cc_map(&self.cc_map);
                    self.set_toast(format!("CC{cc} -> {}", control_name(id)));
                } else if let Some(&id) = self.cc_map.get(&cc) {
                    self.set_norm(id, value as f32 / 127.0);
                }
            }
            _ => {}
        }
    }
}

fn control_name(id: usize) -> String {
    if id < 27 {
        format!("{} {}", DRUMS[id / NP], PARAMS[id % NP])
    } else {
        MASTER[id - 27].to_string()
    }
}

// ---------- input ----------

fn handle_key(app: &mut App, k: KeyEvent) -> bool {
    if k.kind != KeyEventKind::Press {
        return false;
    }
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    match k.code {
        KeyCode::Esc => return true,
        KeyCode::Char('c') if ctrl => return true,
        KeyCode::Tab => app.sel_row = (app.sel_row + 1) % NROWS,
        KeyCode::BackTab => app.sel_row = (app.sel_row + NROWS - 1) % NROWS,
        KeyCode::Down => app.sel_row = (app.sel_row + 1) % NROWS,
        KeyCode::Up => app.adjust_or_nav_up(),
        KeyCode::Left => app.sel_drum = (app.sel_drum + 2) % 3,
        KeyCode::Right => app.sel_drum = (app.sel_drum + 1) % 3,
        KeyCode::Char('=') | KeyCode::Char('+') => app.adjust(1),
        KeyCode::Char('-') | KeyCode::Char('_') => app.adjust(-1),
        KeyCode::Enter => {
            app.learn = !app.learn;
            if app.learn {
                app.set_toast(format!("MIDI learn: send a CC for {}", control_name(app.cur_id())));
            }
        }
        KeyCode::Char('m') => app.midi_channel = (app.midi_channel as i32 + 1).rem_euclid(17) as u8,
        KeyCode::Char(c @ '1'..='3') => app.sel_drum = c as usize - '1' as usize,
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

impl App {
    // Up arrow: navigate row up (paired with Down). Value adjust is -/= or scroll/drag.
    fn adjust_or_nav_up(&mut self) {
        self.sel_row = (self.sel_row + NROWS - 1) % NROWS;
    }
}

fn cell_at(app: &App, col: u16, row: u16) -> Option<(usize, Rect)> {
    app.hits.borrow().cells.iter()
        .find(|(_, r)| col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height)
        .copied()
}

fn handle_mouse(app: &mut App, me: MouseEvent) {
    let (col, row) = (me.column, me.row);
    match me.kind {
        MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
            let d = if matches!(me.kind, MouseEventKind::ScrollUp) { 1 } else { -1 };
            if let Some((id, _)) = cell_at(app, col, row) {
                select_id(app, id);
            }
            app.adjust(d);
        }
        MouseEventKind::Down(MouseButton::Left) | MouseEventKind::Drag(MouseButton::Left) => {
            if let Some((id, r)) = cell_at(app, col, row) {
                select_id(app, id);
                if id != 28 {
                    let norm = (col.saturating_sub(r.x)) as f32 / std::cmp::max(r.width, 1) as f32;
                    app.set_norm(id, norm.clamp(0.0, 1.0));
                }
            }
        }
        _ => {}
    }
}

fn select_id(app: &mut App, id: usize) {
    if id < 27 {
        app.sel_drum = id / NP;
        app.sel_row = id % NP;
    } else {
        app.sel_row = NP + (id - 27);
    }
}

// ---------- rendering ----------

fn cell_bar(ratio: f32, width: usize) -> String {
    let filled = (ratio.clamp(0.0, 1.0) * width as f32).round() as usize;
    let mut s = String::with_capacity(width);
    for _ in 0..filled {
        s.push('▓');
    }
    for _ in filled..width {
        s.push('░');
    }
    s
}

fn ui(f: &mut Frame, app: &App) {
    let area = f.area();
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" ◆ zygdrum · FM drums ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)))
        .title_alignment(Alignment::Center)
        .style(Style::default().bg(BG));
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // [0] column headers
            Constraint::Length(NP as u16), // [1] matrix
            Constraint::Length(1),  // [2] separator
            Constraint::Length(3),  // [3] master
            Constraint::Length(1),  // [4] midi/learn
            Constraint::Length(1),  // [5] separator
            Constraint::Min(1),     // [6] footer
        ])
        .split(inner);

    let label_w: u16 = 8;
    let bar_w: u16 = 14;
    let col_gap: u16 = 2;
    let now = app.clock.elapsed().as_secs_f32();

    // column headers
    let mut hd: Vec<Span> = vec![Span::styled(format!("{:<width$}", "", width = label_w as usize), Style::default())];
    for (i, name) in DRUMS.iter().enumerate() {
        let hit = now - app.flash[i] < 0.12;
        let st = if i == app.sel_drum {
            Style::default().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)
        } else if hit {
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(IDLE)
        };
        hd.push(Span::styled(format!("{:^width$}", format!("{} ({})", name, ['C','D','E'][i]), width = bar_w as usize), st));
        hd.push(Span::raw(" ".repeat(col_gap as usize)));
    }
    f.render_widget(Paragraph::new(Line::from(hd)), rows[0]);

    // matrix: param rows x drum columns
    let mut hits = app.hits.borrow_mut();
    hits.cells.clear();
    let mut lines: Vec<Line> = Vec::with_capacity(NP);
    for (pi, pname) in PARAMS.iter().enumerate() {
        let row_sel = app.sel_row == pi;
        let mut spans: Vec<Span> = vec![Span::styled(
            format!("{:<width$}", pname, width = label_w as usize),
            if row_sel { Style::default().fg(ACCENT).add_modifier(Modifier::BOLD) } else { Style::default().fg(IDLE) },
        )];
        for drum in 0..3 {
            let id = drum * NP + pi;
            let sel = drum == app.sel_drum && row_sel;
            let val = app.drums[drum].p[pi];
            let style = if sel {
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(if drum == app.sel_drum { IDLE } else { DIM })
            };
            spans.push(Span::styled(cell_bar(val, bar_w as usize), style));
            spans.push(Span::raw(" ".repeat(col_gap as usize)));
            let x = rows[1].x + label_w + (drum as u16) * (bar_w + col_gap);
            hits.cells.push((id, Rect::new(x, rows[1].y + pi as u16, bar_w, 1)));
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), rows[1]);

    f.render_widget(Block::default().borders(Borders::TOP).border_style(Style::default().fg(DIM)), rows[2]);

    // master params (rows NP..NP+3)
    let mut mlines: Vec<Line> = Vec::new();
    for (mi, mname) in MASTER.iter().enumerate() {
        let id = 27 + mi;
        let sel = app.sel_row == NP + mi;
        let (valstr, ratio) = if mi == 1 {
            (BIT_OPTIONS[app.bits_idx].0.to_string(), app.bits_idx as f32 / (BIT_OPTIONS.len() - 1) as f32)
        } else {
            let v = app.control_norm(id);
            (format!("{:.0}%", v * 100.0), v)
        };
        let style = if sel { Style::default().fg(ACCENT).add_modifier(Modifier::BOLD) } else { Style::default().fg(IDLE) };
        mlines.push(Line::from(vec![
            Span::styled(format!("{:<width$}", mname, width = label_w as usize), style),
            Span::styled(cell_bar(ratio, bar_w as usize), if sel { Style::default().fg(ACCENT) } else { Style::default().fg(DIM) }),
            Span::styled(format!(" {valstr}"), style),
        ]));
        let x = rows[3].x + label_w;
        hits.cells.push((id, Rect::new(x, rows[3].y + mi as u16, bar_w, 1)));
    }
    drop(hits);
    f.render_widget(Paragraph::new(mlines), rows[3]);

    // midi / learn line
    let ch = if app.midi_channel == 0 { "Omni".to_string() } else { app.midi_channel.to_string() };
    let mut ml: Vec<Span> = vec![
        Span::styled(" MIDI ", Style::default().fg(DIM)),
        Span::styled(format!("ch:{ch} "), Style::default().fg(IDLE)),
        Span::styled(format!("· {} ", app.midi_port), Style::default().fg(DIM)),
    ];
    if app.learn {
        ml.push(Span::styled(format!("· ◉ LEARN {} (send CC) ", control_name(app.cur_id())), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)));
    } else if let Some(t) = &app.toast {
        ml.push(Span::styled(format!("· {t} "), Style::default().fg(ACCENT)));
    } else {
        ml.push(Span::styled(format!("· {} CC maps ", app.cc_map.len()), Style::default().fg(DIM)));
        if let Some(m) = &app.last_midi {
            ml.push(Span::styled(format!("· {m}"), Style::default().fg(DIM)));
        }
    }
    f.render_widget(Paragraph::new(Line::from(ml)), rows[4]);

    f.render_widget(Block::default().borders(Borders::TOP).border_style(Style::default().fg(DIM)), rows[5]);

    let hint = "a/s/d (C/D/E) trigger · ←→ drum · Tab/↑↓ row · -/= or drag/scroll adjust · Enter learn · m MIDI · Esc quit";
    f.render_widget(Paragraph::new(Span::styled(hint, Style::default().fg(DIM))).alignment(Alignment::Center), rows[6]);
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
    let crush = move |x: f32, levels: f32| if levels >= 1.0 { (x * levels).round() / levels } else { x };
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let levels = quant.value();
            for frame in data.chunks_mut(channels) {
                let (l, r) = next();
                let (l, r) = (crush(l, levels), crush(r, levels));
                for (i, sample) in frame.iter_mut().enumerate() {
                    *sample = if i & 1 == 0 { T::from_sample(l) } else { T::from_sample(r) };
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
    let _ = execute!(out, DisableMouseCapture);
    let _ = execute!(out, LeaveAlternateScreen);
    let _ = disable_raw_mode();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no default output device")?;
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
    let midi_port = _midi_conn.as_ref().map(|(_, n)| n.clone()).unwrap_or_else(|| "no device".into());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        default_hook(info);
    }));
    let mut terminal: Terminal<CrosstermBackend<Stdout>> = Terminal::new(CrosstermBackend::new(stdout))?;

    // defaults: rough kick / snare / hihat
    let drums = [
        DrumParams { p: [0.18, 0.10, 0.20, 0.70, 0.55, 0.65, 0.45, 0.10, 0.10] },
        DrumParams { p: [0.42, 0.25, 0.40, 0.60, 0.20, 0.70, 0.25, 0.70, 0.50] },
        DrumParams { p: [0.72, 0.45, 0.60, 0.20, 0.00, 0.50, 0.12, 0.60, 0.80] },
    ];

    let mut app = App {
        sequencer,
        _net: net,
        drive_sh,
        vol_sh,
        quant,
        bits_idx: 0,
        drums,
        sel_drum: 0,
        sel_row: 0,
        flash: [-1.0; 3],
        cc_map: load_cc_map(),
        learn: false,
        midi_channel: 0,
        midi_port,
        last_midi: None,
        toast: None,
        toast_until: 0.0,
        hits: RefCell::new(Hits::default()),
        clock: Instant::now(),
    };

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut last_draw = -1.0f32;
        loop {
            while let Ok(msg) = midi_rx.try_recv() {
                app.handle_midi(msg);
            }
            let t = app.clock.elapsed().as_secs_f32();
            if app.toast.is_some() && t > app.toast_until {
                app.toast = None;
            }
            if t - last_draw >= 0.033 {
                terminal.draw(|f| ui(f, &app))?;
                last_draw = t;
            }
            if event::poll(Duration::from_millis(3))? {
                match event::read()? {
                    Event::Key(k) => {
                        if handle_key(&mut app, k) {
                            break;
                        }
                    }
                    Event::Mouse(me) => handle_mouse(&mut app, me),
                    _ => {}
                }
            }
        }
        Ok(())
    })();

    restore_terminal();
    result
}
