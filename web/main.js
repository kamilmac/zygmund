// zygfred web UI: builds the three-voice param matrix from config, talks to the AudioWorklet
// over its message port, and renders per-voice scopes via a main-thread wasm instance.
import init, { capture_envelope, capture_spectrogram } from './pkg/zygfred_web.js';

const PARAMS = ['Tune', 'Ratio', 'FM', 'FMDec', 'PEnv', 'PDec', 'Decay', 'Snap', 'Tone', 'Haas', 'Rand', 'Width', 'Vol'];
const P_HAAS = 9;
const P_RAND = 10;
const P_WIDTH = 11; // 0 = mono (left take on both channels), 1 = full stereo (L take left, R take right)
const P_VOL = 12; // per-voice gain, applied on the velocity path (voice amp is linear in vel)
const DRUMS = [
  { name: 'KICK', key: 'A' },
  { name: 'SNARE', key: 'S' },
  { name: 'HIHAT', key: 'D' },
];
const BIT_OPTIONS = [
  ['off', 0], ['12bit', 2048], ['8bit', 128], ['6bit', 32], ['4bit', 8], ['3bit', 4],
];
const MASTER = [
  { name: 'Drive', msg: 'drive', value: 0.0 },
  { name: 'Reverb', msg: 'reverb', value: 0.0 },
  { name: 'Comp', msg: 'comp', value: 0.0 },
  { name: 'Bits', msg: 'bits', value: 0.0 }, // value = option index / (len-1)
  { name: 'Volume', msg: 'volume', value: 0.7 },
];
const DEFAULTS = [
  [0.18, 0.10, 0.20, 0.70, 0.55, 0.65, 0.45, 0.10, 0.10, 0.10, 0.15, 1.00, 1.00],
  [0.42, 0.25, 0.40, 0.60, 0.20, 0.70, 0.25, 0.70, 0.50, 0.25, 0.35, 1.00, 1.00],
  [0.72, 0.45, 0.60, 0.20, 0.00, 0.50, 0.12, 0.60, 0.80, 0.35, 0.40, 1.00, 1.00],
];
// same trigger mapping as native: keys are semitones, pitch classes C/D/E hit kick/snare/hihat
const KEY_SEMITONE = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12 };
const SEMITONE_DRUM = { 0: 0, 2: 1, 4: 2 };

const drums = DEFAULTS.map((p) => [...p]);
let bitsIdx = 0;
let audioCtx = null;
let node = null;

const $ = (sel, el = document) => el.querySelector(sel);

function send(msg) {
  if (node) node.port.postMessage(msg);
}

// ---------- generic bar control ----------

// registry of every learnable control, indexed by id: drum params first (drum*13+param,
// incl. per-voice Vol), then master
const controls = [];

function makeBar({ id, name, getNorm, setNorm, getLabel, discreteSteps }) {
  const bar = document.createElement('div');
  bar.className = 'bar';
  const fill = document.createElement('div');
  fill.className = 'fill';
  bar.appendChild(fill);
  const val = document.createElement('span');
  val.className = 'val';

  const refresh = () => {
    fill.style.width = `${getNorm() * 100}%`;
    val.textContent = getLabel();
  };
  const setFromEvent = (e) => {
    const r = bar.getBoundingClientRect();
    let n = (e.clientX - r.left) / r.width;
    if (discreteSteps) n = Math.round(n * (discreteSteps - 1)) / (discreteSteps - 1);
    setNorm(Math.min(1, Math.max(0, n)));
    refresh();
  };
  bar.addEventListener('pointerdown', (e) => {
    if (id !== undefined && tryArmLearn(id)) return; // learn mode: pick this control, don't set it
    bar.setPointerCapture(e.pointerId);
    setFromEvent(e);
  });
  bar.addEventListener('pointermove', (e) => {
    if (bar.hasPointerCapture(e.pointerId)) setFromEvent(e);
  });
  bar.addEventListener('wheel', (e) => {
    e.preventDefault();
    const step = discreteSteps ? 1 / (discreteSteps - 1) : 0.05;
    const d = e.deltaY < 0 ? step : -step;
    setNorm(Math.min(1, Math.max(0, getNorm() + d)));
    refresh();
  }, { passive: false });

  refresh();
  if (id !== undefined) {
    controls[id] = {
      apply: (n) => {
        setNorm(Math.min(1, Math.max(0, n)));
        refresh();
      },
      bar,
      name,
    };
  }
  return { bar, val, refresh };
}

const tickSetters = [[], [], []]; // [drum][param] -> (l, r), sound params only

function paramRow(drum, pi) {
  const row = document.createElement('div');
  row.className = 'row';
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = PARAMS[pi];
  const { bar, val } = makeBar({
    id: drum * PARAMS.length + pi,
    name: `${DRUMS[drum].name} ${PARAMS[pi]}`,
    getNorm: () => drums[drum][pi],
    setNorm: (n) => { drums[drum][pi] = n; }, // params are read per hit, at trigger time
    getLabel: () => `${Math.round(drums[drum][pi] * 100)}`,
  });
  if (pi < 9) {
    // per-hit randomisation marker: where the left take actually landed on the last trigger
    const tick = document.createElement('div');
    tick.className = 'tick';
    bar.append(tick);
    tickSetters[drum][pi] = (l) => {
      tick.style.transition = 'none';
      tick.style.left = `${l * 100}%`;
      tick.style.opacity = 1;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        tick.style.transition = 'opacity 1.2s ease-out 0.5s';
        tick.style.opacity = 0;
      }));
    };
  }
  row.append(label, bar, val);
  return row;
}

// ---------- scopes ----------

const scopeCanvases = [];

const SPEC_COLS = 256;
const SPEC_ROWS = 56;

// scope of the hit: min/max amplitude band (DAW-style)
function drawScope(drum, vl) {
  const canvas = scopeCanvases[drum];
  if (!canvas || !audioCtx) return;
  const take = vl ? new Float32Array(vl) : new Float32Array(drums[drum].slice(0, 9));
  // time span = where the amp envelope reaches -60dB (dec rate = 40 - 37*decay, env = e^-t*dec)
  const decRate = 40 - 37 * take[6];
  const seconds = Math.min(1.2, Math.max(0.15, 6.9 / decRate));
  const env = capture_envelope(take, audioCtx.sampleRate, SPEC_COLS, seconds);
  const g = canvas.getContext('2d');
  const { width: w, height: h } = canvas;
  g.clearRect(0, 0, w, h);
  const accent = getComputedStyle(canvas.parentElement).getPropertyValue('--accent');

  // amplitude band: fill between per-column min and max
  const mid = h / 2;
  const yAmp = (v) => mid - v * (mid - 2);
  g.beginPath();
  for (let c = 0; c < SPEC_COLS; c++) {
    const x = (c / (SPEC_COLS - 1)) * w;
    c === 0 ? g.moveTo(x, yAmp(env[2 * c + 1])) : g.lineTo(x, yAmp(env[2 * c + 1]));
  }
  for (let c = SPEC_COLS - 1; c >= 0; c--) {
    g.lineTo((c / (SPEC_COLS - 1)) * w, yAmp(env[2 * c]));
  }
  g.closePath();
  g.globalAlpha = 0.55;
  g.fillStyle = accent;
  g.fill();
  g.globalAlpha = 1;
}

// ---------- voices ----------

const rnd = () => Math.random() * 2 - 1;
const clamp01 = (v) => Math.min(1, Math.max(0, v));

function trigger(drum, vel = 0.9) {
  if (!node) return;
  // per-hit randomisation happens here, not in the engine, so the UI can show where the
  // L/R takes actually landed (slider ticks) and render this hit's real waveform (scope)
  const p = drums[drum];
  const rand = p[P_RAND];
  const vl = new Float32Array(9);
  const vr = new Float32Array(9);
  for (let i = 0; i < 9; i++) {
    vl[i] = clamp01(p[i] + rand * 0.15 * rnd());
    vr[i] = clamp01(p[i] + rand * 0.15 * rnd());
  }
  send({
    type: 'trigger', vl, vr, vel: vel * p[P_VOL],
    haas: p[P_HAAS] * 0.03, // 0..30 ms inter-channel delay
    width: p[P_WIDTH],
    len: 0.08 + 1.8 * p[6],
  });
  for (let i = 0; i < 9; i++) tickSetters[drum][i]?.(vl[i]);
  drawScope(drum, vl);
  const panel = document.querySelectorAll('.voice')[drum];
  panel.classList.remove('hit');
  void panel.offsetWidth; // restart the flash animation
  panel.classList.add('hit');
}

function buildVoice(drum) {
  const panel = document.createElement('section');
  panel.className = 'voice';
  panel.dataset.voice = drum;
  const pad = document.createElement('button');
  pad.className = 'pad';
  pad.innerHTML = `${DRUMS[drum].name} <span class="key">${DRUMS[drum].key}</span>`;
  pad.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    trigger(drum);
  });
  panel.appendChild(pad);

  for (let pi = 0; pi < 6; pi++) panel.appendChild(paramRow(drum, pi));

  const canvas = document.createElement('canvas');
  canvas.className = 'scope';
  canvas.width = 512;
  canvas.height = 112;
  scopeCanvases[drum] = canvas;
  panel.appendChild(canvas);

  for (let pi = 6; pi < PARAMS.length; pi++) {
    const row = paramRow(drum, pi);
    if (pi === P_HAAS) row.classList.add('sect'); // body | stereo+gain
    panel.appendChild(row);
  }
  return panel;
}

// ---------- master strip ----------

function buildMaster() {
  const strip = $('#master');
  for (const [mi, m] of MASTER.entries()) {
    const cell = document.createElement('div');
    cell.className = 'row';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = m.name;
    const isBits = m.msg === 'bits';
    const { bar, val } = makeBar({
      id: DRUMS.length * PARAMS.length + mi,
      name: m.name,
      discreteSteps: isBits ? BIT_OPTIONS.length : 0,
      getNorm: () => (isBits ? bitsIdx / (BIT_OPTIONS.length - 1) : m.value),
      setNorm: (n) => {
        if (isBits) {
          bitsIdx = Math.round(n * (BIT_OPTIONS.length - 1));
          send({ type: 'bits', value: BIT_OPTIONS[bitsIdx][1] });
        } else {
          m.value = n;
          send({ type: m.msg, value: n });
        }
      },
      getLabel: () => (isBits ? BIT_OPTIONS[bitsIdx][0] : `${Math.round(m.value * 100)}`),
    });
    cell.append(label, bar, val);
    strip.appendChild(cell);
  }
}

// ---------- MIDI (Web MIDI: Digitakt note triggers + CC learn + channel filter) ----------

let midiAccess = null;
let midiChannel = +(localStorage.getItem('zygfred-midi-ch') || 0); // 0 = omni
let ccMap = {}; // cc number -> control id
try { ccMap = JSON.parse(localStorage.getItem('zygfred-cc') || '{}'); } catch { /* fresh map */ }
let learn = null; // null | 'pick' (waiting for a slider click) | control id (waiting for a CC)
let toastTimer = null;
const midiEls = {};

function saveCcMap() {
  localStorage.setItem('zygfred-cc', JSON.stringify(ccMap));
}

function refreshMidiStatus() {
  if (!midiEls.status) return;
  if (!midiAccess) return;
  const names = [...midiAccess.inputs.values()].map((p) => p.name);
  const port = names.length ? names.join(' · ') : 'no device';
  midiEls.status.textContent = `${port} · ${Object.keys(ccMap).length} CC maps`;
}

function toast(text) {
  clearTimeout(toastTimer);
  midiEls.status.classList.add('accent');
  midiEls.status.textContent = text;
  toastTimer = setTimeout(() => {
    midiEls.status.classList.remove('accent');
    refreshMidiStatus();
  }, 2000);
}

function tryArmLearn(id) {
  if (learn !== 'pick') return false;
  learn = id;
  controls[id].bar.classList.add('armed');
  midiEls.status.textContent = `turn a knob \u2192 ${controls[id].name}`;
  return true;
}

function disarmLearn() {
  if (typeof learn === 'number') controls[learn]?.bar.classList.remove('armed');
  learn = null;
  midiEls.learnBtn?.classList.remove('on');
  refreshMidiStatus();
}

function onMidiMessage(e) {
  if (e.data.length < 3) return;
  if (audioCtx && audioCtx.state === 'suspended') $('#locked').hidden = false;
  const [status, d1, d2] = e.data;
  const ch = status & 0x0f;
  if (midiChannel !== 0 && midiChannel !== ch + 1) return;
  const type = status & 0xf0;
  if (type === 0x90 && d2 > 0) {
    const drum = SEMITONE_DRUM[d1 % 12]; // any C/D/E pitch class, like native
    if (drum !== undefined) trigger(drum, d2 / 127);
  } else if (type === 0xb0) {
    if (typeof learn === 'number') {
      const id = learn;
      ccMap[d1] = id;
      saveCcMap();
      disarmLearn();
      toast(`CC${d1} → ${controls[id].name}`);
    } else if (ccMap[d1] !== undefined) {
      controls[ccMap[d1]]?.apply(d2 / 127);
    }
  }
}

function attachMidiInputs() {
  for (const input of midiAccess.inputs.values()) input.onmidimessage = onMidiMessage;
  refreshMidiStatus();
}

async function initMidi() {
  if (!navigator.requestMIDIAccess) {
    midiEls.status.textContent = 'Web MIDI not supported in this browser';
    midiEls.learnBtn.disabled = true;
    return;
  }
  try {
    midiAccess = await navigator.requestMIDIAccess();
  } catch {
    midiEls.status.textContent = 'MIDI access denied';
    return;
  }
  midiAccess.onstatechange = attachMidiInputs; // hot-plug: Digitakt can arrive later
  attachMidiInputs();
}

function buildMidiStrip() {
  const strip = $('#midi');
  const tag = document.createElement('span');
  tag.className = 'tag';
  tag.textContent = 'MIDI';

  const chSel = document.createElement('select');
  ['Omni', ...Array.from({ length: 16 }, (_, i) => `ch ${i + 1}`)].forEach((t, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = t;
    chSel.append(o);
  });
  chSel.value = midiChannel;
  chSel.addEventListener('change', () => {
    midiChannel = +chSel.value;
    localStorage.setItem('zygfred-midi-ch', midiChannel);
  });

  const learnBtn = document.createElement('button');
  learnBtn.className = 'learn';
  learnBtn.textContent = 'learn';
  learnBtn.addEventListener('click', () => {
    if (learn !== null) {
      disarmLearn();
      return;
    }
    learn = 'pick';
    learnBtn.classList.add('on');
    midiEls.status.textContent = 'click a slider\u2026';
  });

  const status = document.createElement('span');
  status.className = 'status';
  status.textContent = 'click or play to enable';

  midiEls.learnBtn = learnBtn;
  midiEls.status = status;
  strip.append(tag, chSel, learnBtn, status);
}

// ---------- theme (dev modal: 2 base colors, every other shade derived) ----------

const THEME_KEY = 'zygfred-theme';
const THEME_DEFAULT = {
  surface: '#0b0b0b',
  accent: '#f0eeea', // master + chassis (title, power, learn)
  kick: '#7dc9d1',
  snare: '#b8486d',
  hihat: '#d8bf5f',
};
const VOICE_KEYS = ['kick', 'snare', 'hihat'];

function hexToHsl(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = ((n >> 16) & 255) / 255;
  const g = ((n >> 8) & 255) / 255;
  const b = (n & 255) / 255;
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l * 100];
  const d = mx - mn;
  const s = d / (l > 0.5 ? 2 - mx - mn : mx + mn);
  let h;
  if (mx === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (mx === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return [h * 60, s * 100, l * 100];
}

function applyTheme(theme) {
  const [h, s, l] = hexToHsl(theme.surface);
  // derived shades: lightness steps for chrome, desaturated lifts for text
  const shade = (dl, ss = s) => `hsl(${h.toFixed(0)} ${ss.toFixed(0)}% ${Math.min(96, Math.max(0, l + dl)).toFixed(1)}%)`;
  const text = (ll) => `hsl(${h.toFixed(0)} 13% ${ll}%)`;
  const root = document.documentElement.style;
  root.setProperty('--bg', shade(0));
  root.setProperty('--panel', shade(3));
  root.setProperty('--inset', shade(-1.5));
  root.setProperty('--track', shade(8.5));
  root.setProperty('--faint', text(32));
  root.setProperty('--dim', text(42));
  root.setProperty('--idle', text(69));
  root.setProperty('--accent', theme.accent);
  VOICE_KEYS.forEach((k, i) => root.setProperty(`--voice${i}`, theme[k]));
}

function buildDevModal() {
  const modal = $('#dev');
  const fields = ['surface', 'kick', 'snare', 'hihat', 'accent'];
  const inputs = Object.fromEntries(fields.map((f) => [f, $(`#dev-${f}`)]));
  let theme = { ...THEME_DEFAULT };
  try {
    const t = JSON.parse(localStorage.getItem(THEME_KEY) || '{}');
    if (t.kick) theme = { ...theme, ...t };
    else if (t.surface) theme.surface = t.surface; // pre-per-voice schema: keep surface only
  } catch { /* defaults */ }

  const refresh = () => {
    fields.forEach((f) => { inputs[f].value = theme[f]; });
    applyTheme(theme);
    if (audioCtx) for (let d = 0; d < 3; d++) drawScope(d); // scopes read their voice accent
  };
  const update = () => {
    fields.forEach((f) => { theme[f] = inputs[f].value; });
    localStorage.setItem(THEME_KEY, JSON.stringify(theme));
    refresh();
  };
  refresh();
  fields.forEach((f) => inputs[f].addEventListener('input', update));
  $('#dev-reset').addEventListener('click', () => {
    theme = { ...THEME_DEFAULT };
    localStorage.setItem(THEME_KEY, JSON.stringify(theme));
    refresh(); // refresh writes theme -> inputs; update() would read stale inputs back
  });
  $('#dev-open').addEventListener('click', () => { modal.hidden = !modal.hidden; });
}

function buildHelp() {
  const help = $('#help');
  $('#help-open').addEventListener('click', () => { help.hidden = false; });
  help.addEventListener('pointerdown', (e) => {
    if (e.target === help) help.hidden = true; // backdrop click closes; the sheet doesn't
  });
}

// ---------- presets (hold a slot to save, click / keys 1-8 to load) ----------

const PRESET_KEY = 'zygfred-presets';
const PRESET_HELP = 'hold to save \u00b7 click or 1\u20138 to load';
const PRESET_SLOTS = 8;
let presets = {};
try { presets = JSON.parse(localStorage.getItem(PRESET_KEY) || '{}'); } catch { /* fresh */ }
let currentSlot = null;
const slotSyncs = [];
let presetHintTimer = null;

function snapshotState() {
  return { drums: drums.map((p) => [...p]), master: MASTER.map((mm) => mm.value), bits: bitsIdx };
}

function applyState(st) {
  st.drums.forEach((p, d) => p.forEach((v, i) => controls[d * PARAMS.length + i]?.apply(v)));
  const base = DRUMS.length * PARAMS.length;
  st.master.forEach((v, mi) => { if (mi !== 3) controls[base + mi]?.apply(v); });
  controls[base + 3]?.apply(st.bits / (BIT_OPTIONS.length - 1));
  if (audioCtx) for (let d = 0; d < 3; d++) drawScope(d);
}

function presetHint(text, sticky) {
  const hint = $('#presets .hint');
  clearTimeout(presetHintTimer);
  hint.textContent = text;
  hint.classList.add('accent');
  if (!sticky) {
    presetHintTimer = setTimeout(() => {
      hint.textContent = PRESET_HELP;
      hint.classList.remove('accent');
    }, 1600);
  }
}

function loadPreset(i) {
  if (!presets[i]) return;
  applyState(presets[i]);
  currentSlot = i;
  slotSyncs.forEach((f) => f());
  presetHint(`loaded ${i}`);
}

function buildPresets() {
  const strip = $('#presets');
  const tag = document.createElement('span');
  tag.className = 'tag';
  tag.textContent = 'PRESET';
  strip.append(tag);
  for (let i = 1; i <= PRESET_SLOTS; i++) {
    const b = document.createElement('button');
    b.className = 'slot';
    b.textContent = i;
    const sync = () => {
      b.classList.toggle('filled', !!presets[i]);
      b.classList.toggle('active', currentSlot === i);
    };
    slotSyncs.push(sync);
    let hold = null;
    let held = false;
    const disarm = () => {
      clearTimeout(hold);
      b.classList.remove('arming');
    };
    b.addEventListener('pointerdown', () => {
      held = false;
      b.classList.add('arming'); // charges toward accent for the hold duration
      hold = setTimeout(() => {
        held = true;
        presets[i] = snapshotState();
        localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
        currentSlot = i;
        slotSyncs.forEach((f) => f());
        b.classList.remove('arming');
        presetHint(`saved ${i}`);
      }, 600);
    });
    b.addEventListener('pointerup', () => {
      disarm();
      if (held) return;
      if (presets[i]) loadPreset(i);
      else presetHint(`slot ${i} empty · hold to save`);
    });
    b.addEventListener('pointerleave', disarm);
    sync();
    strip.append(b);
  }
  const hint = document.createElement('span');
  hint.className = 'hint';
  hint.textContent = PRESET_HELP;
  strip.append(hint);
}

// ---------- boot ----------

async function boot() {
  audioCtx = new AudioContext(); // allowed pre-gesture; starts suspended, resumed by first input
  const bytes = await (await fetch('./pkg/zygfred_web_bg.wasm')).arrayBuffer();
  const module = await WebAssembly.compile(bytes);
  await init({ module_or_path: module }); // main-thread instance, scope rendering only
  await audioCtx.audioWorklet.addModule('./worklet.js');
  node = new AudioWorkletNode(audioCtx, 'zygfred', {
    numberOfInputs: 0,
    outputChannelCount: [2],
    processorOptions: { module },
  });
  await new Promise((resolve) => {
    node.port.onmessage = (e) => e.data.type === 'ready' && resolve();
  });
  node.connect(audioCtx.destination);
  // push full master state — the URL may have loaded a non-default patch
  send({ type: 'drive', value: MASTER[0].value });
  send({ type: 'reverb', value: MASTER[1].value });
  send({ type: 'comp', value: MASTER[2].value });
  send({ type: 'bits', value: BIT_OPTIONS[bitsIdx][1] });
  send({ type: 'volume', value: MASTER[4].value });

  for (let d = 0; d < 3; d++) drawScope(d);
  window.zyg = { ctx: audioCtx, node, trigger, onMidiMessage, capture: (p, secs = 0.6) => capture_spectrogram(new Float32Array(p), audioCtx.sampleRate, SPEC_COLS, SPEC_ROWS, secs) }; // debug/inspection surface
}

// browsers require one user gesture before audio can run — hijack the first natural one
let midiStarted = false;
function wake() {
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
  $('#locked').hidden = true;
  if (!midiStarted && audioCtx) {
    midiStarted = true;
    initMidi();
  }
}
document.addEventListener('pointerdown', wake, { capture: true });
document.addEventListener('keydown', wake, { capture: true });

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    disarmLearn();
    $('#help').hidden = true;
    return;
  }
  if (e.repeat || e.metaKey || e.ctrlKey || e.target.tagName === 'SELECT') return;
  if (e.key >= '1' && e.key <= String(PRESET_SLOTS)) {
    loadPreset(+e.key);
    return;
  }
  const st = KEY_SEMITONE[e.key.toLowerCase()];
  if (st === undefined) return;
  const drum = SEMITONE_DRUM[((st % 12) + 12) % 12];
  if (drum !== undefined) trigger(drum);
});

const voices = $('#voices');
for (let d = 0; d < 3; d++) voices.appendChild(buildVoice(d));
buildMaster();
buildMidiStrip();
buildPresets();
buildDevModal();
buildHelp();
boot().catch((err) => console.error('boot failed:', err));
