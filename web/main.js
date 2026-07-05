// zygfred web UI: builds the three-voice param matrix from config, talks to the AudioWorklet
// over its message port, and renders per-voice scopes via a main-thread wasm instance.
import init, { capture_scope } from './pkg/zygfred_web.js';

const PARAMS = ['Tune', 'Ratio', 'FM', 'FMDec', 'PEnv', 'PDec', 'Decay', 'Snap', 'Tone', 'Haas', 'Detune', 'Rand'];
const DRUMS = [
  { name: 'KICK', key: 'A' },
  { name: 'SNARE', key: 'S' },
  { name: 'HIHAT', key: 'D' },
];
const BIT_OPTIONS = [
  ['off', 0], ['12-bit', 2048], ['8-bit', 128], ['6-bit', 32], ['4-bit', 8], ['3-bit', 4],
];
const MASTER = [
  { name: 'Drive', msg: 'drive', value: 0.0 },
  { name: 'Reverb', msg: 'reverb', value: 0.0 },
  { name: 'Comp', msg: 'comp', value: 0.0 },
  { name: 'Bits', msg: 'bits', value: 0.0 }, // value = option index / (len-1)
  { name: 'Volume', msg: 'volume', value: 0.7 },
];
const DEFAULTS = [
  [0.18, 0.10, 0.20, 0.70, 0.55, 0.65, 0.45, 0.10, 0.10, 0.10, 0.10, 0.15],
  [0.42, 0.25, 0.40, 0.60, 0.20, 0.70, 0.25, 0.70, 0.50, 0.25, 0.30, 0.35],
  [0.72, 0.45, 0.60, 0.20, 0.00, 0.50, 0.12, 0.60, 0.80, 0.35, 0.40, 0.40],
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

function makeBar({ getNorm, setNorm, getLabel, discreteSteps }) {
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
    getNorm: () => drums[drum][pi],
    setNorm: (n) => { drums[drum][pi] = n; }, // params are read per hit, at trigger time
    getLabel: () => `${Math.round(drums[drum][pi] * 100)}%`,
  });
  if (pi < 9) {
    // per-hit randomisation markers: where the L/R takes actually landed on the last trigger
    const tl = document.createElement('div');
    tl.className = 'tick tick-l';
    const tr = document.createElement('div');
    tr.className = 'tick tick-r';
    bar.append(tl, tr);
    tickSetters[drum][pi] = (l, r) => {
      tl.style.left = `${l * 100}%`;
      tr.style.left = `${r * 100}%`;
      tl.style.opacity = tr.style.opacity = 1;
    };
  }
  row.append(label, bar, val);
  return row;
}

// ---------- scopes ----------

const scopeCanvases = [];

function drawScope(drum, vl) {
  const canvas = scopeCanvases[drum];
  if (!canvas || !audioCtx) return;
  const g = canvas.getContext('2d');
  const { width: w, height: h } = canvas;
  g.clearRect(0, 0, w, h);
  const take = vl ? new Float32Array(vl) : new Float32Array(drums[drum].slice(0, 9));
  const pts = capture_scope(take, audioCtx.sampleRate);
  g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent');
  g.lineWidth = 1.5;
  g.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const x = (i / (pts.length - 1)) * w;
    const y = h / 2 - pts[i] * (h / 2 - 2);
    i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
  }
  g.stroke();
}

// ---------- voices ----------

const rnd = () => Math.random() * 2 - 1;
const clamp01 = (v) => Math.min(1, Math.max(0, v));

function trigger(drum, vel = 0.9) {
  if (!node) return;
  // per-hit randomisation happens here, not in the engine, so the UI can show where the
  // L/R takes actually landed (slider ticks) and render this hit's real waveform (scope)
  const p = drums[drum];
  const rand = p[11];
  const vl = new Float32Array(9);
  const vr = new Float32Array(9);
  for (let i = 0; i < 9; i++) {
    vl[i] = clamp01(p[i] + rand * 0.15 * rnd());
    vr[i] = clamp01(p[i] + rand * 0.15 * rnd());
  }
  const det = p[10] * 40; // up to 40 cents L/R spread
  send({
    type: 'trigger', vl, vr, vel,
    mulL: 2 ** (-det / 2 / 1200),
    mulR: 2 ** (det / 2 / 1200),
    haas: p[9] * 0.03, // 0..30 ms inter-channel delay
    len: 0.08 + 1.8 * p[6],
  });
  for (let i = 0; i < 9; i++) tickSetters[drum][i]?.(vl[i], vr[i]);
  drawScope(drum, vl);
  const panel = document.querySelectorAll('.voice')[drum];
  panel.classList.remove('hit');
  void panel.offsetWidth; // restart the flash animation
  panel.classList.add('hit');
}

function buildVoice(drum) {
  const panel = document.createElement('section');
  panel.className = 'voice';
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
  canvas.width = 240;
  canvas.height = 56;
  scopeCanvases[drum] = canvas;
  panel.appendChild(canvas);

  for (let pi = 6; pi < PARAMS.length; pi++) panel.appendChild(paramRow(drum, pi));
  return panel;
}

// ---------- master strip ----------

function buildMaster() {
  const strip = $('#master');
  for (const m of MASTER) {
    const cell = document.createElement('div');
    cell.className = 'row';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = m.name;
    const isBits = m.msg === 'bits';
    const { bar, val } = makeBar({
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
      getLabel: () => (isBits ? BIT_OPTIONS[bitsIdx][0] : `${Math.round(m.value * 100)}%`),
    });
    cell.append(label, bar, val);
    strip.appendChild(cell);
  }
}

// ---------- boot ----------

async function powerOn() {
  audioCtx = new AudioContext();
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
  // engine defaults match DEFAULTS/MASTER; only volume is non-zero
  send({ type: 'volume', value: MASTER[4].value });

  $('#power').remove();
  for (let d = 0; d < 3; d++) drawScope(d);
  window.zyg = { ctx: audioCtx, node, trigger }; // debug/inspection surface
}

document.addEventListener('keydown', (e) => {
  if (e.repeat || e.metaKey || e.ctrlKey) return;
  const st = KEY_SEMITONE[e.key.toLowerCase()];
  if (st === undefined) return;
  const drum = SEMITONE_DRUM[((st % 12) + 12) % 12];
  if (drum !== undefined) trigger(drum);
});

const voices = $('#voices');
for (let d = 0; d < 3; d++) voices.appendChild(buildVoice(d));
buildMaster();
$('#power button').addEventListener('click', () => powerOn().catch((err) => {
  $('#power .hint').textContent = `failed to start: ${err.message}`;
  console.error(err);
}));
