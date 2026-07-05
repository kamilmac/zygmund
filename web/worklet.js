// zygfred audio processor: instantiates the wasm engine from a precompiled module (passed via
// processorOptions — the worklet scope has no fetch) and pulls one render quantum per process().
import './worklet-polyfill.js';
import { initSync, Engine } from './pkg/zygfred_web.js';

class ZygfredProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { module } = options.processorOptions;
    const wasm = initSync({ module });
    this.memory = wasm.memory;
    this.engine = new Engine(sampleRate);
    this.port.onmessage = (e) => this.onMessage(e.data);
    this.port.postMessage({ type: 'ready' });
  }

  onMessage(m) {
    switch (m.type) {
      case 'trigger': this.engine.trigger_voice(m.vl, m.vr, m.vel, m.mulL, m.mulR, m.haas, m.len); break;
      case 'drive': this.engine.set_drive(m.value); break;
      case 'reverb': this.engine.set_reverb(m.value); break;
      case 'comp': this.engine.set_comp(m.value); break;
      case 'volume': this.engine.set_volume(m.value); break;
      case 'bits': this.engine.set_bits_levels(m.value); break;
    }
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    const n = out[0].length;
    this.engine.process(n);
    // views are rebuilt every block: wasm memory may grow (voice allocation) and detach them
    const mem = this.memory.buffer;
    out[0].set(new Float32Array(mem, this.engine.left_ptr(), n));
    (out[1] || out[0]).set(new Float32Array(mem, this.engine.right_ptr(), n));
    return true;
  }
}

registerProcessor('zygfred', ZygfredProcessor);
