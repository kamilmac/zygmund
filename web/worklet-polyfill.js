// AudioWorkletGlobalScope has no TextDecoder/TextEncoder; the wasm-bindgen glue instantiates
// them at module top level. The engine API is numeric-only, so inert stubs are enough.
if (typeof globalThis.TextDecoder === 'undefined') {
  globalThis.TextDecoder = class {
    decode() { return ''; }
  };
}
if (typeof globalThis.TextEncoder === 'undefined') {
  globalThis.TextEncoder = class {
    encode() { return new Uint8Array(0); }
    encodeInto() { return { read: 0, written: 0 }; }
  };
}
