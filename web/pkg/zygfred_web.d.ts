/* tslint:disable */
/* eslint-disable */

export class Engine {
    free(): void;
    [Symbol.dispose](): void;
    left_ptr(): number;
    constructor(sample_rate: number);
    /**
     * Render `frames` (<= BLOCK) into the internal L/R buffers.
     */
    process(frames: number): void;
    right_ptr(): number;
    /**
     * Bit-crush quantization levels (0 = off, 2048 = 12-bit, ... 4 = 3-bit).
     */
    set_bits_levels(levels: number): void;
    set_comp(v: number): void;
    set_drive(v: number): void;
    set_drum_param(drum: number, param: number, value: number): void;
    set_reverb(v: number): void;
    set_volume(v: number): void;
    trigger(drum: number, vel: number): void;
}

/**
 * Render a dry voice offline and capture a short waveform window (peak-per-bin) for the scope.
 * Runs on the main thread in its own wasm instance — never touches the audio engine.
 */
export function capture_scope(params: Float32Array, sample_rate: number): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_engine_free: (a: number, b: number) => void;
    readonly capture_scope: (a: number, b: number, c: number) => [number, number];
    readonly engine_left_ptr: (a: number) => number;
    readonly engine_new: (a: number) => number;
    readonly engine_process: (a: number, b: number) => void;
    readonly engine_right_ptr: (a: number) => number;
    readonly engine_set_bits_levels: (a: number, b: number) => void;
    readonly engine_set_comp: (a: number, b: number) => void;
    readonly engine_set_drive: (a: number, b: number) => void;
    readonly engine_set_drum_param: (a: number, b: number, c: number, d: number) => void;
    readonly engine_set_reverb: (a: number, b: number) => void;
    readonly engine_set_volume: (a: number, b: number) => void;
    readonly engine_trigger: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
