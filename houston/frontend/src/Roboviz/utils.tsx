
export const THEME = '#aa33ff'


export function fromNumpyF32(encoding: string): Float32Array {
    return new Float32Array(Uint8Array.fromBase64(encoding).buffer);
}
