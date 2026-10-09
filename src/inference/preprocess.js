// JavaScript port of the training preprocessing in Bridge_Corrosion_Detection.ipynb:
//   enhance(image)              -> unsharp mask with cv2.GaussianBlur((5, 5), 0)
//   cv2.resize(image, (256, 256)) -> INTER_LINEAR with OpenCV's fixed-point arithmetic
// Images are tightly packed RGB Uint8Arrays.

export const SIZE = 256;

// cv2.GaussianBlur with ksize 5 and sigma 0 uses OpenCV's fixed binomial kernel.
const KERNEL = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];

// OpenCV's default BORDER_REFLECT_101: -1 -> 1, n -> n - 2.
function reflect(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
  return i;
}

export function rgbaToRgb(rgba, width, height) {
  const rgb = new Uint8Array(width * height * 3);
  for (let p = 0, q = 0; q < rgb.length; p += 4, q += 3) {
    rgb[q] = rgba[p]; rgb[q + 1] = rgba[p + 1]; rgb[q + 2] = rgba[p + 2];
  }
  return rgb;
}

export function enhance(rgb, width, height) {
  const n = width * height * 3;
  const src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = rgb[i] / 255;
  // Separable blur: horizontal into tmp, then vertical while combining.
  const tmp = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = -2; k <= 2; k++) s += KERNEL[k + 2] * src[(row + reflect(x + k, width)) * 3 + c];
        tmp[(row + x) * 3 + c] = s;
      }
    }
  }
  const out = new Uint8Array(n);
  for (let y = 0; y < height; y++) {
    const rows = [-2, -1, 0, 1, 2].map(k => reflect(y + k, height) * width);
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        let blur = 0;
        for (let k = 0; k < 5; k++) blur += KERNEL[k] * tmp[(rows[k] + x) * 3 + c];
        const i = (y * width + x) * 3 + c;
        const v = Math.fround(1.5 * src[i] - 0.5 * Math.fround(blur));
        // np.clip(..., 0, 1) then (x * 255).astype(np.uint8) truncates.
        out[i] = Math.fround(Math.min(1, Math.max(0, v)) * 255) | 0;
      }
    }
  }
  return out;
}

// Source index and fixed-point weights (11 bits) for each destination coordinate, as cv2.resize computes them.
function linearTaps(srcSize, dstSize) {
  const scale = srcSize / dstSize;
  const index = new Int32Array(dstSize), w0 = new Int32Array(dstSize);
  for (let d = 0; d < dstSize; d++) {
    let f = Math.fround((d + 0.5) * scale - 0.5);
    let s = Math.floor(f);
    f -= s;
    if (s < 0) { s = 0; f = 0; }
    if (s >= srcSize - 1) { s = srcSize - 1; f = 0; }
    index[d] = s;
    w0[d] = Math.round((1 - f) * 2048);
  }
  return {index, w0};
}

export function resizeLinear(rgb, width, height, outW = SIZE, outH = SIZE) {
  const xs = linearTaps(width, outW), ys = linearTaps(height, outH);
  // Horizontal pass on every source row that a destination row needs, kept as integers.
  const rowCache = new Map();
  const hrow = (y) => {
    let r = rowCache.get(y);
    if (r) return r;
    r = new Int32Array(outW * 3);
    const base = y * width;
    for (let d = 0; d < outW; d++) {
      const s = xs.index[d], a = xs.w0[d], b = 2048 - a, s1 = Math.min(s + 1, width - 1);
      for (let c = 0; c < 3; c++) r[d * 3 + c] = rgb[(base + s) * 3 + c] * a + rgb[(base + s1) * 3 + c] * b;
    }
    rowCache.set(y, r);
    return r;
  };
  const out = new Uint8Array(outW * outH * 3);
  for (let d = 0; d < outH; d++) {
    const s = ys.index[d], a = ys.w0[d], b = 2048 - a;
    const r0 = hrow(s), r1 = hrow(Math.min(s + 1, height - 1));
    for (let i = 0; i < outW * 3; i++) {
      // OpenCV's VResizeLinear<uchar> rounding.
      const v = (((a * (r0[i] >> 4)) >> 16) + ((b * (r1[i] >> 4)) >> 16) + 2) >> 2;
      out[d * outW * 3 + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    if (rowCache.size > 4) for (const k of rowCache.keys()) if (k < s) rowCache.delete(k);
  }
  return out;
}

// HWC uint8 -> NCHW float32 in [0, 1], matching image.transpose(2, 0, 1) / 255.
export function toTensor(rgb, width = SIZE, height = SIZE) {
  const plane = width * height, out = new Float32Array(plane * 3);
  for (let p = 0; p < plane; p++) {
    out[p] = rgb[p * 3] / 255;
    out[plane + p] = rgb[p * 3 + 1] / 255;
    out[2 * plane + p] = rgb[p * 3 + 2] / 255;
  }
  return out;
}

export function summarize(probability, threshold = 0.5) {
  const mask = new Uint8Array(probability.length);
  let hits = 0;
  for (let i = 0; i < probability.length; i++) if (probability[i] > threshold) { mask[i] = 1; hits++; }
  return {mask, coverage: Math.round(hits / probability.length * 10000) / 100};
}
