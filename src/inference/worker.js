import * as ort from 'onnxruntime-web/webgpu';
import {SIZE, rgbaToRgb, enhance, resizeLinear, toTensor, summarize} from './preprocess.js';

const CACHE = 'bridgeguard-model-v1';
const MAX_PIXELS = 25_000_000;
let session = null, backend = null, loading = null;

async function download(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch {}
  const cached = cache && await cache.match(url);
  if (cached) { onProgress(1); return new Uint8Array(await cached.arrayBuffer()); }

  const response = await fetch(url);
  if (!response.ok) throw new Error(response.status === 404 ? 'MODEL_MISSING' : `Model download failed (HTTP ${response.status}).`);
  const total = +response.headers.get('content-length') || 0;
  const reader = response.body.getReader(), chunks = [];
  let received = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress(Math.min(0.99, received / total));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  // A failed cache write only means the next visit downloads again.
  try { await cache?.put(url, new Response(bytes, {headers: {'content-type': 'application/octet-stream'}})); } catch {}
  onProgress(1);
  return bytes;
}

async function load(url, useGpu) {
  const bytes = await download(url, (p) => postMessage({type: 'progress', value: p}));
  if (useGpu) {
    // The client restarts this worker in CPU mode if anything here fails or hangs; some drivers never settle.
    postMessage({type: 'stage', value: 'Starting on the GPU'});
    if (!await navigator.gpu?.requestAdapter().catch(() => null)) throw new Error('GPU_UNAVAILABLE');
    try {
      session = await ort.InferenceSession.create(bytes, {executionProviders: ['webgpu']});
      // Some GPUs accept the model but fail on the first run, so warm it up here.
      // The warm-up also compiles the shaders, which makes the first real photo faster.
      await session.run({[session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE])});
    } catch (e) {
      session = null;
      throw new Error('GPU_UNAVAILABLE');
    }
    backend = 'WebGPU';
    return;
  }
  postMessage({type: 'stage', value: 'Starting on the CPU'});
  try {
    session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm']});
  } catch (e) {
    throw new Error(`The model could not start in this browser (${e.message}).`);
  }
  backend = 'WebAssembly';
}

async function detect(file) {
  const bitmap = await createImageBitmap(file);
  const {width, height} = bitmap;
  if (width * height > MAX_PIXELS) { bitmap.close(); throw new Error('Image exceeds 25 megapixels.'); }
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgb = rgbaToRgb(ctx.getImageData(0, 0, width, height).data, width, height);
  const input = toTensor(resizeLinear(enhance(rgb, width, height), width, height));

  const started = performance.now();
  const feeds = {[session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE])};
  const output = (await session.run(feeds))[session.outputNames[0]];
  const probability = await output.getData();
  const ms = Math.round(performance.now() - started);

  const {mask, coverage} = summarize(probability);
  const overlay = new OffscreenCanvas(SIZE, SIZE);
  const image = new ImageData(SIZE, SIZE);
  for (let i = 0; i < mask.length; i++) if (mask[i]) image.data.set([235, 121, 58, 210], i * 4);
  overlay.getContext('2d').putImageData(image, 0, 0);
  const png = await overlay.convertToBlob({type: 'image/png'});
  return {coverage, mask: new FileReaderSync().readAsDataURL(png), backend, ms};
}

// ONNX Runtime starts its CPU threads from this same script; leave those workers to the runtime.
if (!self.name?.startsWith('em-pthread')) onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      loading ??= load(data.url, data.gpu).catch(e => { loading = null; throw e; });
      await loading;
      postMessage({type: 'ready', id: data.id, backend});
    } else if (data.type === 'detect') {
      await loading;
      if (!session) throw new Error('The model is not loaded yet.');
      postMessage({type: 'result', id: data.id, result: await detect(data.file)});
    }
  } catch (e) {
    postMessage({type: 'error', id: data.id, message: e.message || String(e)});
  }
};
