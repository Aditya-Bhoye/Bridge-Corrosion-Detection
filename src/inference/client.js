// Runs the U-Net in the visitor's browser through a Web Worker, so nothing is uploaded and no server is needed.

export const MODEL_URL = import.meta.env.VITE_MODEL_URL || `${import.meta.env.BASE_URL}models/corrosion_unet.onnx`;

const GPU_TIMEOUT = 20000;
const GPU_FAILED_KEY = 'bridgeguard-webgpu-failed';

let worker = null, nextId = 0;
const pending = new Map();
const listeners = new Set();

function emit(value, type) { listeners.forEach(fn => fn(value, type)); }

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});
  worker.onmessage = ({data}) => {
    if (data.type === 'progress' || data.type === 'stage') { emit(data.value, data.type); return; }
    const job = pending.get(data.id);
    if (!job) return;
    pending.delete(data.id);
    data.type === 'error' ? job.reject(new Error(data.message)) : job.resolve(data);
  };
  worker.onerror = (e) => {
    pending.forEach(job => job.reject(new Error(e.message || 'The in-browser model crashed. Reload the page and try again.')));
    pending.clear();
    worker = null;
  };
  return worker;
}

function restartWorker() {
  worker?.terminate();
  worker = null;
  pending.forEach(job => job.reject(new Error('restarted')));
  pending.clear();
}

function call(message) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    getWorker().postMessage({...message, id});
  });
}

// A device whose GPU failed goes straight to the CPU for a week, then tries the GPU again.
function gpuFailedRecently() {
  try { return Date.now() - Number(localStorage.getItem(GPU_FAILED_KEY) || 0) < 7 * 24 * 3600 * 1000; } catch { return false; }
}

function rememberGpuFailure() {
  try { localStorage.setItem(GPU_FAILED_KEY, String(Date.now())); } catch {}
}

// Some GPU drivers never finish starting, so time the GPU start-up (not the download) and give up after a while.
function loadOnGpu() {
  return new Promise((resolve, reject) => {
    let timer = null;
    const off = onProgress((value, type) => {
      if (type === 'stage' && value === 'Starting on the GPU' && !timer) timer = setTimeout(() => reject(new Error('GPU_UNAVAILABLE')), GPU_TIMEOUT);
    });
    call({type: 'load', url: MODEL_URL, gpu: true})
      .then(d => resolve(d.backend), reject)
      .finally(() => { clearTimeout(timer); off(); });
  });
}

// Cheap check that the model file is deployed, without downloading it.
export async function modelAvailable() {
  try {
    const r = await fetch(MODEL_URL, {method: 'HEAD'});
    return r.ok;
  } catch { return false; }
}

export function onProgress(fn) { listeners.add(fn); return () => listeners.delete(fn); }

async function start() {
  if ('gpu' in navigator && !gpuFailedRecently()) {
    try {
      return await loadOnGpu();
    } catch (e) {
      if (e.message !== 'GPU_UNAVAILABLE') throw e;
      rememberGpuFailure();
      // A stuck GPU start can leave the runtime unusable, so continue in a fresh worker.
      restartWorker();
    }
  }
  return (await call({type: 'load', url: MODEL_URL, gpu: false})).backend;
}

let loaded = null;
export function loadModel() {
  loaded ??= start().catch(e => { loaded = null; throw e; });
  return loaded;
}

export async function detectInBrowser(file) {
  await loadModel();
  return (await call({type: 'detect', file})).result;
}
