import test from 'node:test';
import assert from 'node:assert/strict';
import {enhance, resizeLinear, toTensor, summarize, rgbaToRgb} from '../src/inference/preprocess.js';

// Reference values from the notebook's enhance() and cv2.resize() (OpenCV 5.0) on a 7 × 5 RGB image.
const input = [139,74,229,241,169,65,6,160,149,106,38,175,188,205,175,229,98,249,10,148,95,86,147,198,66,39,106,213,171,45,167,57,69,81,55,14,153,178,215,76,52,39,250,72,215,50,161,223,112,172,161,233,38,17,89,1,221,6,242,127,143,6,60,210,94,25,166,33,249,189,12,204,33,57,124,30,199,149,202,119,72,220,3,209,122,136,147,77,133,82,115,87,162,230,70,71,240,62,47,184,31,34,63,65,146];
const enhanced = [144,54,255,255,194,32,0,183,169,90,6,208,213,254,198,255,88,255,0,161,70,64,165,219,34,1,88,252,202,9,182,37,49,57,31,0,170,209,254,56,18,0,255,57,238,7,190,254,99,210,174,255,13,0,73,0,255,0,255,128,155,0,28,242,94,0,177,5,255,214,0,231,0,44,124,0,251,167,235,131,52,255,0,255,109,158,145,43,157,45,103,93,167,255,65,43,255,47,17,201,3,0,16,59,164];
const down = [141,105,171,88,167,120,156,159,135,96,119,125,162,107,244,119,185,152,96,2,223,97,96,66,129,125,103,144,61,173,170,108,67,139,42,128];
const up = [144,54,255,218,147,106,142,189,93,20,143,178,90,6,208,186,199,200,236,162,230,170,112,193,0,161,70,109,102,239,142,107,117,137,146,75,114,153,108,130,19,138,141,126,117,185,148,191,153,127,183,25,98,39,76,158,220,47,61,139,125,97,63,230,165,25,186,35,46,86,30,22,114,131,144,127,147,164,62,17,2,195,91,232,75,117,212,74,165,167,166,166,99,232,20,15,104,12,140,60,138,171,77,162,118,124,6,19,251,68,164,124,111,224,93,137,226,144,117,158,175,23,39,78,66,185,63,155,159,111,144,102,186,0,99,243,92,15,192,42,175,184,15,243,164,19,203,16,42,116,7,192,160,124,182,108,230,92,118,249,0,241,167,130,82,123,103,118,124,73,163,150,53,169,143,56,78,143,118,82,184,93,49,184,50,83,121,33,204,109,158,145,65,157,78,70,128,99,137,87,139,255,65,43,255,51,23,225,22,7,139,22,55,16,59,164];

// Float rounding may differ from OpenCV by one level on a few values; the model does not notice.
function close(actual, expected) {
  assert.equal(actual.length, expected.length);
  const off = [...actual].filter((v, i) => Math.abs(v - expected[i]) > 1);
  assert.deepEqual(off, []);
}

test('enhance matches the notebook unsharp mask', () => close(enhance(Uint8Array.from(input), 7, 5), enhanced));
test('resize matches cv2.resize INTER_LINEAR when shrinking', () => close(resizeLinear(Uint8Array.from(enhanced), 7, 5, 4, 3), down));
test('resize matches cv2.resize INTER_LINEAR when enlarging', () => close(resizeLinear(Uint8Array.from(enhanced), 7, 5, 9, 8), up));

test('tensor is channel-first and scaled to 0..1', () => {
  const t = toTensor(Uint8Array.from([255, 0, 51, 0, 255, 102]), 2, 1);
  assert.deepEqual([...t].map(v => +v.toFixed(2)), [1, 0, 0, 1, 0.2, 0.4]);
});

test('alpha channel is dropped', () => assert.deepEqual([...rgbaToRgb(Uint8Array.from([1, 2, 3, 255, 4, 5, 6, 0]), 2, 1)], [1, 2, 3, 4, 5, 6]));

test('coverage counts pixels strictly above the 0.5 threshold', () => {
  const {mask, coverage} = summarize(Float32Array.from([0.2, 0.5, 0.51, 0.9]));
  assert.deepEqual([...mask], [0, 0, 1, 1]);
  assert.equal(coverage, 50);
});
