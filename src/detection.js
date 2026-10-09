export const MAX_SIZE = 10 * 1024 * 1024;
export function validateFile(file) {
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) return 'Choose a JPEG, PNG or WebP image.';
  if (file.size > MAX_SIZE) return 'Image must be smaller than 10 MB.';
  if (!file.size) return 'This file is empty.';
  return '';
}
export function severity(coverage) { return coverage >= 25 ? 'High' : coverage >= 10 ? 'Moderate' : 'Low'; }
export function demoResult() { return {coverage:18.7,severity:severity(18.7),demo:true,mask:null}; }
export function validateResult(data) {
  if (!Number.isFinite(data.coverage) || data.coverage < 0 || data.coverage > 100) throw new Error('The API returned an invalid coverage value.');
  return {...data,severity:severity(data.coverage),demo:false};
}
