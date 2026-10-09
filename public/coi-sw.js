// Adds the cross-origin isolation headers that static hosts such as GitHub Pages cannot set.
// Isolation unlocks SharedArrayBuffer, which lets the in-browser model use several CPU cores.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
  event.respondWith(fetch(request).then((response) => {
    // Opaque and error responses cannot be rewritten.
    if (response.status === 0) return response;
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
    return new Response(response.body, {status: response.status, statusText: response.statusText, headers});
  }));
});
