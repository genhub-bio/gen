// Vendored from @jupyterlite/apputils' service-worker.ts (zero runtime dependencies), see
// VENDORED_PACKAGES.md. Broadcasts /api/drive and /api/stdin/ requests to the main thread over a
// BroadcastChannel so a synchronous XHR from a web worker (DriveFS, cockle's stdin) can be
// answered by code running in the main UI thread.

const broadcast = new BroadcastChannel('/sw-api.v1');

// @ts-expect-error TS2769
self.addEventListener('install', onInstall);
// @ts-expect-error TS2769
self.addEventListener('activate', onActivate);
// @ts-expect-error TS2769
self.addEventListener('fetch', onFetch);

function onInstall(event: ExtendableEvent): void {
  // @ts-expect-error TS2339
  void self.skipWaiting();
}

function onActivate(event: ExtendableEvent): void {
  // @ts-expect-error TS2551
  event.waitUntil(self.clients.claim());
}

function onFetch(event: FetchEvent): void {
  const { request } = event;

  const url = new URL(request.url);
  if (url.pathname === '/api/service-worker-heartbeat') {
    event.respondWith(new Response('ok'));
    return;
  }

  if (shouldBroadcast(url)) {
    event.respondWith(broadcastOne(request, url));
  }
}

function shouldBroadcast(url: URL): boolean {
  return (
    url.origin === location.origin &&
    (url.pathname.includes('/api/drive') || url.pathname.includes('/api/stdin/'))
  );
}

async function broadcastOne(request: Request, url: URL): Promise<Response> {
  const message = await request.json();
  const promise = new Promise<Response>(resolve => {
    const messageHandler = (event: MessageEvent) => {
      const data = event.data;
      if (
        data.browsingContextId !== message.browsingContextId ||
        data.requestId !== message.requestId
      ) {
        // bail if the message is not for us
        return;
      }
      resolve(new Response(JSON.stringify(data.response)));
      broadcast.removeEventListener('message', messageHandler);
    };

    broadcast.addEventListener('message', messageHandler);
  });

  message.pathname = url.pathname;
  broadcast.postMessage(message);

  return await promise;
}
