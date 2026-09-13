// Trimmed port of @jupyterlite/apputils' ServiceWorkerManager, dropping the PageConfig/
// settingregistry/theme-manager plugin wiring that package pulls in along with a large slice of
// the JupyterLab application framework (see the design memory this branch's continuation notes
// refer to for why that package was avoided). Keeps only what's needed here: registering the
// vendored service-worker.ts (service-worker.ts) and answering its broadcast messages, routing
// `/api/drive` requests to a Contents.IManager and `/api/stdin/<suffix>` requests to a registered
// handler.
import type { Contents } from '@jupyterlab/services';
import type { IStdinReply, IStdinRequest } from '@jupyterlite/cockle';
import type { TDriveMethod, TDriveRequest } from '@jupyterlite/services';
import { DriveContentsProcessor } from '@jupyterlite/services';
import { PromiseDelegate, UUID } from '@lumino/coreutils';

const BROADCAST_CHANNEL_ID = '/sw-api.v1';
const SW_PING_ENDPOINT = '/api/service-worker-heartbeat';
const HEARTBEAT_MS = 20000;

// Bump whenever service-worker.ts or the broadcast message shape it relays changes: forces
// browsers with an already-installed (and already-controlling, via clients.claim()) worker from a
// previous build to unregister and pick up the new one, rather than keep answering broadcast
// messages with stale/mismatched logic until the tab is manually reloaded twice or hard-refreshed.
const SW_PROTOCOL_VERSION = '1';

async function unregisterStaleServiceWorkers(scriptUrl: string): Promise<void> {
  const versionKey = `${scriptUrl}-version`;
  const installedVersion = localStorage.getItem(versionKey);
  if (installedVersion !== SW_PROTOCOL_VERSION) {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map(registration => registration.unregister()));
    localStorage.setItem(versionKey, SW_PROTOCOL_VERSION);
  }
}

export interface IStdinHandler {
  (message: IStdinRequest): Promise<IStdinReply>;
}

interface BroadcastRequest {
  data: IStdinRequest | TDriveRequest<TDriveMethod>;
  browsingContextId: string;
  requestId?: string;
  pathname: string;
}

export class ServiceWorkerManager {
  constructor(options: ServiceWorkerManager.IOptions) {
    this._workerUrl = options.workerUrl;
    this._browsingContextId = UUID.uuid4();
    this._driveContentsProcessor = new DriveContentsProcessor({
      contentsManager: options.contents,
    });

    this._broadcastChannel = new BroadcastChannel(BROADCAST_CHANNEL_ID);
    this._broadcastChannel.addEventListener('message', this._onBroadcastMessage);

    void this._initialize().catch(error => {
      console.warn('ServiceWorker initialization failed', error);
      this._ready.reject(error);
    });
  }

  get browsingContextId(): string {
    return this._browsingContextId;
  }

  get ready(): Promise<void> {
    return this._ready.promise;
  }

  registerStdinHandler(pathnameSuffix: string, stdinHandler: IStdinHandler): void {
    this._stdinHandlers.set(pathnameSuffix, stdinHandler);
  }

  private async _initialize(): Promise<void> {
    const { serviceWorker } = navigator;

    let registration: ServiceWorkerRegistration | null = null;

    if (!serviceWorker) {
      this._ready.reject(new Error('Service workers are not supported in this browser'));
      return;
    }

    await unregisterStaleServiceWorkers(this._workerUrl);

    if (serviceWorker.controller) {
      registration = (await serviceWorker.getRegistration(serviceWorker.controller.scriptURL)) ?? null;
    }

    if (!registration) {
      try {
        registration = await serviceWorker.register(this._workerUrl, { type: 'module' });
      } catch (error: unknown) {
        console.warn(`ServiceWorker registration failed: ${String(error)}`);
      }
    }

    if (!registration) {
      this._ready.reject(new Error('Service worker registration failed'));
    } else {
      this._ready.resolve(void 0);
      setTimeout(this._pingServiceWorker.bind(this), HEARTBEAT_MS);
    }
  }

  private _onBroadcastMessage = async (
    event: MessageEvent<BroadcastRequest>
  ): Promise<void> => {
    const { data, browsingContextId, requestId, pathname } = event.data;

    if (browsingContextId !== this._browsingContextId) {
      return;
    }

    if (pathname.includes('/api/stdin/')) {
      await this._onStdinMessage(pathname, data as IStdinRequest);
    } else {
      await this._onDriveMessage(data as TDriveRequest<TDriveMethod>, requestId);
    }
  };

  private async _onDriveMessage(data: TDriveRequest<TDriveMethod>, requestId?: string): Promise<void> {
    const response = await this._driveContentsProcessor.processDriveRequest(data);
    this._broadcastChannel.postMessage({
      response,
      browsingContextId: this._browsingContextId,
      requestId,
    });
  }

  private async _onStdinMessage(pathname: string, data: IStdinRequest): Promise<void> {
    const suffix = pathname.slice(pathname.lastIndexOf('/') + 1);
    const stdinHandler = this._stdinHandlers.get(suffix);
    if (stdinHandler === undefined) {
      console.warn(`No stdin handler registered for '${pathname}'`);
      return;
    }
    const response = await stdinHandler(data);
    this._broadcastChannel.postMessage({ response, browsingContextId: this._browsingContextId });
  }

  private async _pingServiceWorker(): Promise<void> {
    try {
      const response = await fetch(SW_PING_ENDPOINT, { cache: 'no-store' });
      const text = await response.text();
      if (!response.ok || text !== 'ok') {
        console.warn(`ServiceWorker heartbeat returned HTTP ${response.status}: ${text}`);
      }
    } catch (error) {
      console.warn('ServiceWorker heartbeat failed; retrying', error);
    } finally {
      setTimeout(this._pingServiceWorker.bind(this), HEARTBEAT_MS);
    }
  }

  private _browsingContextId: string;
  private _broadcastChannel: BroadcastChannel;
  private _driveContentsProcessor: DriveContentsProcessor;
  private _ready = new PromiseDelegate<void>();
  private _stdinHandlers = new Map<string, IStdinHandler>();
  private _workerUrl: string;
}

export namespace ServiceWorkerManager {
  export interface IOptions {
    contents: Contents.IManager;
    workerUrl: string;
  }
}
