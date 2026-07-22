// Trimmed port of @jupyterlite/apputils' ServiceWorkerManager, dropping the PageConfig/
// settingregistry/theme-manager plugin wiring that package pulls in along with a large slice of
// the JupyterLab application framework (see the design memory this branch's continuation notes
// refer to for why that package was avoided). Keeps only what's needed here: registering the
// vendored service-worker.ts (service-worker.ts) and answering its broadcast messages, routing
// `/api/drive` requests to a Contents.IManager and `/api/stdin/<suffix>` requests to a registered
// handler.
import type { Contents } from '@jupyterlab/services';
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
  (message: any): Promise<any>;
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

    void this._initialize().catch(console.warn);
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
      console.warn('ServiceWorkers not supported in this browser');
      this._ready.reject(void 0);
      return;
    }

    await unregisterStaleServiceWorkers(this._workerUrl);

    if (serviceWorker.controller) {
      registration = (await serviceWorker.getRegistration(serviceWorker.controller.scriptURL)) ?? null;
    }

    if (!registration) {
      try {
        registration = await serviceWorker.register(this._workerUrl, { type: 'module' });
      } catch (err: any) {
        console.warn(`ServiceWorker registration failed: ${err}`);
      }
    }

    if (!registration) {
      this._ready.reject(void 0);
    } else {
      this._ready.resolve(void 0);
      setTimeout(this._pingServiceWorker.bind(this), HEARTBEAT_MS);
    }
  }

  private _onBroadcastMessage = async (
    event: MessageEvent<{
      data: any;
      browsingContextId: string;
      requestId?: string;
      pathname: string;
    }>
  ): Promise<void> => {
    const { data, browsingContextId, requestId, pathname } = event.data;

    if (browsingContextId !== this._browsingContextId) {
      return;
    }

    if (pathname.includes('/api/stdin/')) {
      await this._onStdinMessage(pathname, data);
    } else {
      await this._onDriveMessage(data, requestId);
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

  private async _onStdinMessage(pathname: string, data: any): Promise<void> {
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
    const response = await fetch(SW_PING_ENDPOINT);
    const text = await response.text();
    if (text === 'ok') {
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
