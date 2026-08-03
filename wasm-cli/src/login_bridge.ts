// Browser-side half of `gen`'s Emscripten login flow (see `src/commands/remote/browser.rs` for
// the Rust side and the full protocol description). This module runs on the main thread only:
// `window.open`, popup-close polling, and the `message` listener all require a real `window`,
// which the Cockle command worker does not have.
//
// The Rust process (running inside the worker) cannot call any of this directly. Instead it
// writes a sentinel-framed "begin" message to stdout -- which `index.ts` already routes through
// `outputCallback` on the main thread before it reaches the visible terminal -- and blocks
// reading its own stdin for a sentinel-framed "result" message. This bridge is what produces
// that result and delivers it back over the same stdin stream via `shell.input()`, so no new
// Cockle transport, SharedArrayBuffer, or Asyncify is needed: the existing stdin channel
// (already used for keyboard/mouse input in `gen view`) is the suspend/resume mechanism.

/** Parsed contents of a `\0GEN_LOGIN_BEGIN\0{...}\0`-framed stdout message. */
export interface BeginLoginMessage {
  login_url: string;
  callback_url: string;
  expected_state: string;
  expected_origin: string;
}

// Must match BEGIN_SENTINEL / RESULT_SENTINEL / MESSAGE_TERMINATOR in
// src/commands/remote/browser.rs exactly: a NUL byte, framing text, and a NUL byte.
const NUL = String.fromCharCode(0);
const BEGIN_SENTINEL = `${NUL}GEN_LOGIN_BEGIN${NUL}`;
const RESULT_SENTINEL = `${NUL}GEN_LOGIN_RESULT${NUL}`;
const MESSAGE_TERMINATOR = NUL;

const RESULT_MESSAGE_TYPE = 'gen-login-result';
const ACK_MESSAGE_TYPE = 'gen-login-ack';

const LOGIN_TIMEOUT_MS = 5 * 60 * 1000;
const POPUP_CLOSE_POLL_MS = 500;

/** Parses a complete `BEGIN_SENTINEL`-framed message, or returns `null` if not found/malformed. */
export function extractBeginMessage(text: string): BeginLoginMessage | null {
  const start = text.indexOf(BEGIN_SENTINEL);
  if (start === -1) {
    return null;
  }
  const rest = text.slice(start + BEGIN_SENTINEL.length);
  const end = rest.indexOf(MESSAGE_TERMINATOR);
  if (end === -1) {
    return null;
  }
  try {
    const parsed = JSON.parse(rest.slice(0, end));
    if (
      typeof parsed?.login_url === 'string' &&
      typeof parsed?.callback_url === 'string' &&
      typeof parsed?.expected_state === 'string' &&
      typeof parsed?.expected_origin === 'string'
    ) {
      return parsed as BeginLoginMessage;
    }
  } catch {
    // Malformed JSON between the sentinels; treated the same as "no message found".
  }
  return null;
}

/**
 * Splits `buffer` (accumulated terminal output not yet processed) around the first
 * `BEGIN_SENTINEL`-framed message.
 *
 * Returns:
 * - `{ before, begin, after }` once a complete message has arrived: `before`/`after` are the
 *   surrounding text to display, `begin` the parsed message (or `null` if malformed, in which
 *   case it is dropped silently rather than shown to the user).
 * - `null` if there is no sentinel yet (nothing to buffer) or the sentinel has started but the
 *   terminator has not yet arrived (caller should keep buffering).
 */
export function splitOnBeginMessage(
  buffer: string
): { before: string; begin: BeginLoginMessage | null; after: string } | null {
  const start = buffer.indexOf(BEGIN_SENTINEL);
  if (start === -1) {
    return null;
  }
  const rest = buffer.slice(start + BEGIN_SENTINEL.length);
  const end = rest.indexOf(MESSAGE_TERMINATOR);
  if (end === -1) {
    return null;
  }
  const before = buffer.slice(0, start);
  const after = rest.slice(end + 1);
  const begin = extractBeginMessage(buffer.slice(start, start + BEGIN_SENTINEL.length + end + 1));
  return { before, begin, after };
}

type BridgeStatus = 'success' | 'cancelled' | 'timeout' | 'bridge_error';

interface BridgeResult {
  status: BridgeStatus;
  state?: string;
  jwt?: string;
  refresh_token?: string;
  message?: string;
}

function encodeResult(result: BridgeResult): string {
  return `${RESULT_SENTINEL}${JSON.stringify(result)}${MESSAGE_TERMINATOR}`;
}

interface Attempt {
  state: string;
  expectedOrigin: string;
  windows: Set<Window>;
  settled: boolean;
  // `number`, not `NodeJS.Timeout`: these always come from `window.setTimeout`/`window.setInterval`
  // (the DOM/webworker overload), never Node's.
  timeoutHandle: number;
  popupPollHandles: Set<number>;
}

/**
 * Drives one Cockle terminal's browser login attempts: opens the popup (or exposes the fallback
 * link), validates and delivers the callback result, and cleans up on completion, timeout, or
 * cancellation. One instance per terminal; a new `begin()` call always invalidates whatever
 * attempt came before it, so a stale popup from an earlier `gen remote login` can never complete
 * a newer one.
 */
export class BrowserLoginBridge {
  constructor(
    private readonly writeStdin: (data: string) => void | Promise<void>,
    private readonly onFallbackNeeded: (loginUrl: string) => void
  ) {
    window.addEventListener('message', this.handleMessage);
  }

  /** True if `uri` is the active attempt's login URL (used by the xterm web-links handler). */
  isPendingLoginUrl(uri: string): boolean {
    return this.pendingLoginUrl !== null && uri === this.pendingLoginUrl;
  }

  /** Opens (or re-opens) the login popup for the active attempt, e.g. from a fallback-link click. */
  openLoginWindow(): Window | null {
    if (!this.attempt || this.attempt.settled || this.pendingLoginUrl === null) {
      return null;
    }
    const popup = window.open(this.pendingLoginUrl, 'gen-login');
    if (popup) {
      this.attempt.windows.add(popup);
      this.pollPopupClosed(this.attempt, popup);
    }
    return popup;
  }

  /** Starts a new login attempt. Any previous attempt on this bridge is invalidated first. */
  begin(begin: BeginLoginMessage): void {
    this.invalidateActiveAttempt();

    const attempt: Attempt = {
      state: begin.expected_state,
      expectedOrigin: begin.expected_origin,
      windows: new Set(),
      settled: false,
      timeoutHandle: window.setTimeout(() => this.finish(attempt, { status: 'timeout' }), LOGIN_TIMEOUT_MS),
      popupPollHandles: new Set(),
    };
    this.attempt = attempt;
    this.pendingLoginUrl = begin.login_url;

    const popup = window.open(begin.login_url, 'gen-login');
    if (popup) {
      attempt.windows.add(popup);
      this.pollPopupClosed(attempt, popup);
    } else {
      this.onFallbackNeeded(begin.login_url);
    }
  }

  /** Cancels the active attempt (e.g. Ctrl-C). Never touches previously saved credentials. */
  cancel(): void {
    if (this.attempt) {
      this.finish(this.attempt, { status: 'cancelled' });
    }
  }

  /** Removes all listeners/timers and forgets the active attempt without delivering a result. */
  dispose(): void {
    window.removeEventListener('message', this.handleMessage);
    this.invalidateActiveAttempt();
  }

  private pendingLoginUrl: string | null = null;
  private attempt: Attempt | null = null;

  private pollPopupClosed(attempt: Attempt, popup: Window): void {
    const handle = window.setInterval(() => {
      if (popup.closed) {
        window.clearInterval(handle);
        attempt.popupPollHandles.delete(handle);
        attempt.windows.delete(popup);
      }
    }, POPUP_CLOSE_POLL_MS);
    attempt.popupPollHandles.add(handle);
  }

  private readonly handleMessage = (event: MessageEvent): void => {
    const attempt = this.attempt;
    if (!attempt || attempt.settled) {
      return;
    }
    // Exact-origin check: never widened to a wildcard, and never the GenHub API origin -- only
    // the final browser callback page's own origin, supplied by the Rust side's `begin` message.
    if (event.origin !== attempt.expectedOrigin) {
      return;
    }
    const data = event.data as Record<string, unknown> | null;
    if (!data || data['type'] !== RESULT_MESSAGE_TYPE) {
      return;
    }
    // Source validation: accept either window we ourselves opened for this attempt (the
    // automatic popup and/or the clicked fallback), never an arbitrary sender.
    const source = event.source as Window | null;
    if (attempt.windows.size > 0 && (!source || !attempt.windows.has(source))) {
      return;
    }
    const state = data['state'];
    const jwt = data['jwt'];
    const refreshToken = data['refreshToken'];
    if (typeof state !== 'string' || typeof jwt !== 'string' || typeof refreshToken !== 'string') {
      return;
    }

    // Acknowledge so the callback page can stop retrying and close itself. JS-side state
    // matching here is defense in depth only -- Rust is the sole authority on whether `state`
    // actually matches the nonce it generated (see `parse_bridge_message` in browser.rs).
    source?.postMessage({ type: ACK_MESSAGE_TYPE, state }, attempt.expectedOrigin);

    this.finish(attempt, { status: 'success', state, jwt, refresh_token: refreshToken });
  };

  private finish(attempt: Attempt, result: BridgeResult): void {
    if (attempt.settled || attempt !== this.attempt) {
      return;
    }
    attempt.settled = true;
    window.clearTimeout(attempt.timeoutHandle);
    for (const handle of attempt.popupPollHandles) {
      window.clearInterval(handle);
    }
    this.attempt = null;
    this.pendingLoginUrl = null;
    void this.writeStdin(encodeResult(result));
  }

  private invalidateActiveAttempt(): void {
    const attempt = this.attempt;
    if (!attempt) {
      return;
    }
    window.clearTimeout(attempt.timeoutHandle);
    for (const handle of attempt.popupPollHandles) {
      window.clearInterval(handle);
    }
    this.attempt = null;
    this.pendingLoginUrl = null;
  }
}
