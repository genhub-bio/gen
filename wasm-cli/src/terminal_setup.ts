import type { ITerminalInitOnlyOptions, ITerminalOptions } from '@xterm/xterm';
import type { ThemeMode } from './theme';
import { xtermThemeForMode } from './theme';

export const TERMINAL_FONT_FAMILY = '"JetBrains Mono", monospace';
export const TERMINAL_FONT_SIZE = 14;

// Cell-geometry and rendering options shared by both Catppuccin flavors. `lineHeight`,
// `letterSpacing`, and `customGlyphs` are load-bearing for seamless block-element, eighth-width
// bar, and box-drawing rendering; do not adjust them here without re-checking that.
export function buildTerminalOptions(
  mode: ThemeMode
): ITerminalOptions & ITerminalInitOnlyOptions {
  return {
    fontSize: TERMINAL_FONT_SIZE,
    fontFamily: TERMINAL_FONT_FAMILY,
    fontWeight: '400',
    fontWeightBold: '600',
    lineHeight: 1,
    letterSpacing: 0,
    customGlyphs: true,
    drawBoldTextInBrightColors: false,
    minimumContrastRatio: 4.5,
    cursorStyle: 'block',
    cursorInactiveStyle: 'outline',
    cursorBlink: true,
    scrollback: 5000,
    rows: 40,
    theme: xtermThemeForMode(mode),
  };
}

// Merges `GEN_THEME` into a shell's environment without disturbing any other entries
// (in particular `GEN_TERMINAL_ORIGIN` and `GEN_LOGIN_CALLBACK_URL`).
export function buildShellEnvironment(
  mode: ThemeMode,
  baseEnvironment: Record<string, string>
): Record<string, string> {
  return {
    ...baseEnvironment,
    GEN_THEME: mode,
  };
}
