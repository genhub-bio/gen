import type { ITheme } from '@xterm/xterm';

// Narrow representation of which Catppuccin flavor is active for a terminal/shell session.
// Selected once at startup (see `selectThemeMode`) and never changed for the lifetime of the
// session; there is no live theme switching.
export type ThemeMode = 'light' | 'dark';

// A Base16 (https://github.com/chriskempson/base16) palette: sixteen named hex colors shared by
// terminal, editor, and other themes built on the Base16 convention.
export interface Base16Palette {
  base00: string;
  base01: string;
  base02: string;
  base03: string;
  base04: string;
  base05: string;
  base06: string;
  base07: string;
  base08: string;
  base09: string;
  base0A: string;
  base0B: string;
  base0C: string;
  base0D: string;
  base0E: string;
  base0F: string;
}

// Catppuccin Mocha, used for dark-mode terminals.
export const CATPPUCCIN_MOCHA: Base16Palette = {
  base00: '#1e1e2e',
  base01: '#181825',
  base02: '#313244',
  base03: '#45475a',
  base04: '#585b70',
  base05: '#cdd6f4',
  base06: '#f5e0dc',
  base07: '#b4befe',
  base08: '#f38ba8',
  base09: '#fab387',
  base0A: '#f9e2af',
  base0B: '#a6e3a1',
  base0C: '#94e2d5',
  base0D: '#89b4fa',
  base0E: '#cba6f7',
  base0F: '#f2cdcd',
};

// Catppuccin Latte, used for light-mode terminals.
export const CATPPUCCIN_LATTE: Base16Palette = {
  base00: '#eff1f5',
  base01: '#e6e9ef',
  base02: '#ccd0da',
  base03: '#bcc0cc',
  base04: '#acb0be',
  base05: '#4c4f69',
  base06: '#dc8a78',
  base07: '#7287fd',
  base08: '#d20f39',
  base09: '#fe640b',
  base0A: '#df8e1d',
  base0B: '#40a02b',
  base0C: '#179299',
  base0D: '#1e66f5',
  base0E: '#8839ef',
  base0F: '#dd7878',
};

const PALETTES_BY_MODE: Record<ThemeMode, Base16Palette> = {
  light: CATPPUCCIN_LATTE,
  dark: CATPPUCCIN_MOCHA,
};

// Maps a Base16 palette onto xterm.js's `ITheme` fields, per the standard Base16-to-terminal
// convention (https://github.com/chriskempson/base16-vim/blob/master/README.md#256-colors).
export function base16ToXtermTheme(palette: Base16Palette): ITheme {
  return {
    background: palette.base00,
    foreground: palette.base05,
    cursor: palette.base05,
    cursorAccent: palette.base00,
    selectionBackground: palette.base02,
    selectionForeground: palette.base05,
    selectionInactiveBackground: palette.base01,

    black: palette.base00,
    red: palette.base08,
    green: palette.base0B,
    yellow: palette.base0A,
    blue: palette.base0D,
    magenta: palette.base0E,
    cyan: palette.base0C,
    white: palette.base05,

    brightBlack: palette.base03,
    brightRed: palette.base08,
    brightGreen: palette.base0B,
    brightYellow: palette.base0A,
    brightBlue: palette.base0D,
    brightMagenta: palette.base0E,
    brightCyan: palette.base0C,
    brightWhite: palette.base07,
  };
}

// Picks the Catppuccin flavor for this session from the browser's current color-scheme
// preference. Called once, before the terminal is constructed; the result is not re-evaluated if
// the OS preference changes later.
export function selectThemeMode(): ThemeMode {
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function base16PaletteForMode(mode: ThemeMode): Base16Palette {
  return PALETTES_BY_MODE[mode];
}

export function xtermThemeForMode(mode: ThemeMode): ITheme {
  return base16ToXtermTheme(base16PaletteForMode(mode));
}
