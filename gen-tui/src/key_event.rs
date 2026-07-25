//! Keyboard/mouse event types used by `gen-tui` and its consumers.
//!
//! When the `crossterm` feature is enabled (the default, used natively) these are plain
//! re-exports of `crossterm::event` types. When it's disabled (the `wasm32-unknown-emscripten`
//! build, where crossterm's `event` module is unavailable because it requires `mio`), a minimal
//! shim mirroring the same variant/field shape is used instead, so call sites don't need to
//! change based on target.

#[cfg(feature = "crossterm")]
pub use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};

#[cfg(not(feature = "crossterm"))]
mod shim {
    use core::ops::BitOr;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Event {
        Key(KeyEvent),
        Mouse(MouseEvent),
        Resize(u16, u16),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum KeyCode {
        Char(char),
        Enter,
        Esc,
        Backspace,
        Left,
        Right,
        Up,
        Down,
        Tab,
        BackTab,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum KeyEventKind {
        Press,
        Release,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct KeyModifiers(u8);

    impl KeyModifiers {
        pub const ALT: Self = Self(1 << 2);
        pub const CONTROL: Self = Self(1 << 1);
        pub const NONE: Self = Self(0);
        pub const SHIFT: Self = Self(1 << 0);

        pub fn contains(&self, other: Self) -> bool {
            self.0 & other.0 == other.0
        }
    }

    impl BitOr for KeyModifiers {
        type Output = Self;

        fn bitor(self, rhs: Self) -> Self {
            Self(self.0 | rhs.0)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct KeyEvent {
        pub code: KeyCode,
        pub modifiers: KeyModifiers,
        pub kind: KeyEventKind,
    }

    impl KeyEvent {
        pub fn new(code: KeyCode, modifiers: KeyModifiers) -> Self {
            Self {
                code,
                modifiers,
                kind: KeyEventKind::Press,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum MouseButton {
        Left,
        Right,
        Middle,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum MouseEventKind {
        Down(MouseButton),
        Up(MouseButton),
        Drag(MouseButton),
        Moved,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MouseEvent {
        pub kind: MouseEventKind,
        pub column: u16,
        pub row: u16,
        pub modifiers: KeyModifiers,
    }
}

#[cfg(not(feature = "crossterm"))]
pub use shim::*;
