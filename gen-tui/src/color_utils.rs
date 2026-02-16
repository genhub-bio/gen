use ratatui::style::Color;

/// Convert sRGB [0-255] to linear RGB [0-1]
fn srgb_to_linear(srgb: u8) -> f32 {
    let v = srgb as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear RGB [0-1] to sRGB [0-255]
fn linear_to_srgb(linear: f32) -> u8 {
    let v = linear.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        (v * 12.92 * 255.0) as u8
    } else {
        ((1.055 * v.powf(1.0 / 2.4) - 0.055) * 255.0) as u8
    }
}

/// Convert a Color to RGB tuple if possible
fn rgb_from_color(color: Color) -> Option<(u8, u8, u8)> {
    match color {
        Color::Rgb(r, g, b) => Some((r, g, b)),
        Color::Black => Some((0, 0, 0)),
        Color::Red => Some((255, 0, 0)),
        Color::Green => Some((0, 255, 0)),
        Color::Yellow => Some((255, 255, 0)),
        Color::Blue => Some((0, 0, 255)),
        Color::Magenta => Some((255, 0, 255)),
        Color::Cyan => Some((0, 255, 255)),
        Color::Gray => Some((128, 128, 128)),
        Color::DarkGray => Some((64, 64, 64)),
        Color::LightRed => Some((255, 128, 128)),
        Color::LightGreen => Some((128, 255, 128)),
        Color::LightYellow => Some((255, 255, 128)),
        Color::LightBlue => Some((128, 128, 255)),
        Color::LightMagenta => Some((255, 128, 255)),
        Color::LightCyan => Some((128, 255, 255)),
        Color::White => Some((255, 255, 255)),
        _ => None, // Indexed, Reset, etc.
    }
}

/// Calculate relative luminance from RGB (WCAG formula)
fn relative_luminance(r: u8, g: u8, b: u8) -> f32 {
    let r_linear = srgb_to_linear(r);
    let g_linear = srgb_to_linear(g);
    let b_linear = srgb_to_linear(b);
    0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear
}

/// Calculate WCAG contrast ratio between two luminances
fn contrast_ratio(lum1: f32, lum2: f32) -> f32 {
    let lighter = lum1.max(lum2);
    let darker = lum1.min(lum2);
    (lighter + 0.05) / (darker + 0.05)
}

/// Blend two RGB colors in linear space with gamma correction
fn blend_rgb(a: (u8, u8, u8), b: (u8, u8, u8), factor: f32) -> (u8, u8, u8) {
    let factor = factor.clamp(0.0, 1.0);

    // Convert to linear
    let a_r = srgb_to_linear(a.0);
    let a_g = srgb_to_linear(a.1);
    let a_b = srgb_to_linear(a.2);

    let b_r = srgb_to_linear(b.0);
    let b_g = srgb_to_linear(b.1);
    let b_b = srgb_to_linear(b.2);

    // Blend in linear space
    let r = a_r * (1.0 - factor) + b_r * factor;
    let g = a_g * (1.0 - factor) + b_g * factor;
    let b = a_b * (1.0 - factor) + b_b * factor;

    // Convert back to sRGB
    (linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b))
}

/// Tint foreground and background colors towards a tint color with contrast preservation
pub fn tint_colors(fg: Color, bg: Color, tint: Color, strength: f32) -> (Color, Color) {
    let Some(tint_rgb) = rgb_from_color(tint) else {
        return (fg, bg);
    };

    let Some(fg_rgb) = rgb_from_color(fg) else {
        return (fg, bg);
    };

    let Some(bg_rgb) = rgb_from_color(bg) else {
        return (fg, bg);
    };

    // Blend colors
    let mut new_fg_rgb = blend_rgb(fg_rgb, tint_rgb, strength);
    let mut new_bg_rgb = blend_rgb(bg_rgb, tint_rgb, strength);

    // Check contrast ratio
    let fg_lum = relative_luminance(new_fg_rgb.0, new_fg_rgb.1, new_fg_rgb.2);
    let bg_lum = relative_luminance(new_bg_rgb.0, new_bg_rgb.1, new_bg_rgb.2);
    let ratio = contrast_ratio(fg_lum, bg_lum);

    // If contrast is too low, reduce the blend strength until we reach acceptable contrast
    if ratio < 4.5 {
        let mut min_strength = 0.0;
        let mut max_strength = strength;

        // Binary search for the maximum strength that satisfies the contrast requirement
        for _ in 0..10 {
            let mid_strength = (min_strength + max_strength) / 2.0;
            let test_fg = blend_rgb(fg_rgb, tint_rgb, mid_strength);
            let test_bg = blend_rgb(bg_rgb, tint_rgb, mid_strength);
            let test_fg_lum = relative_luminance(test_fg.0, test_fg.1, test_fg.2);
            let test_bg_lum = relative_luminance(test_bg.0, test_bg.1, test_bg.2);

            if contrast_ratio(test_fg_lum, test_bg_lum) >= 4.5 {
                min_strength = mid_strength;
                new_fg_rgb = test_fg;
                new_bg_rgb = test_bg;
            } else {
                max_strength = mid_strength;
            }
        }

        // If even with strength 0 we didn't reach 4.5, it means the original colors
        // already had low contrast. In that case, we at least return the original colors.
        if min_strength == 0.0 {
            new_fg_rgb = fg_rgb;
            new_bg_rgb = bg_rgb;
        }
    }

    (
        Color::Rgb(new_fg_rgb.0, new_fg_rgb.1, new_fg_rgb.2),
        Color::Rgb(new_bg_rgb.0, new_bg_rgb.1, new_bg_rgb.2),
    )
}

/// Brighten colors by tinting towards white
pub fn brighten_colors(fg: Color, bg: Color, strength: f32) -> (Color, Color) {
    tint_colors(fg, bg, Color::White, strength)
}

/// Dim colors by tinting towards black
pub fn dim_colors(fg: Color, bg: Color, strength: f32) -> (Color, Color) {
    tint_colors(fg, bg, Color::Black, strength)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tint_contrast_preservation() {
        let fg = Color::White;
        let bg = Color::Black;
        let tint = Color::White;
        let strength = 1.0;

        let (new_fg, new_bg) = tint_colors(fg, bg, tint, strength);

        let Color::Rgb(r1, g1, b1) = new_fg else {
            panic!("Expected Rgb")
        };
        let Color::Rgb(r2, g2, b2) = new_bg else {
            panic!("Expected Rgb")
        };

        let lum1 = relative_luminance(r1, g1, b1);
        let lum2 = relative_luminance(r2, g2, b2);
        let ratio = contrast_ratio(lum1, lum2);

        assert!(ratio >= 4.5, "Contrast ratio {} is less than 4.5", ratio);
    }

    #[test]
    fn test_tint_indexed_256_colors() {
        let fg = Color::Indexed(231); // White in 256-cube
        let bg = Color::Indexed(16); // Black in 256-cube
        let tint = Color::White;
        let strength = 0.5;

        let (new_fg, new_bg) = tint_colors(fg, bg, tint, strength);

        assert!(matches!(new_fg, Color::Rgb(_, _, _)));
        assert!(matches!(new_bg, Color::Rgb(_, _, _)));

        let Color::Rgb(r1, g1, b1) = new_fg else {
            panic!()
        };
        let Color::Rgb(r2, g2, b2) = new_bg else {
            panic!()
        };

        let ratio = contrast_ratio(
            relative_luminance(r1, g1, b1),
            relative_luminance(r2, g2, b2),
        );
        assert!(ratio >= 4.5);
    }
}
