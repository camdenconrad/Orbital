//! A dark theme that stays readable on a projector and in a dim control room.
//!
//! Deliberately self-contained: this crate keeps its UI dependencies to
//! egui, so it clones and builds anywhere.

use egui::{Color32, FontFamily, FontId, TextStyle};

/// Nominal. Used for healthy telemetry and the horizon.
#[allow(dead_code)]
pub const GOOD: Color32 = Color32::from_rgb(126, 200, 140);
/// Off-nominal: dropped packets, error events. The only red in the UI, so it
/// means something when it appears.
#[allow(dead_code)]
pub const BAD: Color32 = Color32::from_rgb(224, 108, 92);
/// The accent: the spacecraft, its track, the active control.
pub const ACCENT: Color32 = Color32::from_rgb(233, 148, 72);
/// Dimmed text for units, hints, and inactive rows.
#[allow(dead_code)]
pub const MUTED: Color32 = Color32::from_rgb(140, 140, 152);

pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.text_styles = [
        (TextStyle::Heading, FontId::new(18.0, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(14.0, FontFamily::Proportional)),
        (TextStyle::Small, FontId::new(11.5, FontFamily::Proportional)),
        // Telemetry is tabular: monospace keeps columns from dancing as values change.
        (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
    ]
    .into();

    let s = &mut style.spacing;
    s.item_spacing = egui::vec2(8.0, 6.0);
    s.button_padding = egui::vec2(10.0, 5.0);

    let v = &mut style.visuals;
    v.dark_mode = true;
    v.panel_fill = Color32::from_rgb(24, 24, 28);
    v.window_fill = Color32::from_rgb(20, 20, 24);
    v.faint_bg_color = Color32::from_rgb(34, 34, 40);
    v.extreme_bg_color = Color32::from_rgb(14, 14, 17);
    v.selection.bg_fill = ACCENT.linear_multiply(0.35);
    v.hyperlink_color = ACCENT;

    ctx.set_style(style);
}
