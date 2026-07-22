//! Rune window chrome — a decorationless title bar with autumn traffic lights.
//!
//! Ported into this crate rather than depending on Rune's `uikit`: this repo is a
//! standalone public repo and must clone and build on any machine. The chrome is
//! self-contained (egui only), so it works under any compositor — it just happens
//! to match Rune, which advertises no server-side decorations.
//!
//! Enabled by the `rune` feature; without it the window keeps its normal
//! decorations.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};

/// The bar itself — near-black, so the content reads as the window.
pub const CHROME: Color32 = Color32::from_rgb(19, 18, 22);

// Autumn traffic lights: the macOS three-dot idea in fall tones.
const MIN: Color32 = Color32::from_rgb(217, 164, 65); // golden amber
const MAX: Color32 = Color32::from_rgb(150, 110, 184); // plum
const CLOSE: Color32 = Color32::from_rgb(197, 82, 46); // burnt rust

/// Draw the title bar as the top-most panel of a `with_decorations(false)` window.
pub fn title_bar(ctx: &egui::Context, title: &str) {
    let maximized = ctx.input(|i| i.viewport().maximized).unwrap_or(false);
    let mut cmd: Option<egui::ViewportCommand> = None;

    egui::TopBottomPanel::top("rune-titlebar")
        .exact_height(36.0)
        .frame(egui::Frame::none().fill(CHROME))
        .show(ctx, |ui| {
            let full = ui.max_rect();
            let cy = full.center().y;

            // Lights top-right (KDE/Windows placement), close in the far corner.
            let r = 6.5;
            let gap = 20.0;
            let close_c = Pos2::new(full.right() - 18.0, cy);
            let max_c = Pos2::new(close_c.x - gap, cy);
            let min_c = Pos2::new(max_c.x - gap, cy);

            // Glyphs only appear when the pointer is over the cluster, so the bar
            // stays quiet at rest.
            let cluster = Rect::from_min_max(
                Pos2::new(min_c.x - r - 3.0, full.top()),
                Pos2::new(close_c.x + r + 3.0, full.bottom()),
            );
            let hot = ui.rect_contains_pointer(cluster);

            if light(ui, close_c, r, CLOSE, hot) {
                cmd = Some(egui::ViewportCommand::Close);
            }
            if light(ui, min_c, r, MIN, hot) {
                cmd = Some(egui::ViewportCommand::Minimized(true));
            }
            if light(ui, max_c, r, MAX, hot) {
                cmd = Some(egui::ViewportCommand::Maximized(!maximized));
            }

            ui.painter().text(
                full.center(),
                egui::Align2::CENTER_CENTER,
                title,
                egui::FontId::proportional(13.0),
                Color32::from_rgb(150, 148, 158),
            );

            ui.painter().hline(
                full.x_range(),
                full.bottom(),
                Stroke::new(1.0, Color32::from_rgb(38, 36, 44)),
            );

            // The rest of the bar drags the window. Hand the drag to the
            // compositor on the PRESS itself, not on egui's drag_started() --
            // that waits for a ~6px threshold plus a frame, which under Rune
            // shows up as "a fresh window won't drag".
            let drag = Rect::from_min_max(full.left_top(), Pos2::new(min_c.x - r - 6.0, full.bottom()));
            let resp = ui.interact(drag, ui.id().with("drag"), Sense::click_and_drag());
            if resp.is_pointer_button_down_on() {
                cmd = Some(egui::ViewportCommand::StartDrag);
            }
        });

    if let Some(c) = cmd {
        ctx.send_viewport_cmd(c);
    }
}

fn light(ui: &mut egui::Ui, c: Pos2, r: f32, colour: Color32, hot: bool) -> bool {
    let rect = Rect::from_center_size(c, Vec2::splat(r * 2.4));
    let resp = ui.interact(rect, ui.id().with(c.x as i32), Sense::click());

    let fill = if resp.hovered() { colour } else { colour.linear_multiply(0.82) };
    ui.painter().circle_filled(c, r, fill);

    if hot {
        // A dark glyph inside the dot, drawn small enough to read as a hint.
        let g = r * 0.5;
        let ink = Stroke::new(1.3, Color32::from_rgb(40, 26, 16));
        ui.painter()
            .line_segment([Pos2::new(c.x - g, c.y), Pos2::new(c.x + g, c.y)], ink);
    }

    resp.clicked()
}
