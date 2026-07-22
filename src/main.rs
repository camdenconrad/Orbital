//! Orbital — solar-system simulator with relativistic dynamics.
//! See docs/ISSUE-01-solar-system-sim.md for scope.

mod bodies;
mod chrome;
mod cli;
mod dynamics;
mod ephemeris;
mod mission;
mod porkchop;
mod render;
mod solver;
#[cfg(test)]
mod tests;
mod theme;
mod ui;

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let tour_auto = if args.get(1).map(String::as_str) == Some("tour") {
        Some(cli::parse_search_args())
    } else {
        None
    };
    if args.get(1).map(String::as_str) == Some("search") {
        // Args after the target may appear in any order: a YYYY-MM-DD start
        // date and/or via=body,body flyby routes (parsed in headless_search).
        let date = args[2..]
            .iter()
            .find(|a| a.len() >= 8 && a.chars().next().is_some_and(|c| c.is_ascii_digit()));
        cli::headless_search(args.get(2).map(String::as_str), date.map(String::as_str));
        return Ok(());
    }
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_decorations(false)
            .with_icon(ui::app_icon()),
        depth_buffer: 24,
        ..Default::default()
    };
    eframe::run_native(
        "Orbital",
        native,
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            rs.renderer
                .write()
                .callback_resources
                .insert(render::SceneRenderer::new(&rs.device, rs.target_format));
            theme::apply(&cc.egui_ctx);
            let mut app = ui::App::new(tour_auto);
            app.gpu = Some((rs.device.clone(), rs.queue.clone()));
            Ok(Box::new(app))
        }),
    )
}
