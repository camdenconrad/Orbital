//! Orbital — solar-system simulator with relativistic dynamics.
//! See docs/ISSUE-01-solar-system-sim.md for scope.

mod bodies;
mod dynamics;
mod ephemeris;
mod chrome;
mod porkchop;
mod render;
mod solver;
#[cfg(test)]
mod tests;
mod theme;

use bodies::{BodyId, ALL_BODIES, AU_KM, DAY_S};
use dynamics::{DynamicsConfig, ScState};
use ephemeris::Ephemeris;
use solver::{Engine, Launcher, MissionType, Objective, SolverConfig};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use glam::Vec3;
use hifitime::{Duration, Epoch};
use render::{Camera, LineVertex, Scene, SceneCallback, SphereInstance};

/// Background-built render/search ephemeris table with its valid span.
type FastEphSlot = Arc<std::sync::Mutex<Option<(Arc<Ephemeris>, Epoch, Epoch)>>>;
/// Channel carrying a corrected direct transfer back from its worker.
type DirectRx = std::sync::mpsc::Receiver<(ScState, Vec<(Epoch, ScState)>, f64)>;
/// Channel restoring a saved mission after boot.
type MissionRx = std::sync::mpsc::Receiver<(solver::Solution, SolverConfig)>;

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let tour_auto = if args.get(1).map(String::as_str) == Some("tour") {
        Some(parse_search_args())
    } else {
        None
    };
    if args.get(1).map(String::as_str) == Some("search") {
        // Args after the target may appear in any order: a YYYY-MM-DD start
        // date and/or via=body,body flyby routes (parsed in headless_search).
        let date = args[2..]
            .iter()
            .find(|a| a.len() >= 8 && a.chars().next().is_some_and(|c| c.is_ascii_digit()));
        headless_search(args.get(2).map(String::as_str), date.map(String::as_str));
        return Ok(());
    }
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_decorations(false)
            .with_icon(app_icon()),
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
            let mut app = App::new(tour_auto);
            app.gpu = Some((rs.device.clone(), rs.queue.clone()));
            Ok(Box::new(app))
        }),
    )
}

/// Procedural taskbar icon: sun, tilted orbit ring, and the orange ship —
/// drawn analytically so there is no asset to ship.
fn app_icon() -> egui::IconData {
    const N: usize = 64;
    let mut rgba = vec![0u8; N * N * 4];
    let put = |rgba: &mut Vec<u8>, x: usize, y: usize, c: [f32; 4]| {
        let i = (y * N + x) * 4;
        let a = c[3].clamp(0.0, 1.0);
        for k in 0..3 {
            let src = c[k] * 255.0 * a;
            let dst = rgba[i + k] as f32;
            rgba[i + k] = (src + dst * (1.0 - a)).min(255.0) as u8;
        }
        rgba[i + 3] = ((a + rgba[i + 3] as f32 / 255.0 * (1.0 - a)) * 255.0) as u8;
    };
    let c = (N as f32 - 1.0) / 2.0;
    for y in 0..N {
        for x in 0..N {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            // Dark disc backdrop.
            let r = (dx * dx + dy * dy).sqrt();
            if r < c {
                let edge = ((c - r) / 2.0).clamp(0.0, 1.0);
                put(&mut rgba, x, y, [0.08, 0.08, 0.10, edge]);
            }
            // Tilted orbit ring (ellipse, squashed y).
            let er = (dx * dx + (dy / 0.45) * (dy / 0.45)).sqrt();
            let ring = 1.4 - (er - 22.0).abs();
            if ring > 0.0 {
                put(&mut rgba, x, y, [0.55, 0.42, 0.62, ring.min(1.0) * 0.9]);
            }
            // Sun.
            let sun = 1.0 - (r - 6.5).max(0.0) / 2.0;
            if sun > 0.0 {
                put(&mut rgba, x, y, [1.0, 0.83, 0.35, sun.clamp(0.0, 1.0)]);
            }
            // Ship on the ring at ~35°.
            let (sx, sy2) = (22.0 * 0.819, 0.45 * 22.0 * -0.574);
            let ds = ((dx - sx).powi(2) + (dy - sy2).powi(2)).sqrt();
            let ship = 1.0 - (ds - 3.4).max(0.0) / 1.6;
            if ship > 0.0 {
                put(&mut rgba, x, y, [1.0, 0.58, 0.15, ship.clamp(0.0, 1.0)]);
            }
        }
    }
    egui::IconData {
        rgba,
        width: N as u32,
        height: N as u32,
    }
}

const MISSION_FILE: &str = "mission.orbital";

/// Persist the accepted mission (config + genome + departure) as a plain
/// key=value file — dependency-free and diffable. The trajectory itself is
/// not stored; it re-propagates deterministically on load.
fn save_mission(sol: &solver::Solution, cfg: &SolverConfig) {
    let join = |v: &[f64]| {
        v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(",")
    };
    let mut s = String::new();
    s += &format!("target={}\n", cfg.target.name());
    s += &format!("mission={}\n", cfg.mission.label());
    s += &format!("engine={}\n", match cfg.engine {
        Engine::Ballistic => "ballistic",
        Engine::Nstar => "nstar",
        Engine::NextC => "next",
        Engine::Aeps => "aeps",
    });
    s += &format!("launcher={}\n", match cfg.launcher {
        Launcher::FalconHeavy => "fh",
        Launcher::Sls => "sls",
        Launcher::KickStage => "kick",
    });
    s += &format!(
        "route={}\n",
        cfg.route.iter().map(|b| b.name()).collect::<Vec<_>>().join(",")
    );
    s += &format!("depart_tdb_s={:e}\n", sol.depart.to_tdb_seconds());
    s += &format!("legs={}\n", join(&sol.genome.legs));
    s += &format!("vinf={}\n", join(&sol.genome.vinf_dep));
    s += &format!(
        "thrust={}\n",
        sol.genome
            .thrust
            .iter()
            .map(|u| format!("{:e}:{:e}:{:e}", u[0], u[1], u[2]))
            .collect::<Vec<_>>()
            .join(",")
    );
    s += &format!(
        "dsm={}\n",
        sol.genome
            .dsm
            .iter()
            .map(|d| format!("{:e}:{:e}:{:e}:{:e}", d[0], d[1], d[2], d[3]))
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Err(e) = std::fs::write(MISSION_FILE, s) {
        eprintln!("could not save {MISSION_FILE}: {e}");
    }
}

/// Load the saved mission, if any. Returns (config, genome, departure epoch).
fn load_mission() -> Option<(SolverConfig, solver::Genome, Epoch)> {
    let text = std::fs::read_to_string(MISSION_FILE).ok()?;
    let mut cfg = SolverConfig::default();
    cfg.auto_route = false;
    let mut legs = Vec::new();
    let mut vinf = [0.0f64; 3];
    let mut thrust = Vec::new();
    let mut dsm = Vec::new();
    let mut depart = None;
    let body = |n: &str| ALL_BODIES.iter().find(|b| b.name().eq_ignore_ascii_case(n)).copied();
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        match k {
            "target" => cfg.target = body(v)?,
            "mission" => {
                cfg.mission = *MissionType::ALL.iter().find(|t| t.label() == v)?;
            }
            "engine" => {
                cfg.engine = match v {
                    "nstar" => Engine::Nstar,
                    "next" => Engine::NextC,
                    "aeps" => Engine::Aeps,
                    _ => Engine::Ballistic,
                }
            }
            "launcher" => {
                cfg.launcher = match v {
                    "sls" => Launcher::Sls,
                    "kick" => Launcher::KickStage,
                    _ => Launcher::FalconHeavy,
                }
            }
            "route" => {
                for n in v.split(',').filter(|n| !n.is_empty()) {
                    cfg.route.push(body(n)?);
                }
            }
            "depart_tdb_s" => depart = v.parse::<f64>().ok(),
            "legs" => legs = v.split(',').filter_map(|x| x.parse().ok()).collect(),
            "vinf" => {
                let p: Vec<f64> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                if p.len() == 3 {
                    vinf = [p[0], p[1], p[2]];
                }
            }
            "thrust" => {
                for seg in v.split(',').filter(|s| !s.is_empty()) {
                    let p: Vec<f64> = seg.split(':').filter_map(|x| x.parse().ok()).collect();
                    if p.len() == 3 {
                        thrust.push([p[0], p[1], p[2]]);
                    }
                }
            }
            "dsm" => {
                for node in v.split(',').filter(|s| !s.is_empty()) {
                    let p: Vec<f64> = node.split(':').filter_map(|x| x.parse().ok()).collect();
                    if p.len() == 4 {
                        dsm.push([p[0], p[1], p[2], p[3]]);
                    }
                }
            }
            _ => {}
        }
    }
    if legs.is_empty() {
        return None;
    }
    let depart = Epoch::from_tdb_seconds(depart?);
    Some((
        cfg,
        solver::Genome {
            depart_days: 0.0,
            legs,
            vinf_dep: vinf,
            thrust,
            dsm,
        },
        depart,
    ))
}

/// Parse target/route/mission/date from argv (shared by `search` and `tour`).
fn parse_search_args() -> (SolverConfig, Option<Epoch>) {
    let mut cfg = SolverConfig::default(); // auto_route defaults on
    for arg in std::env::args().skip(2) {
        if let Some(list) = arg.strip_prefix("via=") {
            cfg.auto_route = false;
            if list == "direct" || list == "none" {
                continue;
            }
            for name in list.split(',') {
                match ALL_BODIES.iter().find(|b| b.name().eq_ignore_ascii_case(name)) {
                    Some(b) => cfg.route.push(*b),
                    None => {
                        eprintln!("unknown flyby body {name:?}");
                        std::process::exit(1);
                    }
                }
            }
        }
        if let Some(m) = arg.strip_prefix("mode=") {
            match MissionType::ALL.iter().find(|t| t.label() == m) {
                Some(t) => cfg.mission = *t,
                None => {
                    eprintln!("unknown mode {m:?} (flyby|orbit|land)");
                    std::process::exit(1);
                }
            }
        }
        if let Some(e) = arg.strip_prefix("engine=") {
            cfg.engine = match e {
                "nstar" => Engine::Nstar,
                "next" | "nextc" => Engine::NextC,
                "aeps" => Engine::Aeps,
                _ => Engine::Ballistic,
            };
        }
        if let Some(v) = arg.strip_prefix("beam=") {
            cfg.beam_width = v.parse().unwrap_or(cfg.beam_width);
        }
        if let Some(v) = arg.strip_prefix("seed=") {
            cfg.seed = v.parse().unwrap_or(cfg.seed);
        }
    }
    if let Some(t) = std::env::args().nth(2) {
        if let Some(b) = ALL_BODIES.iter().find(|b| b.name().eq_ignore_ascii_case(&t)) {
            cfg.target = *b;
        }
    }
    let date = std::env::args().skip(2).find(|a| {
        a.len() >= 8 && a.chars().next().is_some_and(|c| c.is_ascii_digit())
    });
    let epoch0 = date.and_then(|s| {
        let parts: Vec<i32> = s.split('-').filter_map(|p| p.parse().ok()).collect();
        match parts[..] {
            [y, m, d] => Some(Epoch::from_gregorian_utc_at_midnight(y, m as u8, d as u8)),
            _ => None,
        }
    });
    cfg.scale_tof_to_target();
    (cfg, epoch0)
}

/// Headless mode: `orbital search [target] [YYYY-MM-DD]` — run the
/// deterministic beam search without a window and print what it finds.
fn headless_search(target: Option<&str>, start: Option<&str>) {
    let (eph, label) = Ephemeris::load();
    println!("ephemeris: {label}");
    let mut cfg = SolverConfig::default(); // auto_route defaults on
    let mut steps: usize = 400;
    let mut restarts: usize = 1;
    // Route: any arg of the form via=venus,earth,earth (checked below).
    for arg in std::env::args().skip(2) {
        if let Some(v) = arg.strip_prefix("steps=") {
            steps = v.parse().unwrap_or(400);
        }
        if let Some(v) = arg.strip_prefix("beam=") {
            cfg.beam_width = v.parse().unwrap_or(cfg.beam_width);
        }
        if let Some(v) = arg.strip_prefix("seed=") {
            cfg.seed = v.parse().unwrap_or(cfg.seed);
        }
        if let Some(v) = arg.strip_prefix("restarts=") {
            restarts = v.parse::<usize>().unwrap_or(1).max(1);
        }
        if let Some(m) = arg.strip_prefix("mode=") {
            match MissionType::ALL.iter().find(|t| t.label() == m) {
                Some(t) => cfg.mission = *t,
                None => {
                    eprintln!("unknown mode {m:?} (flyby|orbit|land)");
                    std::process::exit(1);
                }
            }
        }
        if let Some(e) = arg.strip_prefix("engine=") {
            cfg.engine = match e {
                "nstar" => Engine::Nstar,
                "next" | "nextc" => Engine::NextC,
                "aeps" => Engine::Aeps,
                _ => Engine::Ballistic,
            };
        }
        if let Some(list) = arg.strip_prefix("via=") {
            cfg.auto_route = false;
            if list == "direct" || list == "none" {
                continue;
            }
            for name in list.split(',') {
                match ALL_BODIES
                    .iter()
                    .find(|b| b.name().eq_ignore_ascii_case(name))
                {
                    Some(b) => cfg.route.push(*b),
                    None => {
                        eprintln!("unknown flyby body {name:?}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }
    if let Some(t) = target {
        match ALL_BODIES
            .iter()
            .find(|b| b.name().eq_ignore_ascii_case(t))
        {
            Some(b) => cfg.target = *b,
            None => {
                eprintln!("unknown body {t:?}");
                std::process::exit(1);
            }
        }
    }
    cfg.scale_tof_to_target();
    let epoch0 = match start {
        Some(s) => {
            let parts: Vec<i32> = s.split('-').filter_map(|p| p.parse().ok()).collect();
            match parts[..] {
                [y, m, d] => Epoch::from_gregorian_utc_at_midnight(y, m as u8, d as u8),
                _ => {
                    eprintln!("bad date {s:?}, want YYYY-MM-DD");
                    std::process::exit(1);
                }
            }
        }
        None => Epoch::from_gregorian_utc_at_midnight(2026, 7, 20),
    };
    let route_str = if cfg.auto_route {
        "auto (discovering)".to_string()
    } else if cfg.route.is_empty() {
        "direct".to_string()
    } else {
        cfg.route.iter().map(|b| b.name()).collect::<Vec<_>>().join("-")
    };
    println!(
        "target {} ({}) via {route_str} · objective {} · window {:.0} d · seed {}",
        cfg.target.name(),
        cfg.mission.label(),
        cfg.objective.label(),
        cfg.window_days,
        cfg.seed
    );
    let describe = |sol: &solver::Solution| -> String {
        let (y, mo, d, ..) = sol.depart.to_gregorian_utc();
        let flybys: String = sol
            .flybys
            .iter()
            .map(|f| {
                let (fy, fm, fd, ..) = f.epoch.to_gregorian_utc();
                format!(
                    " | {} {fy:04}-{fm:02}-{fd:02} v∞ {:.1} Δv {:.2} alt {:.0}km",
                    f.body.name(),
                    f.vinf_kms,
                    f.dv_kms,
                    f.periapsis_alt_km
                )
            })
            .collect();
        let sep = if sol.thrust_dv_kms > 0.0 {
            format!(" | SEP Δv {:.1}", sol.thrust_dv_kms)
        } else {
            String::new()
        };
        format!(
            "depart {y:04}-{mo:02}-{d:02} TOF {:5.0} d | v∞ dep {:5.2} arr {:5.2} km/s | arrival Δv {:.2}{sep} | assist Δv {:.2}{flybys}",
            sol.genome.total_tof_days(),
            sol.vinf_dep_kms,
            sol.vinf_arr_kms,
            sol.arrival_dv_kms,
            sol.assist_dv_kms
        )
    };

    let t0 = std::time::Instant::now();
    let mut evals = 0u64;
    let mut overall: Option<solver::Solution> = None;
    if cfg.auto_route {
        // Route discovery: screening → refine → polish, budget from steps=.
        let keep = || true;
        let mut status = |st: String| {
            println!("· {st}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        };
        let mut on_best = |sol: &solver::Solution| {
            println!("  best: score {:.3} | {}", sol.score, describe(sol));
            use std::io::Write;
            let _ = std::io::stdout().flush();
            overall = Some(sol.clone());
        };
        evals += solver::auto_search(
            &eph, &cfg, epoch0, &keep, &mut status, &mut on_best,
            solver::AutoBudget { screen: 120, refine: 600, polish: steps },
        );
        let dt = t0.elapsed().as_secs_f64();
        if let Some(o) = &overall {
            println!("\nBEST: score {:.3} | {}", o.score, describe(o));
        }
        println!("{evals} evals in {dt:.1}s ({:.0} evals/s)", evals as f64 / dt);
        return;
    }
    // Deterministic multi-start: seeds seed, seed+1, … explore independently;
    // the best across all restarts wins.
    for r in 0..restarts {
        let mut cfg_r = cfg.clone();
        cfg_r.seed = cfg.seed + r as u64;
        let mut search = solver::Search::new(&eph, cfg_r, epoch0, None);
        let mut best = f64::INFINITY;
        for step in 0..steps {
            let (e, (score, g)) = search.step(&eph);
            evals += e;
            if score < best {
                best = score;
                let sol = search.solution_for(&eph, &g);
                let better = overall.as_ref().is_none_or(|o| sol.score < o.score);
                if better {
                    println!("seed {} step {step:4}: score {score:7.3} | {}", cfg.seed + r as u64, describe(&sol));
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    overall = Some(sol);
                }
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    if let Some(o) = &overall {
        println!("\nBEST: score {:.3} | {}", o.score, describe(o));
    }
    println!(
        "{restarts} restart(s) × {steps} steps, {evals} evals in {dt:.1}s ({:.0} evals/s)",
        evals as f64 / dt
    );
}

/// One clock tick of simulated time, seconds.
///
/// Determinism (the besom rule): the sim clock is an integer tick count, and
/// the epoch is *derived* — `epoch0 + ticks * TICK_S` — never accumulated from
/// wall-clock frame deltas. Same tick count in, same solar system out,
/// regardless of frame rate, pauses, or host timing.
const TICK_S: f64 = 1.0;

/// Longest orbit-trail arc, days (~38 yr — one Saturn period; outer bodies
/// draw partial arcs rather than clamped-garbage full loops).
const TRAIL_SPAN_MAX_D: f64 = 14_000.0;

struct App {
    eph: Arc<Ephemeris>,
    eph_label: String,
    epoch0: Epoch,
    ticks: i64,
    playing: bool,
    /// Simulated ticks advanced per rendered frame (integer, so replay is exact).
    ticks_per_frame: i64,
    camera: Camera,
    true_scale: bool,
    show_trails: bool,
    focus: BodyId,
    /// Follow the spacecraft instead of a body.
    focus_ship: bool,
    /// True after the user pans: the camera stays where they put it until
    /// they pick a focus again.
    free_pan: bool,
    cfg: DynamicsConfig,

    // Demo spacecraft.
    sc_state: ScState,
    sc_epoch: Epoch,
    sc_days: f64,
    sc_traj: Vec<(Epoch, ScState)>,

    trails: Vec<(BodyId, Vec<[f64; 3]>)>,
    trails_epoch: Epoch,

    solver_cfg: SolverConfig,
    solver: Option<Arc<solver::Shared>>,
    /// Solver fast-path ephemeris, tabulated once at startup on a background
    /// thread: (table, end of its valid span). Searches whose window fits
    /// inside reuse it and start instantly.
    fast_eph: FastEphSlot,
    /// Kept so the UI can show the winning transfer after the search stops.
    solver_best: Option<solver::Solution>,
    gpu: Option<(Arc<eframe::egui_wgpu::wgpu::Device>, Arc<eframe::egui_wgpu::wgpu::Queue>)>,
    porkchop: Option<porkchop::Porkchop>,
    porkchop_tex: Option<egui::TextureHandle>,
    /// Miss distance after differential correction of an accepted transfer.
    refined_miss_km: Option<f64>,
    /// In-flight multi-leg tour refinement (runs on a background thread).
    refine_rx: Option<std::sync::mpsc::Receiver<Option<solver::RefinedTour>>>,
    /// In-flight direct-transfer correction (background thread).
    direct_rx: Option<DirectRx>,
    refined_info: Option<String>,
    /// Set when a solution is accepted: drives the ship marker's behavior
    /// before launch and after arrival.
    accepted: Option<(MissionType, BodyId)>,
    /// The accepted mission, kept for the breakdown panel. Solution fields
    /// are updated in place when background refinement completes.
    mission: Option<(solver::Solution, SolverConfig)>,
    mission_rx: Option<MissionRx>,
    /// One-shot: start a search on the first frame (`tour` launch mode).
    auto_start: bool,
}

impl App {
    fn new(auto: Option<(SolverConfig, Option<Epoch>)>) -> Self {
        let (eph, eph_label) = Ephemeris::load();
        let eph = Arc::new(eph);
        let mut epoch = Epoch::from_gregorian_utc_at_midnight(2026, 7, 19);
        if let Some((_, Some(e0))) = &auto {
            epoch = *e0;
        }
        let r0 = 1.1 * AU_KM;
        let v0 = dynamics::circular_speed_kms(r0);
        let mut app = Self {
            eph,
            eph_label,
            epoch0: epoch,
            ticks: 0,
            playing: true,
            ticks_per_frame: 14_400, // ~10 days/s at 60 fps
            camera: Camera::default(),
            true_scale: true,
            show_trails: true,
            focus: BodyId::Sun,
            focus_ship: false,
            free_pan: false,
            cfg: DynamicsConfig::default(),
            sc_state: ScState {
                pos: [r0, 0.0, 0.0],
                vel: [0.0, v0 * 1.08, 0.0],
            },
            sc_epoch: epoch,
            sc_days: 1200.0,
            sc_traj: Vec::new(),
            trails: Vec::new(),
            trails_epoch: epoch,
            solver_cfg: auto.as_ref().map(|(c, _)| c.clone()).unwrap_or_default(),
            solver: None,
            solver_best: None,
            accepted: None,
            mission: None,
            mission_rx: None,
            gpu: None,
            porkchop: None,
            porkchop_tex: None,
            refined_miss_km: None,
            refine_rx: None,
            direct_rx: None,
            refined_info: None,
            auto_start: auto.is_some(),
            fast_eph: Arc::new(std::sync::Mutex::new(None)),
        };
        // Tabulate planetary positions in the background: ~38 years back (so
        // orbit trails have a full period for everything through Saturn) and
        // 16 years forward (searches). Trails and searches wait on this
        // rather than fall back to raw SPICE walks, which would freeze the UI.
        {
            let eph = app.eph.clone();
            let slot = app.fast_eph.clone();
            let start = epoch;
            std::thread::spawn(move || {
                let t0 = start - Duration::from_days(TRAIL_SPAN_MAX_D + 1.0);
                let end = start + Duration::from_days(16.0 * 365.25);
                let table = Arc::new(eph.cached_span(
                    t0,
                    end,
                    &solver::solver_dynamics().perturbers,
                ));
                *slot.lock().unwrap() = Some((table, t0, end));
            });
        }
        // Restore the saved mission, if any: re-evaluate its genome (fast,
        // deterministic) off-thread and hand the result back.
        if let Some((cfg, genome, depart)) = load_mission() {
            let (tx, rx) = std::sync::mpsc::channel();
            app.mission_rx = Some(rx);
            let eph = app.eph.clone();
            std::thread::spawn(move || {
                let sol = solver::evaluate_saved(&eph, &cfg, depart, &genome);
                let _ = tx.send((sol, cfg));
            });
        }
        app.rebuild_trails();
        app
    }

    fn start_search(&mut self, ctx: &egui::Context) {
        self.start_search_with(ctx, self.solver_cfg.clone(), self.epoch());
    }

    fn start_search_with(&mut self, ctx: &egui::Context, cfg: SolverConfig, epoch0: Epoch) {
        let shared = solver::Shared::new();
        self.solver = Some(shared.clone());
        self.solver_best = None;
        let eph = self.eph.clone();
        // Reuse the startup table when it covers this search's window.
        let need_end =
            epoch0 + Duration::from_days(cfg.window_days + cfg.max_total_tof_days() + 2.0);
        let prebuilt = self
            .fast_eph
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|(table, _, end)| (need_end <= *end).then(|| table.clone()));
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            solver::solver_thread(eph, cfg, epoch0, prebuilt, shared, ctx2);
        });
    }

    /// Body state for *rendering*: reads the startup interpolation table
    /// (~0.03 km error, microseconds) instead of walking the SPK kernels
    /// (~2 ms each with five kernels loaded — per body, per frame, it froze
    /// the UI). Falls back to direct queries outside the table's 16-year span.
    fn render_state(&self, body: BodyId, epoch: Epoch) -> ephemeris::StateVec {
        if let Some((table, t0, end)) = self.fast_eph.lock().unwrap().as_ref() {
            if epoch >= *t0 && epoch <= *end {
                return table.state(body, epoch);
            }
        }
        self.eph.state(body, epoch)
    }

    /// Colorize a porkchop grid, mission-aware (score = dep v∞ + arrival Δv).
    fn porkchop_texture(&self, ctx: &egui::Context, pc: &porkchop::Porkchop) -> egui::TextureHandle {
        let mut scores = vec![f32::INFINITY; pc.grid.len()];
        let mut lo = f32::INFINITY;
        for (k, cell) in pc.grid.iter().enumerate() {
            if cell[0] < 1e6 {
                let arr = solver::arrival_dv_kms(
                    self.solver_cfg.target,
                    cell[1] as f64,
                    self.solver_cfg.mission,
                ) as f32;
                scores[k] = cell[0] + arr;
                lo = lo.min(scores[k]);
            }
        }
        let hi = lo + 8.0; // km/s of visible dynamic range above the minimum
        let mut img = egui::ColorImage::new([porkchop::NX, porkchop::NY], egui::Color32::BLACK);
        for j in 0..porkchop::NY {
            for i in 0..porkchop::NX {
                let sc = scores[j * porkchop::NX + i];
                let c = if sc.is_finite() {
                    let t = ((sc - lo) / (hi - lo)).clamp(0.0, 1.0);
                    // Low cost bright: yellow → teal → deep blue.
                    let (r, g, b) = if t < 0.5 {
                        let u = t * 2.0;
                        (1.0 - 0.8 * u, 0.85 - 0.25 * u, 0.25 + 0.35 * u)
                    } else {
                        let u = (t - 0.5) * 2.0;
                        (0.2 - 0.15 * u, 0.6 - 0.45 * u, 0.6 - 0.25 * u)
                    };
                    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                } else {
                    egui::Color32::from_rgb(18, 18, 22)
                };
                // Flip vertically: TOF increases upward.
                img.pixels[(porkchop::NY - 1 - j) * porkchop::NX + i] = c;
            }
        }
        ctx.load_texture("porkchop", img, egui::TextureOptions::LINEAR)
    }

    /// The accepted mission's full breakdown: schedule, flybys, resources.
    fn mission_ui(&self, ui: &mut egui::Ui, sol: &solver::Solution, cfg: &SolverConfig) {
        let date = |e: Epoch| {
            let (y, m, d, ..) = e.to_gregorian_utc();
            format!("{y:04}-{m:02}-{d:02}")
        };
        let dur = sol.arrive - sol.depart;
        let days = dur.to_seconds() / DAY_S;
        let route = if cfg.route.is_empty() {
            "direct".to_string()
        } else {
            cfg.route.iter().map(|b| b.name()).collect::<Vec<_>>().join(" → ")
        };
        let mut rows: Vec<(String, String)> = vec![
            (
                "profile".into(),
                format!("{} {} · {route}", cfg.target.name(), cfg.mission.label()),
            ),
            (
                "hardware".into(),
                format!("{} · {}", cfg.launcher.label(), cfg.engine.label()),
            ),
            ("depart".into(), date(sol.depart)),
            (
                "launch".into(),
                format!(
                    "v∞ {:.2} km/s · C3 {:.0} of {:.0}",
                    sol.vinf_dep_kms,
                    sol.vinf_dep_kms * sol.vinf_dep_kms,
                    cfg.launcher.c3_max()
                ),
            ),
        ];
        for f in &sol.flybys {
            rows.push((
                format!("flyby {}", f.body.name()),
                format!(
                    "{} · v∞ {:.1} · {:.0} km alt{}",
                    date(f.epoch),
                    f.vinf_kms,
                    f.periapsis_alt_km,
                    if f.dv_kms > 0.005 {
                        format!(" · Δv {:.2}", f.dv_kms)
                    } else {
                        " · free".into()
                    }
                ),
            ));
        }
        if sol.thrust_dv_kms > 0.0 {
            // Rocket-equation propellant on the modeled 1500 kg / 2-engine bus.
            let (_, isp) = match cfg.engine {
                Engine::Nstar => (0.0, 3100.0),
                Engine::NextC => (0.0, 4190.0),
                Engine::Aeps => (0.0, 2900.0),
                Engine::Ballistic => (0.0, 1.0),
            };
            let ve = isp * 9.80665e-3;
            let m_prop = 1500.0 * (1.0 - (-sol.thrust_dv_kms / ve).exp());
            let burn_days = sol.thrust_dv_kms / cfg.engine.accel_kms2() / DAY_S;
            rows.push((
                "SEP cruise".into(),
                format!(
                    "Δv {:.1} of {:.1} km/s · {m_prop:.0} kg xenon · {burn_days:.0} d thrust-on",
                    sol.thrust_dv_kms,
                    cfg.engine.max_dv_kms()
                ),
            ));
        }
        // Arrival burn on storable bi-prop (Isp 320 s) from the mass left
        // after the cruise.
        let m_after_sep = if sol.thrust_dv_kms > 0.0 {
            let ve = match cfg.engine {
                Engine::Nstar => 3100.0,
                Engine::NextC => 4190.0,
                Engine::Aeps => 2900.0,
                Engine::Ballistic => 1.0,
            } * 9.80665e-3;
            1500.0 * (-sol.thrust_dv_kms / ve).exp()
        } else {
            1500.0
        };
        let m_capture = m_after_sep * (1.0 - (-sol.arrival_dv_kms / (320.0 * 9.80665e-3)).exp());
        rows.push((
            "arrive".into(),
            format!("{} · v∞ {:.2} km/s", date(sol.arrive), sol.vinf_arr_kms),
        ));
        if sol.arrival_dv_kms > 0.001 {
            rows.push((
                format!("{} burn", cfg.mission.label()),
                format!("Δv {:.2} km/s · {m_capture:.0} kg bi-prop", sol.arrival_dv_kms),
            ));
        }
        rows.push((
            "duration".into(),
            format!("{:.1} yr ({days:.0} d)", days / 365.25),
        ));
        if sol.dsm_dv_kms > 0.001 {
            rows.push((
                "deep-space maneuvers".into(),
                format!("Δv {:.2} km/s", sol.dsm_dv_kms),
            ));
        }
        let total =
            sol.assist_dv_kms + sol.thrust_dv_kms + sol.dsm_dv_kms + sol.arrival_dv_kms;
        rows.push(("post-launch Δv".into(), format!("{total:.2} km/s")));

        egui::Grid::new("mission-plan")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                for (k, v) in &rows {
                    ui.label(egui::RichText::new(k).weak());
                    ui.label(v);
                    ui.end_row();
                }
            });
        let c3 = sol.vinf_dep_kms * sol.vinf_dep_kms;
        if c3 > cfg.launcher.c3_max() {
            ui.colored_label(theme::BAD, "⚠ exceeds launcher C3");
        }
        if sol.thrust_dv_kms > cfg.engine.max_dv_kms() {
            ui.colored_label(theme::BAD, "⚠ exceeds propellant budget");
        }
        if cfg.engine == Engine::Ballistic && sol.arrival_dv_kms > 4.0 {
            ui.colored_label(theme::BAD, "⚠ capture beyond chemical propulsion");
        }
        if let Some(miss) = self.refined_miss_km {
            ui.label(
                egui::RichText::new(format!("targeting residual {miss:.0} km"))
                    .weak()
                    .small(),
            );
        }
    }

    /// Current simulation epoch, derived from the tick count.
    fn epoch(&self) -> Epoch {
        self.epoch0 + Duration::from_seconds(self.ticks as f64 * TICK_S)
    }

    /// Sample each heliocentric body's trailing arc: up to one orbital
    /// period, capped to the tabulated span so no sample ever clamps to the
    /// table edge (clamped samples collapse to a point and render as long
    /// straight chords). Waits for the background table — raw SPICE sampling
    /// here would freeze the UI for seconds.
    fn rebuild_trails(&mut self) {
        let bounds = self
            .fast_eph
            .lock()
            .unwrap()
            .as_ref()
            .map(|(_, t0, end)| (*t0, *end));
        let (t0, tend) = match bounds {
            Some(b) => b,
            None => {
                if matches!(*self.eph, Ephemeris::Kepler) {
                    // Analytic fallback is cheap; sample without a table.
                    (Epoch::from_gregorian_utc_at_midnight(1900, 1, 1),
                     Epoch::from_gregorian_utc_at_midnight(2200, 1, 1))
                } else {
                    self.trails.clear();
                    return; // table not built yet; retried next frame
                }
            }
        };
        let epoch = self.epoch();
        self.trails_epoch = epoch;
        self.trails.clear();
        for body in ALL_BODIES {
            // Trails only for heliocentric bodies; moons vanish at solar scale.
            if body == BodyId::Sun || body.parent().is_some() {
                continue;
            }
            let arc_d = body.period_days().abs().min(TRAIL_SPAN_MAX_D);
            let n = 256;
            let pts: Vec<[f64; 3]> = (0..=n)
                .filter_map(|i| {
                    let e = epoch + Duration::from_days(arc_d * (i as f64 / n as f64 - 1.0));
                    (e >= t0 && e <= tend).then(|| self.render_state(body, e).pos_km)
                })
                .collect();
            if pts.len() >= 2 {
                self.trails.push((body, pts));
            }
        }
    }

    fn propagate_spacecraft(&mut self) {
        self.sc_epoch = self.epoch();
        self.sc_traj = dynamics::propagate(
            &self.eph,
            &self.cfg,
            self.epoch(),
            self.sc_state,
            Duration::from_days(self.sc_days),
            2000,
        );
    }

    /// Where the ship is right now: on Earth before launch, along the
    /// trajectory in flight, then with the target (orbit/land) or coasting
    /// (flyby) after arrival. None if no trajectory exists yet.
    fn ship_pos_km(&self) -> Option<[f64; 3]> {
        if self.sc_traj.is_empty() {
            return None;
        }
        let epoch = self.epoch();
        let launch = self.sc_traj.first().unwrap().0;
        let arrival = self.sc_traj.last().unwrap().0;
        Some(if epoch <= launch {
            self.render_state(BodyId::Earth, epoch).pos_km
        } else if epoch >= arrival {
            match self.accepted {
                Some((m, target)) if m != solver::MissionType::Flyby => {
                    self.render_state(target, epoch).pos_km
                }
                _ => self.sc_traj.last().unwrap().1.pos,
            }
        } else {
            let i = self
                .sc_traj
                .partition_point(|(e, _)| *e <= epoch)
                .clamp(1, self.sc_traj.len() - 1);
            let (e0, s0) = &self.sc_traj[i - 1];
            let (e1, s1) = &self.sc_traj[i];
            let dt = (*e1 - *e0).to_seconds();
            let f = ((epoch - *e0).to_seconds() / dt).clamp(0.0, 1.0);
            // Cubic Hermite with the samples' velocities: no chord-to-chord
            // lurching even where samples are sparse.
            let (f2, f3) = (f * f, f * f * f);
            let (h00, h10, h01, h11) = (
                2.0 * f3 - 3.0 * f2 + 1.0,
                f3 - 2.0 * f2 + f,
                -2.0 * f3 + 3.0 * f2,
                f3 - f2,
            );
            [
                h00 * s0.pos[0] + h10 * dt * s0.vel[0] + h01 * s1.pos[0] + h11 * dt * s1.vel[0],
                h00 * s0.pos[1] + h10 * dt * s0.vel[1] + h01 * s1.pos[1] + h11 * dt * s1.vel[1],
                h00 * s0.pos[2] + h10 * dt * s0.vel[2] + h01 * s1.pos[2] + h11 * dt * s1.vel[2],
            ]
        })
    }

    fn focus_pos(&self) -> Vec3 {
        let p = if self.focus_ship {
            self.ship_pos_km()
                .unwrap_or_else(|| self.render_state(BodyId::Earth, self.epoch()).pos_km)
        } else {
            self.render_state(self.focus, self.epoch()).pos_km
        };
        Vec3::new(
            (p[0] / AU_KM) as f32,
            (p[1] / AU_KM) as f32,
            (p[2] / AU_KM) as f32,
        )
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Orbital");
        ui.label(egui::RichText::new(&self.eph_label).weak().small());
        ui.separator();

        let (y, m, d, hh, mm, _, _) = self.epoch().to_gregorian_utc();
        ui.label(format!("Epoch (UTC): {y:04}-{m:02}-{d:02} {hh:02}:{mm:02}"));
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "⏸ Pause" } else { "▶ Play" }).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("Now").clicked() {
                self.ticks = 0;
                self.rebuild_trails();
            }
        });
        ui.add(
            egui::Slider::new(&mut self.ticks_per_frame, 1..=20_000_000)
                .logarithmic(true)
                .text("ticks/frame"),
        );
        ui.label(format!(
            "≈{:.2} days/s at 60 fps · tick = {TICK_S} s",
            self.ticks_per_frame as f64 * TICK_S * 60.0 / DAY_S
        ));
        ui.separator();

        ui.checkbox(&mut self.true_scale, "true-scale radii");
        ui.checkbox(&mut self.show_trails, "orbit trails");
        ui.checkbox(&mut self.cfg.relativity, "relativity (1PN)");
        egui::ComboBox::from_label("focus")
            .selected_text(self.focus.name())
            .show_ui(ui, |ui| {
                for b in ALL_BODIES {
                    ui.selectable_value(&mut self.focus, b, b.name());
                }
            });
        ui.separator();

        ui.heading("Spacecraft");
        ui.label("Heliocentric state (km, km/s):");
        egui::Grid::new("sc-state").num_columns(2).show(ui, |ui| {
            for (i, label) in ["x", "y", "z"].iter().enumerate() {
                ui.label(*label);
                ui.add(egui::DragValue::new(&mut self.sc_state.pos[i]).speed(1e6));
                ui.end_row();
            }
            for (i, label) in ["vx", "vy", "vz"].iter().enumerate() {
                ui.label(*label);
                ui.add(egui::DragValue::new(&mut self.sc_state.vel[i]).speed(0.1));
                ui.end_row();
            }
        });
        ui.add(egui::Slider::new(&mut self.sc_days, 10.0..=10_000.0).logarithmic(true).text("days"));
        if ui.button("Propagate from current epoch").clicked() {
            self.propagate_spacecraft();
        }
        ui.separator();
        ui.heading("Trajectory solver");
        let busy = self.solver.is_some();
        ui.add_enabled_ui(!busy, |ui| {
            let prev_target = self.solver_cfg.target;
            egui::ComboBox::from_label("target")
                .selected_text(self.solver_cfg.target.name())
                .show_ui(ui, |ui| {
                    for b in ALL_BODIES {
                        if b != BodyId::Sun && b != BodyId::Earth && b.parent().is_none() {
                            ui.selectable_value(&mut self.solver_cfg.target, b, b.name());
                        }
                    }
                });
            if self.solver_cfg.target != prev_target {
                // Reset the flight-time box to something physical for the
                // new target — a 600-day ceiling to Pluto only offers absurd
                // hyperbolic sprints.
                self.solver_cfg.tof_min_days = 60.0;
                self.solver_cfg.tof_max_days = 600.0;
                self.solver_cfg.scale_tof_to_target();
            }
            {
                let presets: [(&str, &[BodyId]); 6] = [
                    ("direct", &[]),
                    ("Venus assist", &[BodyId::Venus]),
                    ("Venus-Earth", &[BodyId::Venus, BodyId::Earth]),
                    ("VEEGA (V-E-E)", &[BodyId::Venus, BodyId::Earth, BodyId::Earth]),
                    ("Earth return (ΔVEGA)", &[BodyId::Earth]),
                    ("Mars assist", &[BodyId::Mars]),
                ];
                let current = if self.solver_cfg.auto_route {
                    "auto (discover)"
                } else {
                    presets
                        .iter()
                        .find(|(_, r)| *r == self.solver_cfg.route.as_slice())
                        .map(|(n, _)| *n)
                        .unwrap_or("custom")
                };
                egui::ComboBox::from_label("route")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(self.solver_cfg.auto_route, "auto (discover)")
                            .clicked()
                        {
                            self.solver_cfg.auto_route = true;
                            self.solver_cfg.route.clear();
                        }
                        for (name, route) in presets {
                            if ui
                                .selectable_label(
                                    !self.solver_cfg.auto_route
                                        && self.solver_cfg.route.as_slice() == route,
                                    name,
                                )
                                .clicked()
                            {
                                self.solver_cfg.auto_route = false;
                                self.solver_cfg.route = route.to_vec();
                            }
                        }
                    });
            }
            egui::ComboBox::from_label("objective")
                .selected_text(self.solver_cfg.objective.label())
                .show_ui(ui, |ui| {
                    for o in Objective::ALL {
                        ui.selectable_value(&mut self.solver_cfg.objective, o, o.label());
                    }
                });
            egui::ComboBox::from_label("at target")
                .selected_text(self.solver_cfg.mission.label())
                .show_ui(ui, |ui| {
                    for m in MissionType::ALL {
                        ui.selectable_value(&mut self.solver_cfg.mission, m, m.label());
                    }
                });
            egui::ComboBox::from_label("propulsion")
                .selected_text(self.solver_cfg.engine.label())
                .show_ui(ui, |ui| {
                    for e in Engine::ALL {
                        ui.selectable_value(&mut self.solver_cfg.engine, e, e.label());
                    }
                });
            egui::ComboBox::from_label("launcher")
                .selected_text(self.solver_cfg.launcher.label())
                .show_ui(ui, |ui| {
                    for l in Launcher::ALL {
                        ui.selectable_value(&mut self.solver_cfg.launcher, l, l.label());
                    }
                });
            ui.add(
                egui::Slider::new(&mut self.solver_cfg.window_days, 30.0..=3000.0)
                    .text("depart window (d)"),
            );
            ui.add(
                egui::Slider::new(&mut self.solver_cfg.tof_max_days, 100.0..=30_000.0)
                    .logarithmic(true)
                    .text("max TOF (d)"),
            );
            ui.horizontal(|ui| {
                ui.label("seed");
                ui.add(egui::DragValue::new(&mut self.solver_cfg.seed));
            });
        });
        if !busy {
            if ui.button("▶ Search").clicked() {
                self.start_search(ctx);
            }
        } else if ui.button("⏹ Stop").clicked() {
            if let Some(sh) = self.solver.take() {
                sh.running.store(false, Ordering::Relaxed);
            }
        }
        if let Some(sh) = &self.solver {
            let steps = sh.steps.load(Ordering::Relaxed);
            if steps == 0 && self.solver_best.is_none() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("sampling ephemeris tables…");
                });
                // Keep painting while the worker warms up, so the spinner
                // spins and the first best appears without a nudge.
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            } else {
                let status = sh.status.lock().unwrap().clone();
                if !status.is_empty() {
                    ui.label(status);
                }
                ui.label(format!(
                    "step {steps} · {} evals",
                    sh.evals.load(Ordering::Relaxed)
                ));
                // Keep stats and the dotted preview fresh even when the sim
                // clock is paused.
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
            }
        }
        if let Some(best) = self.solver_best.clone() {
            let (y, m, d, ..) = best.depart.to_gregorian_utc();
            let (ay, am, ad, ..) = best.arrive.to_gregorian_utc();
            let mut text = format!(
                "depart {y:04}-{m:02}-{d:02}\narrive {ay:04}-{am:02}-{ad:02}\nTOF {:.0} d\nv∞ dep {:.2} km/s\nv∞ arr {:.2} km/s",
                best.genome.total_tof_days(),
                best.vinf_dep_kms,
                best.vinf_arr_kms
            );
            text += &format!(
                "\narrival Δv ({}) {:.2} km/s",
                self.solver_cfg.mission.label(),
                best.arrival_dv_kms
            );
            if best.thrust_dv_kms > 0.0 {
                text += &format!(
                    "\nSEP Δv used {:.1} / {:.1} km/s budget",
                    best.thrust_dv_kms,
                    self.solver_cfg.engine.max_dv_kms()
                );
            }
            // Physical honesty: numbers this hardware could not buy.
            if best.arrival_dv_kms > 4.0 {
                text += "\n⚠ arrival Δv beyond chemical propulsion";
            }
            let c3 = best.vinf_dep_kms * best.vinf_dep_kms;
            if c3 > self.solver_cfg.launcher.c3_max() {
                text += &format!(
                    "\n⚠ C3 {c3:.0} exceeds {}",
                    self.solver_cfg.launcher.label()
                );
            }
            if best.thrust_dv_kms > self.solver_cfg.engine.max_dv_kms() {
                text += "\n⚠ propellant budget exceeded";
            }
            if best.flybys.is_empty() {
                text += &format!("\nmiss {:.0} km", best.miss_km);
            } else {
                text += &format!("\nassist Δv {:.2} km/s", best.assist_dv_kms);
                for f in &best.flybys {
                    let (fy, fm, fd, ..) = f.epoch.to_gregorian_utc();
                    text += &format!(
                        "\n {} {fy:04}-{fm:02}-{fd:02} v∞ {:.1} alt {:.0} km",
                        f.body.name(),
                        f.vinf_kms,
                        f.periapsis_alt_km
                    );
                }
            }
            ui.monospace(text);
            if ui.button("Accept → full-fidelity propagate").clicked() {
                if let Some(sh) = self.solver.take() {
                    sh.running.store(false, Ordering::Relaxed);
                }
                let (_, s0) = best.traj[0];
                self.sc_state = s0;
                self.sc_epoch = best.depart;
                self.sc_days = best.genome.total_tof_days();
                if best.flybys.is_empty() && !best.genome.thrust.is_empty() {
                    // Low-thrust: re-fly densely with the throttle profile at
                    // tight tolerance in the background (no Newton pass — the
                    // profile itself is the control).
                    self.sc_traj = best.traj.clone();
                    let (tx, rx) = std::sync::mpsc::channel();
                    self.direct_rx = Some(rx);
                    let eph = self.eph.clone();
                    let dyn_cfg = self.cfg;
                    let engine = self.solver_cfg.engine;
                    let genome = best.genome.clone();
                    let depart = best.depart;
                    let miss = best.miss_km;
                    let (_, s0) = best.traj[0];
                    let ctx2 = ctx.clone();
                    std::thread::spawn(move || {
                        let total_s = genome.legs[0] * DAY_S;
                        let thrust = dynamics::Thrust {
                            segs: &genome.thrust,
                            accel_kms2: engine.accel_kms2(),
                            total_s,
                        };
                        let traj = dynamics::propagate_thrusted(
                            &eph,
                            &dyn_cfg,
                            depart,
                            s0,
                            Duration::from_seconds(total_s),
                            2000,
                            Some(&thrust),
                        );
                        let _ = tx.send((s0, traj, miss));
                        ctx2.request_repaint();
                    });
                } else if best.flybys.is_empty() {
                    // Phase two runs in the background — ~20 full-fidelity
                    // propagations would freeze the UI for a minute if done
                    // here. The scouted path shows immediately; the corrected
                    // one swaps in when ready.
                    self.sc_traj = best.traj.clone();
                    let (tx, rx) = std::sync::mpsc::channel();
                    self.direct_rx = Some(rx);
                    let eph = self.eph.clone();
                    let dyn_cfg = self.cfg;
                    let target = self.solver_cfg.target;
                    let depart = best.depart;
                    let tof = Duration::from_days(best.genome.total_tof_days());
                    let vinf0 = best.genome.vinf_dep;
                    let ctx2 = ctx.clone();
                    std::thread::spawn(move || {
                        let (vinf, miss) = solver::differential_correct(
                            &eph, &dyn_cfg, depart, tof, target, vinf0,
                        );
                        let earth = eph.state(BodyId::Earth, depart);
                        let vmag = (vinf[0].powi(2) + vinf[1].powi(2) + vinf[2].powi(2))
                            .sqrt()
                            .max(0.05);
                        let dir = [vinf[0] / vmag, vinf[1] / vmag, vinf[2] / vmag];
                        let s0c = ScState {
                            pos: [
                                earth.pos_km[0] + dir[0] * 925_000.0,
                                earth.pos_km[1] + dir[1] * 925_000.0,
                                earth.pos_km[2] + dir[2] * 925_000.0,
                            ],
                            vel: [
                                earth.vel_km_s[0] + vinf[0],
                                earth.vel_km_s[1] + vinf[1],
                                earth.vel_km_s[2] + vinf[2],
                            ],
                        };
                        let traj =
                            dynamics::propagate(&eph, &dyn_cfg, depart, s0c, tof, 2000);
                        let _ = tx.send((s0c, traj, miss));
                        ctx2.request_repaint();
                    });
                } else {
                    // Show the patched-conic path immediately; multi-leg
                    // shooting refines it to mission grade in the background
                    // and swaps the real n-body path in when done.
                    self.sc_traj = best.traj.clone();
                    let (tx, rx) = std::sync::mpsc::channel();
                    self.refine_rx = Some(rx);
                    self.refined_info = None;
                    let eph = self.eph.clone();
                    let cfg = self.solver_cfg.clone();
                    let genome = best.genome.clone();
                    // Reconstruct the search's epoch0 from the solution: the
                    // genome's depart offset is relative to it.
                    let epoch0 = best.depart
                        - Duration::from_seconds(best.genome.depart_days * DAY_S);
                    let ctx2 = ctx.clone();
                    std::thread::spawn(move || {
                        let out = solver::refine_tour(&eph, &cfg, epoch0, &genome);
                        let _ = tx.send(out);
                        ctx2.request_repaint();
                    });
                }
                self.accepted = Some((self.solver_cfg.mission, self.solver_cfg.target));
                self.mission = Some((best.clone(), self.solver_cfg.clone()));
                save_mission(&best, &self.solver_cfg);
                self.focus_ship = true;
                self.free_pan = false;
                // Cut to just before launch and roll time so the ship visibly
                // departs Earth and arrives; whole mission ≈ 1 minute on screen.
                let launch_ticks =
                    ((best.depart - self.epoch0).to_seconds() / TICK_S) as i64;
                self.ticks = launch_ticks - 400_000; // ~5 s of pre-launch
                let total_s = best.genome.total_tof_days() * DAY_S;
                self.ticks_per_frame = ((total_s / (60.0 * 60.0)) as i64).max(1);
                self.playing = true;
                self.solver_best = None;
            }
        }
        ui.separator();
        ui.heading("Mission plan");
        if let Some((sol, cfg)) = self.mission.clone() {
            self.mission_ui(ui, &sol, &cfg);
        } else {
            ui.label(
                egui::RichText::new(
                    "run a search, then Accept — the full breakdown (dates, flybys, propellant, duration) appears here and is saved to mission.orbital",
                )
                .weak()
                .small(),
            );
        }
        ui.separator();
        ui.heading("Porkchop (GPU)");
        ui.label(
            egui::RichText::new("Lambert over the whole window on the RTX; click a cell to refine it on CPU")
                .weak()
                .small(),
        );
        if ui.button("⚡ Compute").clicked() {
            if let Some((dev, q)) = self.gpu.clone() {
                let table = self.fast_eph.lock().unwrap().as_ref().map(|(t, _, _)| t.clone());
                let eph_ref: &Ephemeris = table.as_deref().unwrap_or(&self.eph);
                match porkchop::compute(
                    &dev,
                    &q,
                    eph_ref,
                    self.solver_cfg.target,
                    porkchop::GridSpec {
                        start: self.epoch(),
                        window_days: self.solver_cfg.window_days,
                        tof_min_days: self.solver_cfg.tof_min_days,
                        tof_max_days: self.solver_cfg.tof_max_days,
                    },
                ) {
                    Some(pc) => {
                        self.porkchop_tex = Some(self.porkchop_texture(ctx, &pc));
                        self.porkchop = Some(pc);
                    }
                    None => {
                        self.porkchop = None;
                        self.porkchop_tex = None;
                    }
                }
            }
        }
        if let (Some(pc_tex), Some(pc)) = (&self.porkchop_tex, &self.porkchop) {
            let w = ui.available_width().min(360.0);
            let h = w * porkchop::NY as f32 / porkchop::NX as f32;
            let resp = ui.add(
                egui::Image::new((pc_tex.id(), egui::vec2(w, h)))
                    .sense(egui::Sense::click()),
            );
            ui.label(
                egui::RichText::new("x: departure date → · y: flight time ↑")
                    .weak()
                    .small(),
            );
            if let Some(pos) = resp.hover_pos() {
                let rel = (pos - resp.rect.min) / egui::vec2(resp.rect.width(), resp.rect.height());
                let i = ((rel.x * porkchop::NX as f32) as usize).min(porkchop::NX - 1);
                let j_img = ((rel.y * porkchop::NY as f32) as usize).min(porkchop::NY - 1);
                let j = porkchop::NY - 1 - j_img; // image y is flipped
                let [vd, va] = pc.cell(i, j);
                let (dep_d, tof_d) = pc.cell_time(i, j);
                let (y, m, d, ..) = (pc.start + Duration::from_days(dep_d)).to_gregorian_utc();
                if vd < 1e6 {
                    let arr_dv = solver::arrival_dv_kms(
                        self.solver_cfg.target,
                        va as f64,
                        self.solver_cfg.mission,
                    );
                    resp.clone().on_hover_text(format!(
                        "depart {y:04}-{m:02}-{d:02} · TOF {tof_d:.0} d
v∞ dep {vd:.2} arr {va:.2} km/s
score {:.2}",
                        vd as f64 + arr_dv
                    ));
                } else {
                    resp.clone().on_hover_text("no single-rev solution");
                }
                if resp.clicked() && vd < 1e6 {
                    // CPU f64 refinement around the picked cell — the GPU only
                    // proposed; determinism lives on this side.
                    let mut cfg = self.solver_cfg.clone();
                    cfg.auto_route = false;
                    cfg.route.clear();
                    cfg.window_days = 40.0;
                    cfg.tof_min_days = (tof_d - 60.0).max(20.0);
                    cfg.tof_max_days = tof_d + 60.0;
                    let epoch0 = pc.start + Duration::from_days((dep_d - 20.0).max(0.0));
                    if let Some(sh) = self.solver.take() {
                        sh.running.store(false, Ordering::Relaxed);
                    }
                    self.start_search_with(ctx, cfg, epoch0);
                }
            }
        }
        ui.separator();
        if !self.sc_traj.is_empty() {
            let (_, last) = self.sc_traj.last().unwrap();
            let r = (last.pos[0].powi(2) + last.pos[1].powi(2) + last.pos[2].powi(2)).sqrt();
            ui.label(format!(
                "final r = {:.4} AU, {} pts",
                r / AU_KM,
                self.sc_traj.len()
            ));
            if let Some(miss) = self.refined_miss_km {
                ui.label(format!("corrected: hits target to {miss:.0} km"));
            }
            if self.refine_rx.is_some() || self.direct_rx.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(if self.refine_rx.is_some() {
                        "multi-leg shooting (full n-body)…"
                    } else {
                        "differential correction (full n-body)…"
                    });
                });
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
            if let Some(info) = &self.refined_info {
                ui.monospace(info);
            }
        }
    }

    fn build_scene(&self, aspect: f32) -> Scene {
        let epoch = self.epoch();
        let mut scene = Scene {
            view_proj: self.camera.view_proj(aspect),
            ..Default::default()
        };
        let au = |p: [f64; 3]| {
            [
                (p[0] / AU_KM) as f32,
                (p[1] / AU_KM) as f32,
                (p[2] / AU_KM) as f32,
            ]
        };
        for body in ALL_BODIES {
            // Moons: only draw once the camera is close enough that they
            // separate from their planet's dot — at solar zoom the planet's
            // visibility floor covers their whole orbit and they just
            // z-fight with it.
            if let Some(_) = body.parent() {
                let orbit_au = (body.moon_orbit_radius_km() / AU_KM) as f32;
                if self.camera.distance > orbit_au * 80.0 {
                    continue;
                }
            }
            let s = self.render_state(body, epoch);
            let true_r = (body.radius_km() / AU_KM) as f32;
            let floor = if body.parent().is_some() {
                // Smaller floor for moons so they stay distinct from the
                // planet at the zoom where they first appear.
                self.camera.distance * 0.0015
            } else {
                self.camera.distance * 0.0035
            };
            let radius = if self.true_scale {
                // To scale, but never smaller than a few pixels — otherwise
                // everything but the Sun vanishes at solar zoom. Zoom in and
                // the true radius takes over.
                true_r.max(floor)
            } else {
                (true_r * 400.0).clamp(0.004, 0.05)
            };
            scene.spheres.push(SphereInstance {
                center: au(s.pos_km),
                radius,
                color: body.color(),
                emissive: if body == BodyId::Sun { 1.0 } else { 0.0 },
            });
        }
        if self.show_trails {
            for (body, pts) in &self.trails {
                let start = scene.line_verts.len() as u32;
                let c = body.color();
                let n = pts.len();
                for (i, p) in pts.iter().enumerate() {
                    let fade = 0.08 + 0.5 * (i as f32 / n as f32);
                    scene.line_verts.push(LineVertex {
                        pos: au(*p),

                        color: [c[0], c[1], c[2], fade],
                    });
                }
                scene.strips.push((start, n as u32));
            }
        }
        if !self.sc_traj.is_empty() {
            let start = scene.line_verts.len() as u32;
            for (_, st) in &self.sc_traj {
                scene.line_verts.push(LineVertex {
                    pos: au(st.pos),

                    color: [0.91, 0.58, 0.28, 0.9], // theme ACCENT
                });
            }
            scene.strips.push((start, self.sc_traj.len() as u32));
            // The ship marker: screen-constant size, so it stays visible at
            // solar zoom yet doesn't dwarf true-scale planets up close.
            if let Some(ship_pos) = self.ship_pos_km() {
                // Bright core + soft halo so the ship reads at any zoom.
                scene.spheres.push(SphereInstance {
                    center: au(ship_pos),
                    radius: (self.camera.distance * 0.011).max(5e-7),
                    color: [1.0, 0.62, 0.15],
                    emissive: 1.0,
                });
                scene.spheres.push(SphereInstance {
                    center: au(ship_pos),
                    radius: (self.camera.distance * 0.0045).max(2e-7),
                    color: [1.0, 0.92, 0.75],
                    emissive: 1.0,
                });
            }
        }
        // The search's current best, live: a thick solid bright-orange ribbon
        // (camera-facing triangle strip — wgpu lines are always 1 px).
        if let Some(best) = &self.solver_best {
            const ORANGE: [f32; 4] = [1.0, 0.55, 0.1, 1.0];
            let (sy, cy) = self.camera.yaw.sin_cos();
            let (sp, cp) = self.camera.pitch.sin_cos();
            let view = Vec3::new(cp * cy, cp * sy, sp); // eye direction
            let w = self.camera.distance * 0.0018; // screen-constant width
            let pts: Vec<Vec3> = best
                .traj
                .iter()
                .map(|(_, st)| {
                    let p = au(st.pos);
                    Vec3::new(p[0], p[1], p[2])
                })
                .collect();
            for seg in pts.windows(2) {
                let dir = seg[1] - seg[0];
                let perp = dir.cross(view);
                let n = perp.length();
                if n < 1e-12 {
                    continue;
                }
                let off = perp * (w / n);
                let quad = [
                    seg[0] - off,
                    seg[0] + off,
                    seg[1] - off,
                    seg[1] + off,
                ];
                for &v in &[quad[0], quad[1], quad[2], quad[2], quad[1], quad[3]] {
                    scene.tri_verts.push(LineVertex {
                        pos: [v.x, v.y, v.z],
                        color: ORANGE,
                    });
                }
            }
        }
        scene
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        chrome::title_bar(ctx, "Orbital");
        // Collect a restored mission from boot.
        if let Some(rx) = &self.mission_rx {
            if let Ok((sol, cfg)) = rx.try_recv() {
                self.mission_rx = None;
                self.sc_traj = sol.traj.clone();
                self.accepted = Some((cfg.mission, cfg.target));
                self.mission = Some((sol, cfg));
            }
        }
        // Collect a finished direct-transfer correction.
        if let Some(rx) = &self.direct_rx {
            if let Ok((s0, traj, miss)) = rx.try_recv() {
                self.direct_rx = None;
                self.sc_state = s0;
                self.sc_traj = traj;
                self.refined_miss_km = Some(miss);
            }
        }
        // Collect a finished multi-leg refinement.
        if let Some(rx) = &self.refine_rx {
            if let Ok(result) = rx.try_recv() {
                self.refine_rx = None;
                if let Some(rt) = result {
                    self.sc_traj = rt.traj.clone();
                    self.refined_miss_km = Some(rt.worst_miss_km);
                    if let Some((sol, _)) = &mut self.mission {
                        sol.flybys = rt.flybys.clone();
                        sol.vinf_dep_kms = rt.vinf_dep_kms;
                        sol.vinf_arr_kms = rt.vinf_arr_kms;
                        sol.assist_dv_kms = rt.assist_dv_kms;
                        sol.dsm_dv_kms = rt.dsm_dv_kms;
                        // The corrector is allowed to move patch epochs, so
                        // the schedule it flew — not the scouted one — is the
                        // mission.
                        sol.genome = rt.genome.clone();
                        sol.depart = rt.epochs[0];
                        sol.arrive = *rt.epochs.last().unwrap();
                    }
                    let mut info = format!(
                        "n-body tour: v∞ dep {:.2} arr {:.2} · assist Δv {:.2} km/s",
                        rt.vinf_dep_kms, rt.vinf_arr_kms, rt.assist_dv_kms
                    );
                    if rt.dsm_dv_kms > 0.001 {
                        info += &format!(" · DSM Δv {:.2} km/s", rt.dsm_dv_kms);
                    }
                    for f in &rt.flybys {
                        let (y, mo, d, ..) = f.epoch.to_gregorian_utc();
                        info += &format!(
                            "\n {} {y:04}-{mo:02}-{d:02} v∞ {:.1} alt {:.0} km",
                            f.body.name(),
                            f.vinf_kms,
                            f.periapsis_alt_km
                        );
                    }
                    self.refined_info = Some(info);
                } else {
                    self.refined_info = Some("tour refinement failed (kept conic path)".into());
                }
            }
        }
        if self.auto_start {
            self.auto_start = false;
            self.start_search(ctx);
        }
        if self.playing {
            // Scale by real frame time (referenced to 60 fps) so playback is
            // smooth and speed-consistent on any refresh rate. State remains
            // a pure function of the integer tick count.
            let dt = ctx.input(|i| i.stable_dt).clamp(0.001, 0.1) as f64;
            self.ticks += ((self.ticks_per_frame as f64) * dt * 60.0).round() as i64;
            ctx.request_repaint();
        }
        if let Some(sh) = &self.solver {
            if let Some(b) = sh.best.lock().unwrap().clone() {
                self.solver_best = Some(b);
            }
        }
        // Re-center trails when the epoch drifts, and keep retrying until the
        // background table exists (the build is a no-op before that).
        if self.trails.is_empty()
            || (self.epoch() - self.trails_epoch).abs() > Duration::from_days(40.0)
        {
            self.rebuild_trails();
        }

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(280.0)
            .width_range(240.0..=420.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.controls_ui(ui, ctx);
                        ui.add_space(12.0);
                    });
            });



        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

                if !self.free_pan {
                    self.camera.target = self.focus_pos();
                }
                let shift = ui.input(|i| i.modifiers.shift);
                if response.dragged_by(egui::PointerButton::Secondary)
                    || response.dragged_by(egui::PointerButton::Middle)
                    || (shift && response.dragged())
                {
                    // Pan in the view plane; scale with zoom so it feels 1:1.
                    let d = response.drag_delta();
                    let (sy, cy) = self.camera.yaw.sin_cos();
                    let (sp, cp) = self.camera.pitch.sin_cos();
                    let fwd = Vec3::new(-cp * cy, -cp * sy, -sp);
                    let right = fwd.cross(Vec3::Z).normalize();
                    let up = right.cross(fwd).normalize();
                    let k = self.camera.distance * 0.0016;
                    self.camera.target += (-right * d.x + up * d.y) * k;
                    self.free_pan = true;
                } else if response.dragged() {
                    let d = response.drag_delta();
                    self.camera.yaw -= d.x * 0.01;
                    self.camera.pitch = (self.camera.pitch + d.y * 0.01)
                        .clamp(-1.55, 1.55);
                }
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.distance =
                            (self.camera.distance * (1.0 - scroll * 0.002)).clamp(2e-7, 400.0);
                    }
                }

                let aspect = rect.width().max(1.0) / rect.height().max(1.0);
                let scene = self.build_scene(aspect);
                ui.painter().add(egui_wgpu_callback(rect, scene));
            });
    }
}

fn egui_wgpu_callback(rect: egui::Rect, scene: Scene) -> egui::Shape {
    egui::Shape::Callback(eframe::egui_wgpu::Callback::new_paint_callback(
        rect,
        SceneCallback { scene },
    ))
}
