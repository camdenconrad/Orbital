//! Command-line entry points: `orbital search …` and argv parsing shared
//! with `orbital tour`.

use crate::bodies::ALL_BODIES;
use crate::ephemeris::Ephemeris;
use crate::solver::{self, Engine, MissionType, SolverConfig};
use hifitime::Epoch;

/// Parse target/route/mission/date from argv (shared by `search` and `tour`).
pub fn parse_search_args() -> (SolverConfig, Option<Epoch>) {
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
pub fn headless_search(target: Option<&str>, start: Option<&str>) {
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
