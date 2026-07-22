//! Mission persistence: a plain key=value file — dependency-free and
//! diffable. The trajectory itself is not stored; it re-propagates
//! deterministically on load.
//!
//! # Format versioning
//!
//! The file opens with `version=<n>`. Every enum is written as a *stable
//! token* defined here, deliberately decoupled from the UI `label()` strings:
//! renaming a label is a presentation change and must never invalidate a
//! saved mission.
//!
//! Version 1 (implicit — files with no `version=` line) used `label()` output
//! as the mission-type key. Those files still load: `from_token` falls back to
//! matching `MissionType::label()` when the token is unknown.

use crate::bodies::ALL_BODIES;
use crate::solver::{self, Engine, Launcher, MissionType, SolverConfig};
use hifitime::Epoch;

pub const MISSION_FILE: &str = "mission.orbital";

/// Current on-disk format version.
pub const FORMAT_VERSION: u32 = 2;

/// Stable serialization token for a mission type — independent of `label()`.
fn mission_token(m: MissionType) -> &'static str {
    match m {
        MissionType::Flyby => "flyby",
        MissionType::Orbit => "orbit",
        MissionType::Land => "land",
    }
}

fn mission_from_token(tok: &str) -> Option<MissionType> {
    MissionType::ALL
        .iter()
        .find(|m| mission_token(**m) == tok)
        // v1 back-compat: the key used to be the UI label.
        .or_else(|| MissionType::ALL.iter().find(|m| m.label() == tok))
        .copied()
}

fn engine_token(e: Engine) -> &'static str {
    match e {
        Engine::Ballistic => "ballistic",
        Engine::Nstar => "nstar",
        Engine::NextC => "next",
        Engine::Aeps => "aeps",
    }
}

fn engine_from_token(tok: &str) -> Engine {
    match tok {
        "nstar" => Engine::Nstar,
        "next" => Engine::NextC,
        "aeps" => Engine::Aeps,
        _ => Engine::Ballistic,
    }
}

fn launcher_token(l: Launcher) -> &'static str {
    match l {
        Launcher::FalconHeavy => "fh",
        Launcher::Sls => "sls",
        Launcher::KickStage => "kick",
    }
}

fn launcher_from_token(tok: &str) -> Launcher {
    match tok {
        "sls" => Launcher::Sls,
        "kick" => Launcher::KickStage,
        _ => Launcher::FalconHeavy,
    }
}

/// Serialize the accepted mission (config + genome + departure).
pub fn serialize(sol: &solver::Solution, cfg: &SolverConfig) -> String {
    let join = |v: &[f64]| {
        v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(",")
    };
    let mut s = String::new();
    s += &format!("version={FORMAT_VERSION}\n");
    s += &format!("target={}\n", cfg.target.name());
    s += &format!("mission={}\n", mission_token(cfg.mission));
    s += &format!("engine={}\n", engine_token(cfg.engine));
    s += &format!("launcher={}\n", launcher_token(cfg.launcher));
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
    s
}

/// Persist the accepted mission next to the binary's working directory.
pub fn save_mission(sol: &solver::Solution, cfg: &SolverConfig) {
    if let Err(e) = std::fs::write(MISSION_FILE, serialize(sol, cfg)) {
        eprintln!("could not save {MISSION_FILE}: {e}");
    }
}

/// Parse a mission file body. Returns (config, genome, departure epoch).
pub fn deserialize(text: &str) -> Option<(SolverConfig, solver::Genome, Epoch)> {
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
            // Unknown/absent version means v1; every v1 key is still read
            // below, so nothing further is needed here.
            "version" => {}
            "target" => cfg.target = body(v)?,
            "mission" => cfg.mission = mission_from_token(v)?,
            "engine" => cfg.engine = engine_from_token(v),
            "launcher" => cfg.launcher = launcher_from_token(v),
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

/// Load the saved mission, if any.
pub fn load_mission() -> Option<(SolverConfig, solver::Genome, Epoch)> {
    deserialize(&std::fs::read_to_string(MISSION_FILE).ok()?)
}
