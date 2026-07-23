//! Trajectory solver: beam search over ballistic transfers.
//!
//! Architecture borrowed from Camden's 3DSolver (genome + mutation beam
//! search, pluggable objective, live-publishing best-so-far), adapted to
//! trajectories with one deliberate change: 3DSolver races several workers
//! that cross-pollinate through a shared best, which makes results depend on
//! thread timing. Orbital keeps the besom determinism rule, so the search
//! runs on ONE seeded worker — same seed + config, same sequence of bests,
//! every run.
//!
//! A candidate is a departure epoch, a time of flight, and a departure
//! v-infinity vector. Fresh candidates are **Lambert-seeded** — v∞ solves the
//! two-body Earth→target boundary-value problem for that (depart, TOF) — so
//! every launch window in range competes globally; mutation then refines
//! under the real dynamics. Decode = inject at Earth's SOI along v∞ with
//! velocity v_Earth + v∞, propagate the full 1PN n-body dynamics at
//! solver-grade fidelity, score at arrival. Solving uses a reduced perturber
//! set (Sun + planets + Moon) for speed; the UI re-propagates the winner at
//! full fidelity when you accept it.

use crate::bodies::{BodyId, DAY_S};
use crate::dynamics::{self, DynamicsConfig, ScState};
use crate::ephemeris::Ephemeris;
use hifitime::{Duration, Epoch};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------- rng

/// xorshift64* — seeded, dependency-free, deterministic (same as 3DSolver).
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------- problem

/// What "best transfer" means.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Cheapest ride: departure v∞ + arrival v∞.
    TotalDv,
    /// Get there fast, Δv-weighted lightly.
    TimeOfFlight,
    /// Gentlest arrival (capture/orbit insertion cost).
    ArrivalVinf,
}

impl Objective {
    pub const ALL: [Objective; 3] = [
        Objective::TotalDv,
        Objective::TimeOfFlight,
        Objective::ArrivalVinf,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Objective::TotalDv => "min total Δv",
            Objective::TimeOfFlight => "min time of flight",
            Objective::ArrivalVinf => "min arrival v∞",
        }
    }

    /// km/s-equivalent cost, before the miss penalty. `arr_cost` is the
    /// mission-type arrival Δv (0 for flybys), not the raw arrival v∞.
    fn score(self, vinf_dep: f64, arr_cost: f64, tof_days: f64) -> f64 {
        match self {
            Objective::TotalDv => vinf_dep + arr_cost,
            // 100 days ≈ 1 km/s of willingness to pay.
            Objective::TimeOfFlight => tof_days / 100.0 + 0.2 * (vinf_dep + arr_cost),
            Objective::ArrivalVinf => arr_cost + 0.15 * vinf_dep,
        }
    }
}

/// What happens at the target — determines the arrival Δv the score charges.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MissionType {
    /// Fly past: arrival v∞ is free (reported, not charged).
    Flyby,
    /// Capture burn at 1.5 body radii into a high ellipse (e = 0.95).
    Orbit,
    /// Come to rest on the surface: propulsive for airless bodies, ~free
    /// entry for bodies with a usable atmosphere. Not offered for gas giants.
    Land,
}

impl MissionType {
    pub const ALL: [MissionType; 3] = [MissionType::Flyby, MissionType::Orbit, MissionType::Land];

    pub fn label(self) -> &'static str {
        match self {
            MissionType::Flyby => "flyby",
            MissionType::Orbit => "orbit",
            MissionType::Land => "land",
        }
    }
}

/// NASA electric-propulsion hardware actually available today, modeled as
/// (thrust per engine, Isp), flown as a pair on a 1500 kg-wet probe with a
/// 45% propellant fraction. Accel and total Δv follow from the rocket
/// equation — no vendor optimism.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Chemical/ballistic: impulses only, no cruise thrust.
    Ballistic,
    /// NSTAR ion (Deep Space 1, Dawn): 92 mN, Isp 3100 s.
    Nstar,
    /// NEXT-C ion (DART): 236 mN, Isp 4190 s.
    NextC,
    /// AEPS Hall (Gateway PPE): 589 mN, Isp 2900 s.
    Aeps,
}

impl Engine {
    pub const ALL: [Engine; 4] = [Engine::Ballistic, Engine::Nstar, Engine::NextC, Engine::Aeps];

    pub fn label(self) -> &'static str {
        match self {
            Engine::Ballistic => "ballistic (chemical)",
            Engine::Nstar => "2× NSTAR ion (Dawn-class)",
            Engine::NextC => "2× NEXT-C ion (DART-class)",
            Engine::Aeps => "2× AEPS Hall (Gateway-class)",
        }
    }

    fn thrust_isp(self) -> Option<(f64, f64)> {
        match self {
            Engine::Ballistic => None,
            Engine::Nstar => Some((0.092, 3100.0)),
            Engine::NextC => Some((0.236, 4190.0)),
            Engine::Aeps => Some((0.589, 2900.0)),
        }
    }

    /// Max thrust acceleration, km/s² (2 engines, 1500 kg wet).
    pub fn accel_kms2(self) -> f64 {
        self.thrust_isp()
            .map(|(t, _)| 2.0 * t / 1500.0 / 1000.0)
            .unwrap_or(0.0)
    }

    /// Total Δv the propellant load buys, km/s (rocket equation, 45% prop).
    pub fn max_dv_kms(self) -> f64 {
        self.thrust_isp()
            .map(|(_, isp)| isp * 9.80665e-3 * (1.0f64 / 0.55).ln())
            .unwrap_or(0.0)
    }
}

/// Launch capability actually on the pad today — sets the honest C3 ceiling.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Launcher {
    /// Falcon Heavy expendable, ~3.5 t class probe.
    FalconHeavy,
    /// SLS Block 1 with ICPS, outer-planet class.
    Sls,
    /// Small probe on a heavy + kick stage (New Horizons style).
    KickStage,
}

impl Launcher {
    pub const ALL: [Launcher; 3] = [Launcher::FalconHeavy, Launcher::Sls, Launcher::KickStage];

    pub fn label(self) -> &'static str {
        match self {
            Launcher::FalconHeavy => "Falcon Heavy (C3 ≤ 60)",
            Launcher::Sls => "SLS Block 1 (C3 ≤ 93)",
            Launcher::KickStage => "heavy + kick stage (C3 ≤ 130)",
        }
    }

    pub fn c3_max(self) -> f64 {
        match self {
            Launcher::FalconHeavy => 60.0,
            Launcher::Sls => 93.0,
            Launcher::KickStage => 130.0,
        }
    }
}

/// Thrust-profile segments per low-thrust genome.
pub const N_SEG: usize = 12;

fn is_gas_giant(b: BodyId) -> bool {
    matches!(
        b,
        BodyId::Sun | BodyId::Jupiter | BodyId::Saturn | BodyId::Uranus | BodyId::Neptune
    )
}

fn has_usable_atmosphere(b: BodyId) -> bool {
    matches!(b, BodyId::Venus | BodyId::Earth | BodyId::Mars | BodyId::Titan)
}

/// Δv charged at the target for the chosen mission type, km/s.
pub fn arrival_dv_kms(body: BodyId, vinf_kms: f64, mission: MissionType) -> f64 {
    let mu = body.gm();
    match mission {
        MissionType::Flyby => 0.0,
        MissionType::Orbit => {
            let rp = 1.5 * body.radius_km();
            let vp_hyp = (vinf_kms * vinf_kms + 2.0 * mu / rp).sqrt();
            let vp_ell = (mu / rp * (1.0 + 0.95)).sqrt();
            vp_hyp - vp_ell
        }
        MissionType::Land => {
            if is_gas_giant(body) {
                return 1e3; // no surface to land on
            }
            if has_usable_atmosphere(body) {
                0.05 // entry, descent and landing are aerodynamic
            } else {
                // Propulsive soft landing from the arrival hyperbola.
                let r = body.radius_km();
                (vinf_kms * vinf_kms + 2.0 * mu / r).sqrt()
            }
        }
    }
}

#[derive(Clone)]
pub struct SolverConfig {
    pub target: BodyId,
    pub objective: Objective,
    pub mission: MissionType,
    pub engine: Engine,
    pub launcher: Launcher,
    /// Discover the flyby route automatically: screen candidate routes
    /// (including direct), refine the best few, polish the winner.
    pub auto_route: bool,
    /// Gravity-assist route: bodies flown past between Earth and the target
    /// (e.g. [Venus, Earth, Earth] for a VEEGA). Empty = direct ballistic.
    pub route: Vec<BodyId>,
    /// Departure window: [0, window_days] after `epoch0`.
    pub window_days: f64,
    pub tof_min_days: f64,
    pub tof_max_days: f64,
    pub seed: u64,
    pub beam_width: usize,
    pub mutations: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            target: BodyId::Mars,
            objective: Objective::TotalDv,
            mission: MissionType::Orbit,
            engine: Engine::Ballistic,
            launcher: Launcher::FalconHeavy,
            auto_route: true,
            route: Vec::new(),
            window_days: 900.0,
            tof_min_days: 60.0,
            tof_max_days: 600.0,
            seed: 7,
            beam_width: 12,
            mutations: 6,
        }
    }
}

/// Semi-major axis from a heliocentric body's period.
fn semi_major_km(b: BodyId) -> f64 {
    let mu = BodyId::Sun.gm();
    let t = b.period_days().abs() * DAY_S;
    (mu * (t / (2.0 * std::f64::consts::PI)).powi(2)).cbrt()
}

/// Hohmann half-ellipse time between two heliocentric bodies, days.
fn hohmann_days(b1: BodyId, b2: BodyId) -> f64 {
    let mu = BodyId::Sun.gm();
    let a = 0.5 * (semi_major_km(b1) + semi_major_km(b2));
    std::f64::consts::PI * (a * a * a / mu).sqrt() / DAY_S
}

impl SolverConfig {
    /// Scale the direct-transfer TOF bounds to the target's distance
    /// (~1.35× the Hohmann half-ellipse time). Without this, a Pluto search
    /// capped at Mars-sized flight times can only offer absurd hyperbolic
    /// sprints — which it will then dutifully "optimize".
    pub fn scale_tof_to_target(&mut self) {
        let h = hohmann_days(BodyId::Earth, self.target);
        self.tof_max_days = self.tof_max_days.max(1.35 * h);
        self.tof_min_days = self.tof_min_days.min(0.4 * h).max(30.0);
    }

    /// Full body sequence: Earth, flybys…, target.
    pub fn sequence(&self) -> Vec<BodyId> {
        let mut seq = vec![BodyId::Earth];
        seq.extend(&self.route);
        seq.push(self.target);
        seq
    }

    /// (min, max) TOF per leg, days. Same-body legs (resonant returns) get a
    /// wide fixed range; others scale from the pair's Hohmann time.
    pub fn leg_bounds(&self) -> Vec<(f64, f64)> {
        let seq = self.sequence();
        seq.windows(2)
            .map(|w| {
                if w[0] == w[1] {
                    (250.0, 1500.0)
                } else {
                    let h = hohmann_days(w[0], w[1]);
                    ((0.3 * h).max(30.0), 2.2 * h)
                }
            })
            .collect()
    }

    /// Worst-case total flight time — sizes the ephemeris table span.
    pub fn max_total_tof_days(&self) -> f64 {
        if self.route.is_empty() {
            self.tof_max_days
        } else {
            self.leg_bounds().iter().map(|(_, hi)| hi).sum()
        }
    }
}

/// A candidate transfer: when to leave and how long each leg flies. For the
/// direct (routeless) mode `legs` has one entry and `vinf_dep` is the search
/// variable; for tours the per-leg Lambert arcs determine every velocity and
/// `vinf_dep` is derived.
#[derive(Clone)]
pub struct Genome {
    pub depart_days: f64,
    pub legs: Vec<f64>,
    pub vinf_dep: [f64; 3],
    /// Low-thrust throttle per segment (velocity frame), empty = ballistic.
    pub thrust: Vec<[f64; 3]>,
    /// Per-leg deep-space maneuver, one entry per leg: `[frac, dvx, dvy, dvz]`.
    /// `frac` is where along the leg the burn happens (fraction of the leg's
    /// TOF); the vector is the **burn itself**, in km/s, applied to the
    /// spacecraft's heliocentric velocity *at the maneuver point*. The leg's
    /// departure velocity is then whatever makes that burn close onto the
    /// arrival node (see [`solve_leg`]), so the genome vector and the Δv the
    /// score charges are one and the same quantity — the standard MGA-1DSM
    /// parameterization, where the maneuver is the decision variable and the
    /// departure state is the dependent one.
    ///
    /// Because the departure velocity is solved by shooting from the
    /// ballistic Lambert arc as its seed, the leg tends to that same arc
    /// *continuously* as `dv → 0`, not merely at exactly zero. Empty =
    /// ballistic tour.
    pub dsm: Vec<[f64; 4]>,
    /// Per-leg Lambert branch selector, one entry per leg. Legs commonly have
    /// several arcs that fit the same (r1, r2, TOF) — different revolution
    /// counts, and the low/high branch of each multi-rev family. Index 0 is
    /// the arc with the cheapest departure v∞ (the historical greedy pick);
    /// higher indices are the alternatives, ordered deterministically by
    /// [`lambert_candidates`] and wrapped modulo the number available.
    ///
    /// Making this a gene is the point: a locally cheap departure can be a
    /// globally worse tour once the flyby turn limit, arrival v∞ and capture
    /// burn are counted, and only the full mission score can see that. Empty
    /// (or short) = branch 0 for every leg.
    pub branch: Vec<u32>,
}

impl Genome {
    pub fn total_tof_days(&self) -> f64 {
        self.legs.iter().sum()
    }

    /// The DSM parameters for leg `i`, or `None` if this leg flies ballistic.
    fn leg_dsm(&self, i: usize) -> Option<&[f64; 4]> {
        self.dsm
            .get(i)
            .filter(|d| d[1] != 0.0 || d[2] != 0.0 || d[3] != 0.0)
    }

    /// The Lambert branch selected for leg `i` (0 when unset).
    fn leg_branch(&self, i: usize) -> u32 {
        self.branch.get(i).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------- two-body

fn stumpff_c(z: f64) -> f64 {
    if z > 1e-8 {
        (1.0 - z.sqrt().cos()) / z
    } else if z < -1e-8 {
        ((-z).sqrt().cosh() - 1.0) / (-z)
    } else {
        0.5
    }
}

fn stumpff_s(z: f64) -> f64 {
    if z > 1e-8 {
        let sz = z.sqrt();
        (sz - sz.sin()) / (sz * sz * sz)
    } else if z < -1e-8 {
        let sz = (-z).sqrt();
        (sz.sinh() - sz) / (sz * sz * sz)
    } else {
        1.0 / 6.0
    }
}

/// Two-body propagation by universal variables (Vallado): state after `dt_s`
/// on the conic defined by (r0, v0). Used to sample tour legs for rendering.
pub fn kepler_universal(
    r0: [f64; 3],
    v0: [f64; 3],
    dt_s: f64,
    mu: f64,
) -> ([f64; 3], [f64; 3]) {
    if dt_s.abs() < 1e-9 {
        return (r0, v0);
    }
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let r0n = dot(r0, r0).sqrt();
    let v02 = dot(v0, v0);
    let rv = dot(r0, v0);
    let alpha = 2.0 / r0n - v02 / mu; // 1/a
    let smu = mu.sqrt();
    let mut chi = if alpha > 1e-12 {
        smu * dt_s * alpha
    } else {
        dt_s.signum() * (r0n * 0.1_f64).sqrt() * 10.0 // rough hyperbolic seed
    };
    for _ in 0..80 {
        let z = alpha * chi * chi;
        let c = stumpff_c(z);
        let s = stumpff_s(z);
        let t = (chi * chi * chi * s + rv / smu * chi * chi * c + r0n * chi * (1.0 - z * s))
            / smu;
        let dtdchi =
            (chi * chi * c + rv / smu * chi * (1.0 - z * s) + r0n * (1.0 - z * c)) / smu;
        let dchi = (dt_s - t) / dtdchi;
        chi += dchi.clamp(-chi.abs().max(1.0), chi.abs().max(1.0));
        if dchi.abs() < 1e-8 * chi.abs().max(1.0) {
            break;
        }
    }
    let z = alpha * chi * chi;
    let c = stumpff_c(z);
    let s = stumpff_s(z);
    let f = 1.0 - chi * chi * c / r0n;
    let g = dt_s - chi * chi * chi * s / smu;
    let r = [
        f * r0[0] + g * v0[0],
        f * r0[1] + g * v0[1],
        f * r0[2] + g * v0[2],
    ];
    let rn = dot(r, r).sqrt();
    let fdot = smu / (r0n * rn) * chi * (z * s - 1.0);
    let gdot = 1.0 - chi * chi * c / rn;
    let v = [
        fdot * r0[0] + gdot * v0[0],
        fdot * r0[1] + gdot * v0[1],
        fdot * r0[2] + gdot * v0[2],
    ];
    (r, v)
}

// ---------------------------------------------------------------- lambert

/// Single-revolution prograde Lambert (see `lambert_rev` for multi-rev).
#[allow(dead_code)] // test-facing convenience wrapper over lambert_rev
pub fn lambert(
    r1: [f64; 3],
    r2: [f64; 3],
    tof_s: f64,
    mu: f64,
) -> Option<([f64; 3], [f64; 3])> {
    lambert_rev(r1, r2, tof_s, mu, 0, false)
}

/// Universal-variables Lambert solver (Bate–Mueller–White / Curtis form):
/// given two heliocentric positions and a time of flight, return the
/// departure and arrival velocities of the connecting conic. Prograde.
///
/// `revs` = complete revolutions of the transfer (0 = classic single-rev).
/// For revs ≥ 1 the TOF curve over z is U-shaped and two conics exist;
/// `high_branch` picks the larger-z (higher-energy…lower-eccentricity) one.
/// Bisection on z — slow-and-steady but unconditionally convergent, which
/// matters more here than iteration count.
pub fn lambert_rev(
    r1: [f64; 3],
    r2: [f64; 3],
    tof_s: f64,
    mu: f64,
    revs: u32,
    high_branch: bool,
) -> Option<([f64; 3], [f64; 3])> {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let r1n = dot(r1, r1).sqrt();
    let r2n = dot(r2, r2).sqrt();
    if r1n < 1.0 || r2n < 1.0 || tof_s <= 0.0 {
        return None;
    }
    let cosd = (dot(r1, r2) / (r1n * r2n)).clamp(-1.0, 1.0);
    let mut dtheta = cosd.acos();
    // Prograde: transfer direction follows the +z angular momentum sense.
    let cross_z = r1[0] * r2[1] - r1[1] * r2[0];
    if cross_z < 0.0 {
        dtheta = 2.0 * std::f64::consts::PI - dtheta;
    }
    let a_coef = dtheta.sin() * (r1n * r2n / (1.0 - cosd)).sqrt();
    if !a_coef.is_finite() || a_coef.abs() < 1e-8 {
        return None; // colinear geometry
    }

    let y = |z: f64| {
        let c = stumpff_c(z);
        r1n + r2n + a_coef * (z * stumpff_s(z) - 1.0) / c.sqrt()
    };
    let tof_of = |z: f64| -> f64 {
        let yv = y(z);
        if yv < 0.0 {
            return f64::NEG_INFINITY; // below the parabolic boundary
        }
        let c = stumpff_c(z);
        (yv / c).powf(1.5) * stumpff_s(z) + a_coef * yv.sqrt()
    };
    let target = mu.sqrt() * tof_s;
    let pi = std::f64::consts::PI;

    let z = if revs == 0 {
        // Single rev: tof_of is monotonic increasing over (z_lo, (2π)²).
        let mut z_lo = -16.0 * pi * pi;
        let mut z_hi = 4.0 * pi * pi - 1e-6;
        // For A > 0 the y > 0 validity region starts at some z boundary;
        // bisect *to the boundary* (never past the root, which an eager
        // midpoint walk can overshoot).
        if !tof_of(z_lo).is_finite() {
            let mut bad = z_lo;
            let mut good = z_hi;
            for _ in 0..128 {
                let mid = 0.5 * (bad + good);
                if tof_of(mid).is_finite() {
                    good = mid;
                } else {
                    bad = mid;
                }
            }
            z_lo = good;
        }
        if !(tof_of(z_lo) <= target && target <= tof_of(z_hi)) {
            return None; // requested TOF unreachable single-rev
        }
        let mut z = 0.0;
        for _ in 0..120 {
            z = 0.5 * (z_lo + z_hi);
            if tof_of(z) < target {
                z_lo = z;
            } else {
                z_hi = z;
            }
        }
        z
    } else {
        // Multi-rev: over z ∈ ((2mπ)², (2(m+1)π)²) the TOF curve is U-shaped
        // (asymptotes at both ends). Locate the minimum by ternary search,
        // then bisect the requested branch.
        let m = revs as f64;
        let lo = (2.0 * m * pi).powi(2) + 1e-4;
        let hi = (2.0 * (m + 1.0) * pi).powi(2) - 1e-4;
        let (mut a, mut b) = (lo, hi);
        for _ in 0..200 {
            let m1 = a + (b - a) / 3.0;
            let m2 = b - (b - a) / 3.0;
            if tof_of(m1) < tof_of(m2) {
                b = m2;
            } else {
                a = m1;
            }
        }
        let z_min = 0.5 * (a + b);
        if tof_of(z_min) > target {
            return None; // TOF below the m-rev minimum
        }
        let (mut z_lo, mut z_hi, increasing) = if high_branch {
            (z_min, hi, true)
        } else {
            (lo, z_min, false)
        };
        let mut z = z_min;
        for _ in 0..120 {
            z = 0.5 * (z_lo + z_hi);
            let below = tof_of(z) < target;
            if below == increasing {
                z_lo = z;
            } else {
                z_hi = z;
            }
        }
        z
    };
    let yv = y(z);
    if yv <= 0.0 {
        return None;
    }
    let f = 1.0 - yv / r1n;
    let g = a_coef * (yv / mu).sqrt();
    let gdot = 1.0 - yv / r2n;
    if g.abs() < 1e-9 {
        return None;
    }
    let v1 = [
        (r2[0] - f * r1[0]) / g,
        (r2[1] - f * r1[1]) / g,
        (r2[2] - f * r1[2]) / g,
    ];
    let v2 = [
        (gdot * r2[0] - r1[0]) / g,
        (gdot * r2[1] - r1[1]) / g,
        (gdot * r2[2] - r1[2]) / g,
    ];
    Some((v1, v2))
}

/// Hard cap on the revolution sweep in [`lambert_best`]. The geometric bound
/// `floor(tof / T_min)` is the physically correct limit, but for a very long
/// TOF against a very small transfer ellipse it can run to hundreds — far past
/// anything a mission would fly, and each extra `m` costs two bisections.
const MAX_LAMBERT_REVS: u32 = 20;

/// Largest revolution count swept for this geometry in `tof_s`.
///
/// The m-rev branch only exists once the TOF exceeds the m-rev minimum, which
/// is itself strictly greater than `m` periods of the minimum-energy transfer
/// ellipse (`a_min = s/2`, `s` the triangle semiperimeter), so
/// `floor(tof / T_min)` is a tight geometric upper bound. The value returned
/// is that bound *clamped to [`MAX_LAMBERT_REVS`]*: for a long TOF against a
/// small transfer ellipse the geometric bound alone runs to hundreds, so the
/// sweep is deliberately truncated and is not exhaustive in that regime.
fn max_revs(r1: [f64; 3], r2: [f64; 3], tof_s: f64, mu: f64) -> u32 {
    let n = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (r1n, r2n) = (n(r1), n(r2));
    let c = n([r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]]);
    let a_min = 0.25 * (r1n + r2n + c); // = s/2
    if !(a_min > 0.0) || mu <= 0.0 || !tof_s.is_finite() {
        return 0;
    }
    let t_min = 2.0 * std::f64::consts::PI * (a_min * a_min * a_min / mu).sqrt();
    if !(t_min > 0.0) {
        return 0;
    }
    ((tof_s / t_min).floor().max(0.0) as u32).min(MAX_LAMBERT_REVS)
}

/// Every Lambert conic that fits this leg: the single-rev arc plus both
/// branches of every revolution count the geometry admits, up to
/// [`MAX_LAMBERT_REVS`] (see [`max_revs`]).
///
/// Ordered by departure cost — `|v1 - vref|`, i.e. cheapest v∞ off the start
/// node — so index 0 is the locally best arc and the rest are the genuine
/// alternatives a caller may want to score against the *whole* mission
/// instead of committing greedily. Deterministic: fixed sweep order, and the
/// sort is stable so ties keep first-seen order.
pub fn lambert_candidates(
    r1: [f64; 3],
    r2: [f64; 3],
    tof_s: f64,
    mu: f64,
    vref: [f64; 3],
) -> Vec<([f64; 3], [f64; 3])> {
    let cost = |o: &([f64; 3], [f64; 3])| {
        let d = [o.0[0] - vref[0], o.0[1] - vref[1], o.0[2] - vref[2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    let mut out: Vec<([f64; 3], [f64; 3])> = Vec::new();
    let consider = |out: &mut Vec<_>, o: Option<([f64; 3], [f64; 3])>| {
        if let Some(v) = o {
            if v.0.iter().chain(v.1.iter()).all(|c: &f64| c.is_finite()) {
                out.push(v);
            }
        }
    };
    consider(&mut out, lambert_rev(r1, r2, tof_s, mu, 0, false));
    for m in 1..=max_revs(r1, r2, tof_s, mu) {
        consider(&mut out, lambert_rev(r1, r2, tof_s, mu, m, false));
        consider(&mut out, lambert_rev(r1, r2, tof_s, mu, m, true));
    }
    out.sort_by(|a, b| cost(a).total_cmp(&cost(b)));
    out
}

/// The `k`-th cheapest Lambert arc for a leg, wrapping modulo the number
/// available so any branch gene value is valid. `k = 0` is [`lambert_best`].
fn lambert_branch(
    r1: [f64; 3],
    r2: [f64; 3],
    tof_s: f64,
    mu: f64,
    vref: [f64; 3],
    k: u32,
) -> Option<([f64; 3], [f64; 3])> {
    let c = lambert_candidates(r1, r2, tof_s, mu, vref);
    if c.is_empty() {
        return None;
    }
    Some(c[k as usize % c.len()])
}

/// The arc leaving the start node with the velocity closest to `vref` (the
/// departure body's own velocity — i.e. the cheapest v∞).
///
/// This is a *local* criterion: it sees neither the incoming flyby velocity,
/// nor the arrival v∞, nor the next leg. Callers that care about the tour as
/// a whole should use [`lambert_candidates`] and score the alternatives —
/// which is what the per-leg branch gene ([`Genome::branch`]) exists for.
pub fn lambert_best(
    r1: [f64; 3],
    r2: [f64; 3],
    tof_s: f64,
    mu: f64,
    vref: [f64; 3],
) -> Option<([f64; 3], [f64; 3])> {
    lambert_candidates(r1, r2, tof_s, mu, vref).into_iter().next()
}

/// Where along a leg a DSM may sit. Burning in the first/last few percent of
/// a leg is indistinguishable from retargeting the node itself and makes the
/// second Lambert arc numerically degenerate.
const DSM_FRAC_RANGE: (f64, f64) = (0.05, 0.95);

/// One solved tour leg, ballistic or with a deep-space maneuver.
#[derive(Clone)]
struct LegArc {
    /// Heliocentric velocity leaving the start node.
    v_start: [f64; 3],
    /// Heliocentric velocity arriving at the end node.
    v_end: [f64; 3],
    /// Δv the mid-leg maneuver actually costs, km/s (0 when ballistic).
    dsm_dv: f64,
    /// `(seconds after the start node, position, velocity *after* the burn)`.
    dsm: Option<(f64, [f64; 3], [f64; 3])>,
}

/// How closely the DSM shooter must close onto the arrival node, km. Six
/// orders of magnitude below an AU: far tighter than the patched-conic model
/// is meaningful to, so the shot is exact as far as the score can tell.
const DSM_SHOOT_TOL_KM: f64 = 1.0e-3;

/// Solve for the leg-departure velocity that makes a *given* mid-leg burn
/// close onto the arrival node.
///
/// This is the inversion that makes the genome's vector the real maneuver:
/// coast `t1` from `r1`, add `dv` to the velocity there, coast the remaining
/// `t2`, and require the endpoint to be `r2`. Three equations, three unknowns
/// (the departure velocity), seeded with the ballistic Lambert arc `v_seed`.
///
/// At `dv = 0` the seed is already the exact solution, and the residual is
/// smooth in `dv`, so the converged departure velocity — and hence every
/// state on the leg — varies *continuously* through zero. Nothing here picks
/// a revolution count or a branch, so nothing can flip discontinuously under
/// an arbitrarily small kick.
///
/// Returns `(v_dep, rm, v_after_burn)`, or `None` if the shot did not close.
fn shoot_dsm_leg(
    r1: [f64; 3],
    r2: [f64; 3],
    t1: f64,
    t2: f64,
    mu: f64,
    dv: [f64; 3],
    v_seed: [f64; 3],
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    // Fly the leg for a trial departure velocity: (miss vector, rm, v_after).
    let fly = |v: [f64; 3]| -> Option<([f64; 3], [f64; 3], [f64; 3])> {
        let (rm, vm) = kepler_universal(r1, v, t1, mu);
        if !rm.iter().chain(vm.iter()).all(|c| c.is_finite()) {
            return None;
        }
        let v_after = [vm[0] + dv[0], vm[1] + dv[1], vm[2] + dv[2]];
        let (rf, _) = kepler_universal(rm, v_after, t2, mu);
        if !rf.iter().all(|c| c.is_finite()) {
            return None;
        }
        Some(([rf[0] - r2[0], rf[1] - r2[1], rf[2] - r2[2]], rm, v_after))
    };

    let mut v = v_seed;
    let (mut f, mut rm, mut v_after) = fly(v)?;
    let (mut best, mut best_norm, mut best_rm, mut best_va) = (v, norm(f), rm, v_after);
    for _ in 0..20 {
        if best_norm < DSM_SHOOT_TOL_KM {
            break;
        }
        // Relative finite-difference step: the conics are analytic, so the
        // only floor is round-off, and scaling with |v| keeps the Jacobian
        // conditioned across the inner-planet/outer-planet velocity range.
        let h = 1e-7 * norm(v).max(1.0);
        let mut jac = [[0.0f64; 3]; 3];
        for c in 0..3 {
            let mut vp = v;
            vp[c] += h;
            let (fp, _, _) = fly(vp)?;
            for r in 0..3 {
                jac[r][c] = (fp[r] - f[r]) / h;
            }
        }
        let det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
            - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
            + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
        if det == 0.0 || !det.is_finite() {
            break;
        }
        let rhs = [-f[0], -f[1], -f[2]];
        let solve_col = |col: usize| {
            let mut m = jac;
            for r in 0..3 {
                m[r][col] = rhs[r];
            }
            (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                / det
        };
        let step = [solve_col(0), solve_col(1), solve_col(2)];
        if !step.iter().all(|c| c.is_finite()) {
            break;
        }
        // Backtrack until the step actually improves the miss.
        let mut scale = 1.0;
        let mut stepped = false;
        for _ in 0..8 {
            let trial = [v[0] + step[0] * scale, v[1] + step[1] * scale, v[2] + step[2] * scale];
            if let Some((ft, rmt, vat)) = fly(trial) {
                if norm(ft) < norm(f) {
                    v = trial;
                    f = ft;
                    rm = rmt;
                    v_after = vat;
                    stepped = true;
                    break;
                }
            }
            scale *= 0.5;
        }
        if !stepped {
            break;
        }
        if norm(f) < best_norm {
            best = v;
            best_norm = norm(f);
            best_rm = rm;
            best_va = v_after;
        }
    }
    // A leg that never closed is not a leg: reporting its unconverged
    // endpoint would credit the tour with an arrival it does not make.
    if best_norm > 1.0 {
        return None;
    }
    Some((best, best_rm, best_va))
}

/// Solve one leg from `r1` to `r2` in `tof_s`.
///
/// Ballistic (`dsm = None`): one Lambert arc, the `branch`-th cheapest.
///
/// With a DSM, the genome *is* the maneuver: burn `d[1..4]` at the point
/// `d[0]` of the way through the leg, and let the departure velocity be
/// whatever closes that trajectory onto the arrival node (see
/// [`shoot_dsm_leg`]). The Δv charged is exactly the genome vector's
/// magnitude — the burn the spacecraft actually performs — and the leg
/// approaches the ballistic arc continuously as the burn goes to zero.
fn solve_leg(
    r1: [f64; 3],
    r2: [f64; 3],
    tof_s: f64,
    mu: f64,
    vref: [f64; 3],
    dsm: Option<&[f64; 4]>,
    branch: u32,
) -> Option<LegArc> {
    let (v1, v2) = lambert_branch(r1, r2, tof_s, mu, vref, branch)?;
    let Some(d) = dsm else {
        return Some(LegArc { v_start: v1, v_end: v2, dsm_dv: 0.0, dsm: None });
    };
    let frac = d[0].clamp(DSM_FRAC_RANGE.0, DSM_FRAC_RANGE.1);
    let t1 = frac * tof_s;
    let dv = [d[1], d[2], d[3]];
    let (v_start, rm, v_after) = shoot_dsm_leg(r1, r2, t1, tof_s - t1, mu, dv, v1)?;
    // The arrival velocity is the one the shot actually delivers.
    let (_, v_end) = kepler_universal(rm, v_after, tof_s - t1, mu);
    if !v_end.iter().all(|c| c.is_finite()) {
        return None;
    }
    Some(LegArc {
        v_start,
        v_end,
        dsm_dv: (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt(),
        dsm: Some((t1, rm, v_after)),
    })
}

/// One gravity assist in a tour solution.
#[derive(Clone, Copy)]
pub struct Flyby {
    pub body: BodyId,
    pub epoch: Epoch,
    pub vinf_kms: f64,
    /// Powered-flyby Δv charged for the |v∞ in|−|v∞ out| mismatch plus any
    /// turn beyond the body's bending capability, km/s.
    pub dv_kms: f64,
    /// Periapsis altitude implied by the required turn, km (0 if the turn
    /// needed more bending than the body can give).
    pub periapsis_alt_km: f64,
}

/// Everything the UI wants to show about the current best.
#[derive(Clone)]
pub struct Solution {
    pub genome: Genome,
    pub score: f64,
    pub vinf_dep_kms: f64,
    pub vinf_arr_kms: f64,
    /// Δv charged at the target for the mission type (capture/landing burn).
    pub arrival_dv_kms: f64,
    /// Δv expended by the electric-propulsion profile en route, km/s.
    pub thrust_dv_kms: f64,
    /// Total powered-flyby Δv across the tour, km/s (0 for direct).
    pub assist_dv_kms: f64,
    /// Total deep-space-maneuver Δv across the tour, km/s (0 for direct).
    pub dsm_dv_kms: f64,
    pub flybys: Vec<Flyby>,
    pub miss_km: f64,
    pub depart: Epoch,
    pub arrive: Epoch,
    pub traj: Vec<(Epoch, ScState)>,
}

pub struct Shared {
    pub best: Mutex<Option<Solution>>,
    pub status: Mutex<String>,
    pub steps: AtomicU64,
    pub evals: AtomicU64,
    pub running: AtomicBool,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            best: Mutex::new(None),
            status: Mutex::new(String::new()),
            steps: AtomicU64::new(0),
            evals: AtomicU64::new(0),
            running: AtomicBool::new(true),
        })
    }
}

/// Solver-grade dynamics: every cataloged body's gravity active (same force
/// model as the final propagation), with a looser tolerance and a step
/// budget for search throughput.
pub fn solver_dynamics() -> DynamicsConfig {
    DynamicsConfig {
        rel_tol: 1e-8,
        max_steps: 30_000,
        ..Default::default()
    }
}

/// Sphere-of-influence radius, km — inside it a miss counts as a hit.
fn soi_km(eph: &Ephemeris, body: BodyId, epoch: Epoch) -> f64 {
    let s = eph.state(body, epoch);
    let r = (s.pos_km[0].powi(2) + s.pos_km[1].powi(2) + s.pos_km[2].powi(2)).sqrt();
    r * (body.gm() / BodyId::Sun.gm()).powf(0.4)
}

pub fn evaluate(
    eph: &Ephemeris,
    dyn_cfg: &DynamicsConfig,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
    n_samples: usize,
) -> Solution {
    if !cfg.route.is_empty() {
        // Tours stay patched-conic; engines apply to direct transfers only.
        evaluate_tour(eph, cfg, epoch0, g, n_samples)
    } else if cfg.engine != Engine::Ballistic && !g.thrust.is_empty() {
        evaluate_lowthrust(eph, dyn_cfg, cfg, epoch0, g, n_samples)
    } else {
        evaluate_direct(eph, dyn_cfg, cfg, epoch0, g, n_samples)
    }
}

/// Score offset marking a candidate the selected hardware cannot fly at all.
///
/// Every feasible score is a sum of km/s-scale terms and miss penalties that
/// stay far below this, so adding it is a strict lexicographic demotion: no
/// infeasible candidate can ever outrank a feasible one, however attractive
/// its objective.
const INFEASIBLE_SCORE: f64 = 1.0e9;

/// Shared mission scoring: the objective plus the hardware-feasibility terms
/// every evaluator must charge. `extra` carries evaluator-specific terms
/// (miss, propellant, flyby Δv) already summed by the caller.
///
/// **Hardware capability is a hard bound, not a purchasable penalty.** The
/// launch vehicle either has the performance to inject the stack at
/// `vinf_dep` or it does not, and the tank either holds the propellant the
/// profile burns or it does not; there is no exchange rate at which a better
/// arrival buys the missing launch energy or the missing propellant. So
/// exceeding either demotes the candidate below every feasible one outright,
/// rather than adding a weighted soft term it could outbid.
///
/// The old C3 form added `5 * max(v∞² - C3max, 0)`: a km²/s² quantity summed
/// against km/s Δv under an underived weight. What remains past the bound is
/// the shortfall expressed in the same unit as everything else it sits beside
/// — km/s — purely so the search still sees a downhill gradient back into the
/// feasible region. `hw_over_kms` carries any *other* hardware shortfall the
/// evaluator knows about (the propellant overrun, for the SEP profile),
/// already in km/s.
fn mission_score(
    cfg: &SolverConfig,
    vinf_dep: f64,
    arrival_dv: f64,
    tof_days: f64,
    extra: f64,
    hw_over_kms: f64,
) -> f64 {
    let base = cfg.objective.score(vinf_dep, arrival_dv, tof_days) + extra;
    let vinf_max = cfg.launcher.c3_max().sqrt();
    let over = (vinf_dep - vinf_max).max(0.0) + hw_over_kms.max(0.0);
    if over > 0.0 {
        INFEASIBLE_SCORE + base + over
    } else {
        base
    }
}

/// Score for a candidate whose propagation ran out of integrator steps. The
/// flight never reached its arrival epoch, so nothing about its "arrival" is
/// meaningful: it is infeasible, ranked below every complete candidate and
/// graded by how far short it fell so the search still sees a gradient.
fn truncated_score(fraction: f64) -> f64 {
    1e6 + (1.0 - fraction.clamp(0.0, 1.0)) * 1e6
}

/// Solution for a truncated (step-exhausted) propagation: infeasible, with no
/// arrival claim. `arrive` is the epoch actually reached, not the requested one.
fn infeasible(
    g: &Genome,
    fraction: f64,
    vinf_dep: f64,
    depart: Epoch,
    arrive: Epoch,
    traj: Vec<(Epoch, ScState)>,
) -> Solution {
    Solution {
        genome: g.clone(),
        score: truncated_score(fraction),
        vinf_dep_kms: vinf_dep,
        vinf_arr_kms: f64::INFINITY,
        arrival_dv_kms: f64::INFINITY,
        thrust_dv_kms: 0.0,
        assist_dv_kms: 0.0,
        dsm_dv_kms: 0.0,
        flybys: Vec::new(),
        miss_km: f64::INFINITY,
        depart,
        arrive,
        traj,
    }
}

/// Low-thrust transfer: same injection as ballistic, then the engine's
/// throttle profile shapes the cruise. Score charges the launcher-capped
/// departure v∞, the propellant actually burned, the mission-typed arrival
/// cost from the *post-braking* relative speed, and the usual miss terms.
fn evaluate_lowthrust(
    eph: &Ephemeris,
    dyn_cfg: &DynamicsConfig,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
    n_samples: usize,
) -> Solution {
    let depart = epoch0 + Duration::from_seconds(g.depart_days * DAY_S);
    let earth = eph.state(BodyId::Earth, depart);
    const EARTH_SOI_KM: f64 = 925_000.0;
    let vmag =
        (g.vinf_dep[0].powi(2) + g.vinf_dep[1].powi(2) + g.vinf_dep[2].powi(2)).sqrt();
    let dir = if vmag > 0.05 {
        [
            g.vinf_dep[0] / vmag,
            g.vinf_dep[1] / vmag,
            g.vinf_dep[2] / vmag,
        ]
    } else {
        let ev = earth.vel_km_s;
        let en = (ev[0].powi(2) + ev[1].powi(2) + ev[2].powi(2)).sqrt();
        [ev[0] / en, ev[1] / en, ev[2] / en]
    };
    let s0 = ScState {
        pos: [
            earth.pos_km[0] + dir[0] * EARTH_SOI_KM,
            earth.pos_km[1] + dir[1] * EARTH_SOI_KM,
            earth.pos_km[2] + dir[2] * EARTH_SOI_KM,
        ],
        vel: [
            earth.vel_km_s[0] + g.vinf_dep[0],
            earth.vel_km_s[1] + g.vinf_dep[1],
            earth.vel_km_s[2] + g.vinf_dep[2],
        ],
    };
    let total_s = g.legs[0] * DAY_S;
    let thrust = dynamics::Thrust {
        segs: &g.thrust,
        accel_kms2: cfg.engine.accel_kms2(),
        total_s,
    };
    let prop = dynamics::propagate_checked(
        eph,
        dyn_cfg,
        depart,
        s0,
        Duration::from_seconds(total_s),
        n_samples,
        Some(&thrust),
    );
    let traj = prop.points;
    let (arrive, sf) = *traj.last().unwrap();
    if !prop.complete {
        return infeasible(g, prop.fraction, vmag, depart, arrive, traj);
    }
    let tgt = eph.state(cfg.target, arrive);
    let dr = [
        sf.pos[0] - tgt.pos_km[0],
        sf.pos[1] - tgt.pos_km[1],
        sf.pos[2] - tgt.pos_km[2],
    ];
    let miss_km = (dr[0].powi(2) + dr[1].powi(2) + dr[2].powi(2)).sqrt();
    let dv = [
        sf.vel[0] - tgt.vel_km_s[0],
        sf.vel[1] - tgt.vel_km_s[1],
        sf.vel[2] - tgt.vel_km_s[2],
    ];
    let v_rel = (dv[0].powi(2) + dv[1].powi(2) + dv[2].powi(2)).sqrt();
    let vinf_dep = vmag;

    // Propellant actually burned.
    let seg_dt = total_s / g.thrust.len().max(1) as f64;
    let thrust_dv: f64 = g
        .thrust
        .iter()
        .map(|u| {
            let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().min(1.0);
            un * cfg.engine.accel_kms2() * seg_dt
        })
        .sum();

    let arrival_dv = arrival_dv_kms(cfg.target, v_rel, cfg.mission);
    let excess_miss = (miss_km - soi_km(eph, cfg.target, arrive)).max(0.0);
    // Propellant past the tank is the same kind of statement as launch energy
    // past the rocket: not an expensive option, an unavailable one. Both are
    // handled as hard bounds in `mission_score` — a candidate that "arrives"
    // on a launcher or a tank that doesn't exist must never outrank a
    // feasible candidate that still has distance to close.
    let prop_over = (thrust_dv - cfg.engine.max_dv_kms()).max(0.0);
    let score = mission_score(
        cfg,
        vinf_dep,
        arrival_dv,
        g.legs[0],
        thrust_dv * 0.25 // propellant is cheaper than launch/capture Δv
            + excess_miss * 1e-6
            + miss_km * 3e-8,
        prop_over,
    );

    Solution {
        genome: g.clone(),
        score,
        vinf_dep_kms: vinf_dep,
        vinf_arr_kms: v_rel,
        arrival_dv_kms: arrival_dv,
        thrust_dv_kms: thrust_dv,
        assist_dv_kms: 0.0,
        dsm_dv_kms: 0.0,
        flybys: Vec::new(),
        miss_km,
        depart,
        arrive,
        traj,
    }
}

/// Patched-conic tour evaluation: Lambert arcs between consecutive bodies,
/// flybys checked against the physical bending limit
/// δ_max = 2·asin(μ/(μ + r_p·v∞²)) at minimum safe periapsis, with powered-
/// flyby Δv charged for the |v∞| mismatch and any turn deficit. This is the
/// standard first-order tour-scouting formulation (STOUR/GALLOP class), now
/// with an optional deep-space maneuver per leg (MGA-1DSM); [`refine_tour`]
/// re-flies a chosen tour under the full n-body dynamics.
///
/// Node epochs and body states for a tour genome.
fn tour_nodes(
    eph: &Ephemeris,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
) -> (Vec<BodyId>, Vec<Epoch>, Vec<crate::ephemeris::StateVec>) {
    let seq = cfg.sequence();
    let depart = epoch0 + Duration::from_seconds(g.depart_days * DAY_S);
    let mut epochs = vec![depart];
    for tof in &g.legs {
        epochs.push(*epochs.last().unwrap() + Duration::from_seconds(tof * DAY_S));
    }
    let states = seq
        .iter()
        .zip(&epochs)
        .map(|(b, e)| eph.state(*b, *e))
        .collect();
    (seq, epochs, states)
}

fn evaluate_tour(
    eph: &Ephemeris,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
    n_samples: usize,
) -> Solution {
    let mu = BodyId::Sun.gm();
    let (seq, epochs, states) = tour_nodes(eph, cfg, epoch0, g);
    let depart = epochs[0];

    let bad = |penalty: f64| Solution {
        genome: g.clone(),
        score: 1e4 + penalty,
        vinf_dep_kms: 0.0,
        vinf_arr_kms: 0.0,
        arrival_dv_kms: 0.0,
        thrust_dv_kms: 0.0,
        assist_dv_kms: 0.0,
        dsm_dv_kms: 0.0,
        flybys: Vec::new(),
        miss_km: f64::INFINITY,
        depart,
        arrive: *epochs.last().unwrap(),
        traj: Vec::new(),
    };

    // Solve each leg over every feasible revolution count and branch, with the
    // genome's deep-space maneuver if it carries one.
    let mut arcs: Vec<LegArc> = Vec::with_capacity(g.legs.len());
    for (i, tof) in g.legs.iter().enumerate() {
        let arc = solve_leg(
            states[i].pos_km,
            states[i + 1].pos_km,
            tof * DAY_S,
            mu,
            states[i].vel_km_s,
            g.leg_dsm(i),
            g.leg_branch(i),
        );
        match arc {
            Some(a) => arcs.push(a),
            None => return bad(100.0 * (i + 1) as f64),
        }
    }
    let dsm_dv: f64 = arcs.iter().map(|a| a.dsm_dv).sum();

    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    let vinf_dep = norm(sub(arcs[0].v_start, states[0].vel_km_s));
    let vinf_arr = norm(sub(
        arcs.last().unwrap().v_end,
        states.last().unwrap().vel_km_s,
    ));

    // Flyby feasibility at each intermediate body.
    let mut flybys = Vec::new();
    let mut assist_dv = 0.0;
    for k in 1..seq.len() - 1 {
        let body = seq[k];
        let vin = sub(arcs[k - 1].v_end, states[k].vel_km_s);
        let vout = sub(arcs[k].v_start, states[k].vel_km_s);
        let (nin, nout) = (norm(vin), norm(vout));
        if nin < 1e-3 || nout < 1e-3 {
            return bad(50.0);
        }
        // Powered flyby: the magnitude mismatch is paid in Δv.
        let mut dv = (nin - nout).abs();
        // Turn achievable at minimum safe periapsis for the mean v∞.
        let vinf = 0.5 * (nin + nout);
        let mu_b = body.gm();
        let rp_min = 1.1 * body.radius_km();
        let delta_max = 2.0 * (mu_b / (mu_b + rp_min * vinf * vinf)).asin();
        let cos_turn = ((vin[0] * vout[0] + vin[1] * vout[1] + vin[2] * vout[2])
            / (nin * nout))
            .clamp(-1.0, 1.0);
        let turn = cos_turn.acos();
        let periapsis_alt_km = if turn <= delta_max && turn > 1e-6 {
            // Invert δ(r_p) for the implied periapsis.
            let s = (turn / 2.0).sin();
            (mu_b * (1.0 - s) / (s * vinf * vinf) - body.radius_km()).max(0.0)
        } else {
            0.0
        };
        if turn > delta_max {
            // Charge the deficit as if rotated impulsively at v∞.
            dv += 2.0 * vinf * ((turn - delta_max) / 2.0).sin();
        }
        assist_dv += dv;
        flybys.push(Flyby {
            body,
            epoch: epochs[k],
            vinf_kms: vinf,
            dv_kms: dv,
            periapsis_alt_km,
        });
    }

    let total_tof = g.total_tof_days();
    let arrival_dv = arrival_dv_kms(cfg.target, vinf_arr, cfg.mission);
    // DSM Δv is real propellant, charged exactly like powered-flyby Δv.
    let score = mission_score(cfg, vinf_dep, arrival_dv, total_tof, assist_dv + dsm_dv, 0.0);

    // Sample each conic for rendering — two sub-arcs when the leg has a DSM.
    let per_leg = (n_samples / g.legs.len().max(1)).max(8);
    let mut traj = Vec::with_capacity(per_leg * g.legs.len() + 1);
    for (i, tof) in g.legs.iter().enumerate() {
        let tof_s = tof * DAY_S;
        // (start position, start velocity, elapsed offset, duration) per sub-arc.
        let subs: Vec<([f64; 3], [f64; 3], f64, f64)> = match arcs[i].dsm {
            Some((t1, rm, vm_req)) => vec![
                (states[i].pos_km, arcs[i].v_start, 0.0, t1),
                (rm, vm_req, t1, tof_s - t1),
            ],
            None => vec![(states[i].pos_km, arcs[i].v_start, 0.0, tof_s)],
        };
        for (r0, v0, t0, span) in subs {
            // Samples proportional to the sub-arc's share of the leg.
            let n = ((per_leg as f64 * span / tof_s).round() as usize).max(4);
            for j in 0..n {
                let dt = span * j as f64 / n as f64;
                let (r, v) = kepler_universal(r0, v0, dt, mu);
                traj.push((
                    epochs[i] + Duration::from_seconds(t0 + dt),
                    ScState { pos: r, vel: v },
                ));
            }
        }
    }
    traj.push((
        *epochs.last().unwrap(),
        ScState {
            pos: states.last().unwrap().pos_km,
            vel: arcs.last().unwrap().v_end,
        },
    ));

    Solution {
        genome: g.clone(),
        score,
        vinf_dep_kms: vinf_dep,
        vinf_arr_kms: vinf_arr,
        arrival_dv_kms: arrival_dv,
        thrust_dv_kms: 0.0,
        assist_dv_kms: assist_dv,
        dsm_dv_kms: dsm_dv,
        flybys,
        miss_km: 0.0,
        depart,
        arrive: *epochs.last().unwrap(),
        traj,
    }
}

fn evaluate_direct(
    eph: &Ephemeris,
    dyn_cfg: &DynamicsConfig,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
    n_samples: usize,
) -> Solution {
    let depart = epoch0 + Duration::from_seconds(g.depart_days * DAY_S);
    let earth = eph.state(BodyId::Earth, depart);
    // Inject from the edge of Earth's SOI along v∞, not from Earth's center —
    // that's what v∞ *means* (the escape hyperbola is already flown), and it
    // keeps Earth's point-mass term from collapsing the integrator step. For
    // near-zero v∞ (degenerate candidates) offset along Earth's velocity.
    const EARTH_SOI_KM: f64 = 925_000.0;
    let vmag =
        (g.vinf_dep[0].powi(2) + g.vinf_dep[1].powi(2) + g.vinf_dep[2].powi(2)).sqrt();
    let dir = if vmag > 0.05 {
        [
            g.vinf_dep[0] / vmag,
            g.vinf_dep[1] / vmag,
            g.vinf_dep[2] / vmag,
        ]
    } else {
        let ev = earth.vel_km_s;
        let en = (ev[0].powi(2) + ev[1].powi(2) + ev[2].powi(2)).sqrt();
        [ev[0] / en, ev[1] / en, ev[2] / en]
    };
    let s0 = ScState {
        pos: [
            earth.pos_km[0] + dir[0] * EARTH_SOI_KM,
            earth.pos_km[1] + dir[1] * EARTH_SOI_KM,
            earth.pos_km[2] + dir[2] * EARTH_SOI_KM,
        ],
        vel: [
            earth.vel_km_s[0] + g.vinf_dep[0],
            earth.vel_km_s[1] + g.vinf_dep[1],
            earth.vel_km_s[2] + g.vinf_dep[2],
        ],
    };
    let prop = dynamics::propagate_checked(
        eph,
        dyn_cfg,
        depart,
        s0,
        Duration::from_seconds(g.legs[0] * DAY_S),
        n_samples,
        None,
    );
    let traj = prop.points;
    let (arrive, sf) = *traj.last().unwrap();
    if !prop.complete {
        return infeasible(g, prop.fraction, vmag, depart, arrive, traj);
    }
    let tgt = eph.state(cfg.target, arrive);
    let dr = [
        sf.pos[0] - tgt.pos_km[0],
        sf.pos[1] - tgt.pos_km[1],
        sf.pos[2] - tgt.pos_km[2],
    ];
    let miss_km = (dr[0].powi(2) + dr[1].powi(2) + dr[2].powi(2)).sqrt();
    let dv = [
        sf.vel[0] - tgt.vel_km_s[0],
        sf.vel[1] - tgt.vel_km_s[1],
        sf.vel[2] - tgt.vel_km_s[2],
    ];
    let vinf_arr = (dv[0].powi(2) + dv[1].powi(2) + dv[2].powi(2)).sqrt();
    let vinf_dep =
        (g.vinf_dep[0].powi(2) + g.vinf_dep[1].powi(2) + g.vinf_dep[2].powi(2)).sqrt();

    // Miss beyond the SOI is charged at 1 km/s per million km — steep enough
    // to dominate until candidates actually reach the target, gentle enough
    // to leave a gradient. Inside the SOI a much weaker pull (0.03 km/s per
    // million km) keeps refining toward the target center instead of stalling
    // at the SOI boundary.
    let excess = (miss_km - soi_km(eph, cfg.target, arrive)).max(0.0);
    let arrival_dv = arrival_dv_kms(cfg.target, vinf_arr, cfg.mission);
    let score = mission_score(
        cfg,
        vinf_dep,
        arrival_dv,
        g.legs[0],
        excess * 1e-6 + miss_km * 3e-8,
        0.0,
    );

    Solution {
        genome: g.clone(),
        score,
        vinf_dep_kms: vinf_dep,
        vinf_arr_kms: vinf_arr,
        arrival_dv_kms: arrival_dv,
        thrust_dv_kms: 0.0,
        assist_dv_kms: 0.0,
        dsm_dv_kms: 0.0,
        flybys: Vec::new(),
        miss_km,
        depart,
        arrive,
        traj,
    }
}

/// Quantized identity for beam dedup: candidates within 0.1 d in timing and
/// 10 m/s in v∞ are "the same idea" and only the best-scored one survives.
fn genome_key(g: &Genome) -> u64 {
    // FNV-1a over the quantized fields — no allocation, deterministic.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: i64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    mix((g.depart_days * 10.0) as i64);
    for leg in &g.legs {
        mix((leg * 10.0) as i64);
    }
    for v in &g.vinf_dep {
        mix((v * 100.0) as i64);
    }
    for seg in &g.thrust {
        for c in seg {
            mix((c * 20.0) as i64);
        }
    }
    for node in &g.dsm {
        mix((node[0] * 50.0) as i64); // 2% buckets on the burn's position
        for c in &node[1..] {
            mix((c * 100.0) as i64); // 10 m/s buckets, same as v∞
        }
    }
    for b in &g.branch {
        mix(*b as i64); // a different arc is a different trajectory
    }
    h
}

/// The search itself, stepwise so tests can drive it deterministically.
/// Coarse trajectory sampling while searching; callers re-propagate the
/// winner densely.
pub struct Search {
    pub cfg: SolverConfig,
    epoch0: Epoch,
    dyn_cfg: DynamicsConfig,
    /// Pre-sampled interpolated ephemeris covering the whole search window —
    /// evals never touch the SPK tree, which is ~100x faster and still
    /// deterministic (the table is a pure function of source + span + step).
    fast_eph: Arc<Ephemeris>,
    leg_bounds: Vec<(f64, f64)>,
    rng: Rng,
    beam: Vec<(f64, Genome)>,
    /// Seed-channel telemetry: (uniform-channel draws, launch-feasible draws,
    /// total timing draws made). Records how often a uniform timing draw is
    /// launcher-feasible instead of silently replacing the distribution.
    seed_draws: SeedDrawStats,
}

/// Counts of timing draws by seed channel, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeedDrawStats {
    /// Genomes seeded from the uniform channel (timing kept as drawn).
    pub uniform_genomes: usize,
    /// Genomes seeded from the launch-feasible channel.
    pub feasible_genomes: usize,
    /// Feasible-channel genomes where no draw in the batch met the C3 cap, so
    /// the cheapest draw was kept as a fallback.
    pub feasible_misses: usize,
    /// Individual timing draws evaluated for launcher feasibility.
    pub draws: usize,
    /// How many of those were within the launcher's C3 cap.
    pub draws_feasible: usize,
}

impl SeedDrawStats {
    /// Fraction of uniform timing draws that a real launcher could fly.
    pub fn feasible_rate(&self) -> f64 {
        if self.draws == 0 {
            return f64::NAN;
        }
        self.draws_feasible as f64 / self.draws as f64
    }
}

/// Timing draws taken per feasible-channel seed. Fixed, and always fully
/// consumed, so RNG consumption never depends on Lambert reachability.
pub const SEED_DRAWS: usize = 12;

const SAMPLES: usize = 40;

impl Search {
    /// `prebuilt`: a cached-span ephemeris already covering the search window
    /// (e.g. built at app startup) — pass it and the search starts instantly.
    pub fn new(
        eph: &Ephemeris,
        cfg: SolverConfig,
        epoch0: Epoch,
        prebuilt: Option<Arc<Ephemeris>>,
    ) -> Self {
        let dyn_cfg = solver_dynamics();
        let span_end = epoch0
            + Duration::from_days(cfg.window_days + cfg.max_total_tof_days() + 2.0);
        let fast_eph = prebuilt.unwrap_or_else(|| {
            Arc::new(eph.cached_span(
                epoch0 - Duration::from_days(1.0),
                span_end,
                &dyn_cfg.perturbers,
            ))
        });
        let rng = Rng::new(cfg.seed);
        let leg_bounds = if cfg.route.is_empty() {
            vec![(cfg.tof_min_days, cfg.tof_max_days)]
        } else {
            cfg.leg_bounds()
        };
        let beam_width = cfg.beam_width;
        let mut s = Self {
            cfg,
            epoch0,
            dyn_cfg,
            fast_eph,
            leg_bounds,
            rng,
            beam: Vec::new(),
            seed_draws: SeedDrawStats::default(),
        };
        let mut beam: Vec<(f64, Genome)> = (0..beam_width)
            .map(|_| {
                let g = s.random_genome();
                (s.eval_score(&g), g)
            })
            .collect();
        beam.sort_by(|a, b| a.0.total_cmp(&b.0));
        s.beam = beam;
        s
    }

    /// The current beam's genomes, best-first.
    pub fn beam_genomes(&self) -> Vec<Genome> {
        self.beam.iter().map(|(_, g)| g.clone()).collect()
    }

    /// Seed-channel telemetry, including the launcher-feasible draw rate.
    pub fn seed_draw_stats(&self) -> SeedDrawStats {
        self.seed_draws
    }

    fn eval_score(&self, g: &Genome) -> f64 {
        evaluate(&self.fast_eph, &self.dyn_cfg, &self.cfg, self.epoch0, g, SAMPLES).score
    }

    /// Lambert-aim a v∞ for a (depart, tof) pair: solve the two-body boundary
    /// value problem Earth→target and subtract Earth's velocity. This is what
    /// makes the search *global* — every fresh candidate is already pointed at
    /// the target, so distinct launch windows compete on merit instead of the
    /// beam converging into whichever window it saw first.
    fn lambert_vinf(&self, depart_days: f64, tof_days: f64) -> Option<[f64; 3]> {
        let depart = self.epoch0 + Duration::from_seconds(depart_days * DAY_S);
        let arrive = depart + Duration::from_seconds(tof_days * DAY_S);
        let earth = self.fast_eph.state(BodyId::Earth, depart);
        let tgt = self.fast_eph.state(self.cfg.target, arrive);
        let mu = BodyId::Sun.gm();
        // Closest-to-Earth's-velocity arc = smallest v∞, over every feasible
        // revolution count and branch.
        let (v1, _) = lambert_best(
            earth.pos_km,
            tgt.pos_km,
            tof_days * DAY_S,
            mu,
            earth.vel_km_s,
        )?;
        Some([
            v1[0] - earth.vel_km_s[0],
            v1[1] - earth.vel_km_s[1],
            v1[2] - earth.vel_km_s[2],
        ])
    }

    /// One random (departure epoch, leg flight times) draw.
    fn draw_timing(&mut self, sep: bool) -> (f64, Vec<f64>) {
        let depart_days = self.rng.unit() * self.cfg.window_days;
        let seq = self.cfg.sequence();
        let bounds = self.leg_bounds.clone();
        let legs: Vec<f64> = bounds
            .iter()
            .enumerate()
            .map(|(i, (lo, hi))| {
                if sep && self.cfg.mission != MissionType::Flyby {
                    // SEP capture missions need cruise time to brake — skew
                    // the flight-time seeds long or the beam anchors in the
                    // hot short-TOF regime and never escapes.
                    let lo2 = lo.max(0.4 * hi);
                    return lo2 + self.rng.unit() * (hi - lo2);
                }
                if seq[i] == seq[i + 1] {
                    // Resonant return legs (e.g. Earth→Earth in a VEEGA) only
                    // work near integer multiples of the body's year — seed
                    // there instead of uniformly.
                    let year = seq[i].period_days().abs();
                    let n = 1 + self.rng.below(3) as i32;
                    (year * n as f64 * (0.97 + 0.06 * self.rng.unit())).clamp(*lo, *hi)
                } else {
                    lo + self.rng.unit() * (hi - lo)
                }
            })
            .collect();
        (depart_days, legs)
    }

    fn random_genome(&mut self) -> Genome {
        let sep = self.cfg.engine != Engine::Ballistic && self.cfg.route.is_empty();
        let ballistic_direct = self.cfg.route.is_empty() && !sep;
        // Ballistic direct seeds are Lambert-aimed, and a Lambert aim for an
        // arbitrary (departure, TOF) pair routinely wants C3 far past the
        // launcher's cap — `mission_score` charges that as infeasible, so such
        // a seed is dropped from the beam on arrival and the fresh-seed
        // channel stops contributing. Scaling the v∞ down would keep the
        // candidate but destroy the aim (it is no longer an intercept). So run
        // two explicit channels instead of quietly reshaping one distribution:
        //
        //   - uniform: timing kept exactly as drawn, so short-TOF / good
        //     arrival-v∞ regions with moderate C3 stay reachable (they matter
        //     under the TimeOfFlight and ArrivalVinf objectives, where C3 is
        //     not the goal);
        //   - launch-feasible: a batch of SEED_DRAWS draws, one of the
        //     launcher-feasible ones picked *uniformly* (not the first, and
        //     not the min-C3 order statistic), falling back to the cheapest
        //     draw when the batch finds none.
        //
        // The batch is always drawn in full and the channel/pick selectors are
        // always consumed, so RNG consumption is a function of the config
        // alone — a change in Lambert reachability no longer shifts the whole
        // deterministic sequence (docs/DETERMINISM.md).
        let (depart_days, legs) = if ballistic_direct {
            let uniform_channel = self.rng.below(3) == 0;
            let pick = self.rng.below(SEED_DRAWS);
            let mut draws: Vec<(f64, f64, Vec<f64>)> = Vec::with_capacity(SEED_DRAWS);
            for _ in 0..SEED_DRAWS {
                let (d, l) = self.draw_timing(sep);
                let c3 = self
                    .lambert_vinf(d, l[0])
                    .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                    .unwrap_or(f64::INFINITY);
                draws.push((c3, d, l));
            }
            let cap = self.cfg.launcher.c3_max();
            let feasible: Vec<usize> = (0..SEED_DRAWS).filter(|&i| draws[i].0 <= cap).collect();
            self.seed_draws.draws += SEED_DRAWS;
            self.seed_draws.draws_feasible += feasible.len();
            let idx = if uniform_channel {
                self.seed_draws.uniform_genomes += 1;
                // The first draw is an untouched uniform sample.
                0
            } else {
                self.seed_draws.feasible_genomes += 1;
                if feasible.is_empty() {
                    self.seed_draws.feasible_misses += 1;
                    // Nothing flyable in the batch: keep the cheapest, which is
                    // still the best available lead into a launch window.
                    (0..SEED_DRAWS)
                        .min_by(|&a, &b| draws[a].0.total_cmp(&draws[b].0))
                        .unwrap()
                } else {
                    feasible[pick % feasible.len()]
                }
            };
            let (_, d, l) = draws.swap_remove(idx);
            (d, l)
        } else {
            self.draw_timing(sep)
        };
        let vinf_dep = if !self.cfg.route.is_empty() {
            [0.0; 3] // tours derive v∞ from the first Lambert leg
        } else if sep && self.rng.below(2) == 0 {
            // Half the SEP candidates: classic spiral start — max launcher
            // escape along Earth's prograde direction; the engine does the
            // rest. (Ballistic Lambert arcs fight the thrust profile.)
            let depart = self.epoch0 + Duration::from_seconds(depart_days * DAY_S);
            let ev = self.fast_eph.state(BodyId::Earth, depart).vel_km_s;
            let en = (ev[0].powi(2) + ev[1].powi(2) + ev[2].powi(2)).sqrt();
            let v = self.cfg.launcher.c3_max().sqrt() * (0.5 + 0.5 * self.rng.unit());
            [ev[0] / en * v, ev[1] / en * v, ev[2] / en * v]
        } else {
            let mut v = self.lambert_vinf(depart_days, legs[0]).unwrap_or([
                self.rng.sym() * 5.0,
                self.rng.sym() * 5.0,
                self.rng.sym() * 2.0,
            ]);
            if sep {
                // Never seed a launch the selected rocket cannot buy.
                let cap = self.cfg.launcher.c3_max().sqrt() * 0.98;
                let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if n > cap {
                    for c in &mut v {
                        *c *= cap / n;
                    }
                }
            }
            v
        };
        let thrust = if self.cfg.engine != Engine::Ballistic && self.cfg.route.is_empty() {
            // Seed the classic SEP profile for capture missions: thrust
            // prograde early, coast, brake hard on approach — then scale the
            // whole profile so it burns ~70% of the tank. Over a decades-long
            // cruise an unscaled profile would burn 5-10x the propellant that
            // exists. Flybys start from coast.
            let mut t: Vec<[f64; 3]> = vec![[0.0; 3]; N_SEG];
            if self.cfg.mission != MissionType::Flyby {
                // Brake-only: the launcher buys the escape; the engine's job
                // is bleeding off arrival speed across the last part of the
                // cruise.
                for seg in t.iter_mut().skip(N_SEG / 2) {
                    *seg = [-0.85, 0.0, 0.0];
                }
                let seg_dt = legs[0] * DAY_S / N_SEG as f64;
                let seed_dv: f64 = t
                    .iter()
                    .map(|u| {
                        (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()
                            * self.cfg.engine.accel_kms2()
                            * seg_dt
                    })
                    .sum();
                let budget = 0.7 * self.cfg.engine.max_dv_kms();
                if seed_dv > budget {
                    let k = budget / seed_dv;
                    for seg in &mut t {
                        for c in seg.iter_mut() {
                            *c *= k;
                        }
                    }
                }
            }
            t
        } else {
            Vec::new()
        };
        // Tours get a DSM node per leg, seeded *inert* (zero Δv, mid-leg):
        // an inert node is bit-for-bit the ballistic leg, so seeding them can
        // only ever open search freedom, never cost score. Mutation is what
        // switches one on.
        let dsm = if self.cfg.route.is_empty() {
            Vec::new()
        } else {
            vec![[0.5, 0.0, 0.0, 0.0]; legs.len()]
        };
        // Branch 0 on every leg is the cheapest-departure arc — exactly what
        // the greedy solver used to commit to unconditionally. Seeding it
        // means the alternatives cost nothing until mutation tries one.
        let branch = vec![0u32; if self.cfg.route.is_empty() { 0 } else { legs.len() }];
        Genome {
            depart_days,
            legs,
            vinf_dep,
            thrust,
            dsm,
            branch,
        }
    }

    fn mutate_genome(&mut self, g: &Genome) -> Genome {
        let mut g = g.clone();
        let tour = !self.cfg.route.is_empty();
        // Low-thrust genomes: half the edits go to the throttle profile.
        if !g.thrust.is_empty() && self.rng.below(2) == 0 {
            for _ in 0..1 + self.rng.below(2) {
                let i = self.rng.below(g.thrust.len());
                match self.rng.below(3) {
                    0 => {
                        let c = self.rng.below(3);
                        g.thrust[i][c] = (g.thrust[i][c] + self.rng.sym() * 0.3).clamp(-1.0, 1.0);
                    }
                    1 => {
                        // Scale the whole profile — walks the propellant
                        // budget up or down without losing its shape.
                        let k = 0.7 + 0.6 * self.rng.unit();
                        for seg in &mut g.thrust {
                            for c in seg.iter_mut() {
                                *c = (*c * k).clamp(-1.0, 1.0);
                            }
                        }
                    }
                    _ => {
                        // Full along-track burn, either direction.
                        let s = if self.rng.below(2) == 0 { 1.0 } else { -1.0 };
                        g.thrust[i] = [s * (0.5 + 0.5 * self.rng.unit()), 0.0, 0.0];
                    }
                }
            }
            return g;
        }
        for _ in 0..1 + self.rng.below(3) {
            match self.rng.below(5) {
                0 => {
                    g.depart_days =
                        (g.depart_days + self.rng.sym() * 40.0).clamp(0.0, self.cfg.window_days)
                }
                1 => {
                    // Nudge one leg's flight time.
                    let i = self.rng.below(g.legs.len());
                    let (lo, hi) = self.leg_bounds[i];
                    g.legs[i] = (g.legs[i] + self.rng.sym() * 40.0).clamp(lo, hi);
                }
                2 if !tour => {
                    let i = self.rng.below(3);
                    g.vinf_dep[i] += self.rng.sym() * 0.6;
                }
                3 if !tour => {
                    // Fine-tune all three components at once.
                    for v in &mut g.vinf_dep {
                        *v += self.rng.sym() * 0.05;
                    }
                }
                2 | 3 => {
                    if !g.branch.is_empty() && self.rng.below(6) == 0 {
                        // Tours: try a different Lambert arc for one leg.
                        // The local cheapest-departure pick is only a
                        // heuristic — the full mission score, which does see
                        // the flyby turn, the arrival v∞ and the capture
                        // burn, decides whether the alternative survives.
                        let i = self.rng.below(g.branch.len());
                        g.branch[i] = if self.rng.below(3) == 0 {
                            0 // snap back to the locally cheapest arc
                        } else {
                            self.rng.below(4) as u32
                        };
                    } else if !g.dsm.is_empty() && self.rng.below(2) == 0 {
                        // Tours: edit one leg's deep-space maneuver. Small
                        // kicks — a DSM is a course shaping burn, and the
                        // score charges every m/s of it, so the search has to
                        // discover that a burn pays for itself.
                        let i = self.rng.below(g.dsm.len());
                        if self.rng.below(5) == 0 {
                            // Zero the node: an inert DSM is exactly the
                            // ballistic leg, so the burn must out-score its
                            // own absence to survive selection.
                            g.dsm[i] = [g.dsm[i][0], 0.0, 0.0, 0.0];
                        } else if self.rng.below(4) == 0 {
                            // Slide the burn along the leg.
                            g.dsm[i][0] = (g.dsm[i][0] + self.rng.sym() * 0.15)
                                .clamp(DSM_FRAC_RANGE.0, DSM_FRAC_RANGE.1);
                        } else {
                            let c = 1 + self.rng.below(3);
                            g.dsm[i][c] = (g.dsm[i][c] + self.rng.sym() * 0.15)
                                .clamp(-2.0, 2.0);
                        }
                    } else {
                        // Fine-nudge one leg — flyby feasibility is very
                        // sensitive to timing, so small edits matter.
                        let i = self.rng.below(g.legs.len());
                        let (lo, hi) = self.leg_bounds[i];
                        g.legs[i] = (g.legs[i] + self.rng.sym() * 4.0).clamp(lo, hi);
                    }
                }
                _ => {
                    if tour {
                        // Resample one leg wholesale to hop between geometry
                        // families (e.g. 2:1 vs 3:2 resonant returns).
                        let i = self.rng.below(g.legs.len());
                        let (lo, hi) = self.leg_bounds[i];
                        g.legs[i] = lo + self.rng.unit() * (hi - lo);
                    } else {
                        // Re-aim with Lambert after the timing edits above —
                        // pulls a drifted candidate back onto an intercept
                        // course.
                        if let Some(v) = self.lambert_vinf(g.depart_days, g.legs[0]) {
                            g.vinf_dep = v;
                        }
                    }
                }
            }
        }
        g
    }

    /// One expand-select generation. Returns (evals done, current top).
    pub fn step(&mut self, _eph: &Ephemeris) -> (u64, (f64, Genome)) {
        let mut pool = self.beam.clone();
        for i in 0..self.beam.len() {
            for _ in 0..self.cfg.mutations {
                let parent = self.beam[i].1.clone();
                let m = self.mutate_genome(&parent);
                pool.push((self.eval_score(&m), m));
            }
        }
        // A fresh Lambert-seeded candidate each step: lands on an intercept
        // course anywhere in the window, so it can beat the incumbent beam.
        let fresh = self.random_genome();
        pool.push((self.eval_score(&fresh), fresh));

        pool.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Dedup before truncating: without this, a strong candidate's
        // near-identical mutations flood the beam and diversity collapses
        // (clones crowd out genuinely different timings/windows). Quantized
        // key: 0.1-day timing buckets, 10 m/s v∞ buckets. HashSet is
        // insert-only here, so determinism is unaffected.
        let mut seen = std::collections::HashSet::new();
        pool.retain(|(_, g)| seen.insert(genome_key(g)));
        pool.truncate(self.cfg.beam_width);
        self.beam = pool;
        (
            (self.cfg.beam_width * self.cfg.mutations + 1) as u64,
            self.beam[0].clone(),
        )
    }

    /// Densely sampled so the dotted live-preview reads as a smooth arc.
    pub fn solution_for(&self, _eph: &Ephemeris, g: &Genome) -> Solution {
        evaluate(&self.fast_eph, &self.dyn_cfg, &self.cfg, self.epoch0, g, 240)
    }
}

/// Phase-two polish (the "hand the winner to a local corrector" stage of a
/// classic two-phase trajectory pipeline): Newton shooting on the departure
/// v∞ so the full-fidelity propagation lands on the target's center at
/// arrival. Zeroth-order beam search gets within ~1e5 km; three or four
/// Newton iterations with a finite-difference Jacobian take it to ~km.
/// Deterministic: fixed iteration count/order, pure f64.
pub fn differential_correct(
    eph: &Ephemeris,
    dyn_cfg: &DynamicsConfig,
    depart: Epoch,
    tof: Duration,
    target: BodyId,
    vinf0: [f64; 3],
) -> ([f64; 3], f64) {
    let arrive = depart + tof;
    let tgt = eph.state(target, arrive);
    // `None` when the propagation truncated (step exhaustion): the trajectory's
    // last point is then an early cutoff, not the arrival state, so using it
    // would feed a residual for the wrong epoch into the Jacobian.
    let shoot = |vinf: [f64; 3]| -> Option<[f64; 3]> {
        let g = Genome {
            depart_days: 0.0,
            legs: vec![tof.to_seconds() / DAY_S],
            vinf_dep: vinf,
            thrust: Vec::new(),
            dsm: Vec::new(),
            branch: Vec::new(),
        };
        // Correct against the *requested* target: `evaluate_direct` computes
        // its SOI and arrival Δv from `cfg.target`, so defaulting it here
        // would shoot at Mars no matter which body was asked for.
        let cfg = SolverConfig {
            route: Vec::new(),
            target,
            ..Default::default()
        };
        let sol = evaluate_direct(eph, dyn_cfg, &cfg, depart, &g, 2);
        if !sol.miss_km.is_finite() {
            return None; // truncated: `infeasible` marks it with an infinite miss
        }
        let sf = sol.traj.last().unwrap().1;
        Some([
            sf.pos[0] - tgt.pos_km[0],
            sf.pos[1] - tgt.pos_km[1],
            sf.pos[2] - tgt.pos_km[2],
        ])
    };
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    let mut vinf = vinf0;
    let Some(mut f) = shoot(vinf) else {
        return (vinf0, f64::INFINITY);
    };
    // Newton on an adaptively-integrated arc is not monotone: the step-size
    // controller makes the residual jitter by ~1e3 km under sub-m/s input
    // changes, so a late iterate can be far worse than an early one. Keep the
    // best iterate seen and return that rather than wherever the walk ended.
    let (mut best_vinf, mut best_norm) = (vinf, norm(f));
    for _ in 0..30 {
        if best_norm < 500.0 {
            break;
        }
        // Finite-difference Jacobian dF/dv∞, column per component. H must be
        // large enough that the induced arrival displacement clears that same
        // integrator jitter: at 1e-4 km/s the column is ~1e3 km of signal on
        // ~1e3 km of noise and Newton stalls around a 1000 km miss. 1e-3 km/s
        // moves arrival by ~1e4 km, an order of magnitude above the floor,
        // while staying well inside the linear regime.
        const H: f64 = 1e-3;
        let mut jac = [[0.0f64; 3]; 3];
        for c in 0..3 {
            let mut vp = vinf;
            vp[c] += H;
            let Some(fp) = shoot(vp) else {
                return (best_vinf, best_norm);
            };
            for r in 0..3 {
                jac[r][c] = (fp[r] - f[r]) / H;
            }
        }
        // Solve J·dv = -F (Cramer's rule; 3x3).
        let det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
            - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
            + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
        if det.abs() < 1e-12 {
            break;
        }
        let rhs = [-f[0], -f[1], -f[2]];
        let solve_col = |col: usize| {
            let mut m = jac;
            for r in 0..3 {
                m[r][col] = rhs[r];
            }
            (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                / det
        };
        let dv = [solve_col(0), solve_col(1), solve_col(2)];
        // Damp absurd steps (near-singular geometry).
        let dvn = norm(dv);
        let mut scale = if dvn > 1.0 { 1.0 / dvn } else { 1.0 };
        // Backtracking line search: the full Newton step overshoots once the
        // residual approaches the noise floor, which is what sent the old
        // fixed-step loop wandering back up into the 1e4 km range. Halve until
        // the step actually improves, then accept it.
        let mut stepped = false;
        for _ in 0..6 {
            let mut trial = vinf;
            for c in 0..3 {
                trial[c] += dv[c] * scale;
            }
            if let Some(ft) = shoot(trial) {
                if norm(ft) < norm(f) {
                    vinf = trial;
                    f = ft;
                    stepped = true;
                    break;
                }
            }
            scale *= 0.5;
        }
        if !stepped {
            break; // no improving step along this direction; best_* already holds the answer
        }
        if norm(f) < best_norm {
            best_norm = norm(f);
            best_vinf = vinf;
        }
    }
    (best_vinf, best_norm)
}

/// Outcome of the two-stage direct-transfer correction: the coarse
/// center-at-epoch Newton pass, then (when the arc reaches the target's SOI)
/// a B-plane pass that drives the arc onto a requested periapsis geometry.
#[derive(Clone, Copy, Debug)]
pub struct DirectCorrection {
    pub vinf_dep: [f64; 3],
    /// Stage-one residual: distance from the target's center at the arrival
    /// epoch, km. Near a planet this bottoms out at a gravitational-focusing
    /// floor (~10^4 km at Mars) and is NOT a targeting error — see stage two.
    pub center_miss_km: f64,
    /// Stage-two residual: |B - B_desired| in the B-plane, km. `None` when
    /// the arc never reached the SOI hyperbolic regime.
    pub bplane_err_km: Option<f64>,
    /// Achieved periapsis altitude above the target's mean radius, km.
    pub periapsis_alt_km: Option<f64>,
    /// Arrival v∞ recomputed from the corrected arc's target-relative energy.
    pub vinf_arr_kms: Option<f64>,
    /// Insertion Δv recomputed at the achieved periapsis (same parked-orbit
    /// model as `arrival_dv_kms`, but with the real rp and v∞).
    pub arrival_dv_kms: Option<f64>,
}

/// Target-relative osculating hyperbola, the quantities B-plane targeting
/// needs. Computed from a single (r, v) sample inside the SOI; two-body about
/// the target is a good local model there even though the arc itself is n-body.
struct Hyperbola {
    /// B-vector, km (from the focus to the incoming asymptote, ⟂ to it).
    b: [f64; 3],
    /// Incoming asymptote unit vector Ŝ.
    s_hat: [f64; 3],
    vinf_kms: f64,
    rp_km: f64,
    /// Signed time from the sample epoch to periapsis passage, s.
    dt_to_periapsis_s: f64,
}

fn hyperbola_at(mu: f64, r: [f64; 3], v: [f64; 3]) -> Option<Hyperbola> {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let rn = dot(r, r).sqrt();
    let v2 = dot(v, v);
    let eps = v2 / 2.0 - mu / rn;
    if eps <= 0.0 || rn == 0.0 {
        return None; // not hyperbolic w.r.t. the target
    }
    let vinf = (2.0 * eps).sqrt();
    let a = -mu / (2.0 * eps); // < 0
    let h = cross(r, v);
    let hn = dot(h, h).sqrt();
    // e-vector from the vis-viva identity; e > 1 guaranteed by eps > 0.
    let rv = dot(r, v);
    let e_vec = [
        ((v2 - mu / rn) * r[0] - rv * v[0]) / mu,
        ((v2 - mu / rn) * r[1] - rv * v[1]) / mu,
        ((v2 - mu / rn) * r[2] - rv * v[2]) / mu,
    ];
    let e = dot(e_vec, e_vec).sqrt();
    if e <= 1.0 || hn == 0.0 {
        return None;
    }
    let e_hat = [e_vec[0] / e, e_vec[1] / e, e_vec[2] / e];
    let h_hat = [h[0] / hn, h[1] / hn, h[2] / hn];
    let p_hat = cross(h_hat, e_hat);
    // Incoming asymptote: velocity direction at ν = -ν∞, where cos ν∞ = -1/e.
    let (c, s) = (1.0 / e, (1.0 - 1.0 / (e * e)).sqrt());
    let s_hat = [
        c * e_hat[0] + s * p_hat[0],
        c * e_hat[1] + s * p_hat[1],
        c * e_hat[2] + s * p_hat[2],
    ];
    let b_mag = hn / vinf;
    let bv = cross(s_hat, h_hat);
    // Time to periapsis via the hyperbolic Kepler equation:
    // r·v = e·sqrt(-a·mu)·sinh H,  M = e·sinh H - H = n(t - t_p).
    let n = (mu / (-a).powi(3)).sqrt();
    let sinh_h = rv / (e * ((-a) * mu).sqrt());
    let hh = sinh_h.asinh();
    let m = e * sinh_h - hh;
    Some(Hyperbola {
        b: [bv[0] * b_mag, bv[1] * b_mag, bv[2] * b_mag],
        s_hat,
        vinf_kms: vinf,
        rp_km: a * (1.0 - e),
        dt_to_periapsis_s: -m / n,
    })
}

/// Desired periapsis radius for the correction, by mission profile. Orbit
/// matches the 1.5·R parked-orbit assumption baked into `arrival_dv_kms`;
/// land aims just above the surface (entry interface); flyby takes a safe
/// close pass.
fn target_periapsis_km(target: BodyId, mission: MissionType) -> f64 {
    let r = target.radius_km();
    match mission {
        MissionType::Orbit => 1.5 * r,
        MissionType::Land => 1.02 * r,
        MissionType::Flyby => 3.0 * r,
    }
}

/// Two-stage direct-transfer correction.
///
/// Stage one is `differential_correct`: Newton on position-at-fixed-epoch
/// against the target's center. That formulation cannot converge below a
/// gravitational-focusing floor — near the planet, gravity deflects close
/// arcs harder the closer they aim, so the map from v∞ to position-at-epoch
/// folds and a ring of positions around the center is unreachable (issue
/// #15; ~9,000 km at Mars for the 2026 window). Stage one's job is only to
/// deliver the arc into the SOI.
///
/// Stage two retargets in the B-plane, where the map is smooth: Newton on
/// (B·T, B·R, periapsis timing) against a B-vector sized for the mission's
/// periapsis radius, oriented along the stage-one approach plane so the
/// corrector changes the arc as little as possible. Axes and the desired B
/// are frozen at the stage-one solution — re-deriving them per iterate would
/// let the goalposts move with the walk.
pub fn correct_direct(
    eph: &Ephemeris,
    dyn_cfg: &DynamicsConfig,
    depart: Epoch,
    tof: Duration,
    target: BodyId,
    mission: MissionType,
    vinf0: [f64; 3],
) -> DirectCorrection {
    let (vinf1, center_miss) = differential_correct(eph, dyn_cfg, depart, tof, target, vinf0);
    let mut out = DirectCorrection {
        vinf_dep: vinf1,
        center_miss_km: center_miss,
        bplane_err_km: None,
        periapsis_alt_km: None,
        vinf_arr_kms: None,
        arrival_dv_kms: None,
    };
    let arrive = depart + tof;
    if center_miss > soi_km(eph, target, arrive) {
        return out; // never reached the SOI; B-plane elements undefined
    }
    let mu = target.gm();
    // Target-relative state of the full n-body arc at the arrival epoch.
    let rel_state = |vinf: [f64; 3]| -> Option<([f64; 3], [f64; 3])> {
        let earth = eph.state(BodyId::Earth, depart);
        let vmag = (vinf[0].powi(2) + vinf[1].powi(2) + vinf[2].powi(2)).sqrt();
        if vmag < 0.05 {
            return None;
        }
        let dir = [vinf[0] / vmag, vinf[1] / vmag, vinf[2] / vmag];
        let s0 = ScState {
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
        let prop = dynamics::propagate_checked(eph, dyn_cfg, depart, s0, tof, 2, None);
        if !prop.complete {
            return None;
        }
        let (_, sf) = *prop.points.last().unwrap();
        let tgt = eph.state(target, arrive);
        Some((
            [
                sf.pos[0] - tgt.pos_km[0],
                sf.pos[1] - tgt.pos_km[1],
                sf.pos[2] - tgt.pos_km[2],
            ],
            [
                sf.vel[0] - tgt.vel_km_s[0],
                sf.vel[1] - tgt.vel_km_s[1],
                sf.vel[2] - tgt.vel_km_s[2],
            ],
        ))
    };
    let Some((r1, v1)) = rel_state(vinf1) else {
        return out;
    };
    let Some(hyp1) = hyperbola_at(mu, r1, v1) else {
        return out;
    };
    // Frozen B-plane frame from the stage-one asymptote: T̂ in the asymptote's
    // "horizontal" (⟂ to the reference pole), R̂ completing the triad.
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let s0_hat = hyp1.s_hat;
    let t_raw = cross(s0_hat, [0.0, 0.0, 1.0]);
    let tn = dot(t_raw, t_raw).sqrt();
    if tn < 1e-9 {
        return out; // asymptote along the pole; degenerate frame
    }
    let t_hat = [t_raw[0] / tn, t_raw[1] / tn, t_raw[2] / tn];
    let r_hat = cross(s0_hat, t_hat);
    // Desired |B| for the mission's periapsis radius at the stage-one v∞:
    // b = rp·sqrt(1 + 2μ/(rp·v∞²)). Direction: keep the stage-one approach
    // plane (unit B̂ from stage one).
    let rp_des = target_periapsis_km(target, mission);
    let vinf_ref = hyp1.vinf_kms;
    let b_des_mag = rp_des * (1.0 + 2.0 * mu / (rp_des * vinf_ref * vinf_ref)).sqrt();
    let b1_mag = dot(hyp1.b, hyp1.b).sqrt();
    if b1_mag < 1e-6 {
        return out;
    }
    let b_des = [
        (dot(hyp1.b, t_hat) / b1_mag) * b_des_mag,
        (dot(hyp1.b, r_hat) / b1_mag) * b_des_mag,
    ];
    // Residual: B-plane miss plus periapsis timing (scaled to km by v∞ so
    // the three components condition alike).
    let shoot = |vinf: [f64; 3]| -> [f64; 3] {
        let Some((r, v)) = rel_state(vinf) else {
            return [1e9, 1e9, 1e9];
        };
        let Some(hyp) = hyperbola_at(mu, r, v) else {
            return [1e9, 1e9, 1e9];
        };
        [
            dot(hyp.b, t_hat) - b_des[0],
            dot(hyp.b, r_hat) - b_des[1],
            hyp.dt_to_periapsis_s * vinf_ref,
        ]
    };
    let (vinf2, berr) = newton_shoot(&shoot, vinf1, 15, 1.0);
    if !berr.is_finite() || berr >= 1e9 {
        return out;
    }
    out.vinf_dep = vinf2;
    out.bplane_err_km = Some(berr);
    if let Some((r2, v2)) = rel_state(vinf2) {
        if let Some(hyp2) = hyperbola_at(mu, r2, v2) {
            let rp = hyp2.rp_km;
            out.periapsis_alt_km = Some(rp - target.radius_km());
            out.vinf_arr_kms = Some(hyp2.vinf_kms);
            // Same insertion model as `arrival_dv_kms`, at the real periapsis.
            out.arrival_dv_kms = Some(match mission {
                MissionType::Flyby => 0.0,
                MissionType::Orbit => {
                    let vp_hyp = (hyp2.vinf_kms.powi(2) + 2.0 * mu / rp).sqrt();
                    let vp_ell = (mu / rp * (1.0 + 0.95)).sqrt();
                    vp_hyp - vp_ell
                }
                MissionType::Land => arrival_dv_kms(target, hyp2.vinf_kms, mission),
            });
        }
    }
    out
}

/// Damped Newton on a 3-vector residual with a finite-difference Jacobian.
/// Shared by every shooting stage; returns the corrected vector and the final
/// residual norm. Deterministic: fixed iteration count and step order.
///
/// Like `differential_correct`, this backtracks and returns the *best* iterate
/// rather than the last one. The reason is the same and applies with more force
/// here: `refine_tour` shoots n-body arcs, so the residual carries the adaptive
/// integrator's ~1e3 km step-sequence jitter, and a late iterate can be far
/// worse than an early one. Tour legs additionally sit on Lambert branch
/// boundaries, where the residual is genuinely discontinuous and an unguarded
/// full step can jump to a different transfer family entirely.
pub(crate) fn newton_shoot(
    shoot: &dyn Fn([f64; 3]) -> [f64; 3],
    v0: [f64; 3],
    iters: usize,
    tol_km: f64,
) -> ([f64; 3], f64) {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut v = v0;
    let mut f = shoot(v);
    let (mut best_v, mut best_norm) = (v, norm(f));
    for _ in 0..iters {
        if best_norm < tol_km {
            break;
        }
        // H must clear the integrator jitter or the Jacobian is mostly noise —
        // see the sizing argument in `differential_correct`.
        const H: f64 = 1e-3;
        let mut jac = [[0.0f64; 3]; 3];
        for c in 0..3 {
            let mut vp = v;
            vp[c] += H;
            let fp = shoot(vp);
            for r in 0..3 {
                jac[r][c] = (fp[r] - f[r]) / H;
            }
        }
        let det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
            - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
            + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
        if det.abs() < 1e-12 {
            break;
        }
        let rhs = [-f[0], -f[1], -f[2]];
        let solve_col = |col: usize| {
            let mut m = jac;
            for r in 0..3 {
                m[r][col] = rhs[r];
            }
            (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                / det
        };
        let dv = [solve_col(0), solve_col(1), solve_col(2)];
        // Damp absurd steps (near-singular geometry).
        let dvn = norm(dv);
        let mut scale = if dvn > 1.0 { 1.0 / dvn } else { 1.0 };
        // Backtracking line search: halve until the step actually improves.
        let mut stepped = false;
        for _ in 0..6 {
            let mut trial = v;
            for c in 0..3 {
                trial[c] += dv[c] * scale;
            }
            let ft = shoot(trial);
            if norm(ft) < norm(f) {
                v = trial;
                f = ft;
                stepped = true;
                break;
            }
            scale *= 0.5;
        }
        if !stepped {
            break; // no improving step along this direction; best_* holds the answer
        }
        if norm(f) < best_norm {
            best_norm = norm(f);
            best_v = v;
        }
    }
    (best_v, best_norm)
}

/// Coordinate descent on the patch *schedule*: nudge the departure epoch and
/// each leg's flight time, keeping any move that improves the patched-conic
/// score. Flyby feasibility is far more sensitive to when you arrive than to
/// how you fly there, so letting the corrector move patch epochs — rather than
/// freezing them at whatever the beam happened to sample — is what makes a
/// scouted tour close up under real dynamics.
///
/// Runs on the cheap conic model (a full n-body schedule optimization would
/// cost thousands of propagations); the n-body shooting then flies the
/// improved schedule. Deterministic: fixed step ladder, fixed probe order.
pub fn optimize_patch_epochs(
    eph: &Ephemeris,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
) -> Genome {
    let bounds = cfg.leg_bounds();
    let score = |c: &Genome| evaluate_tour(eph, cfg, epoch0, c, 2).score;
    let mut best = g.clone();
    let mut best_score = score(&best);
    // Days: coarse enough to hop a bad flyby geometry, refined to sub-hour.
    let mut step = 8.0;
    for _ in 0..9 {
        let mut improved = true;
        while improved {
            improved = false;
            // Probe index 0 = departure epoch, 1..=n = each leg's TOF.
            for k in 0..=best.legs.len() {
                for sign in [1.0f64, -1.0] {
                    let mut trial = best.clone();
                    if k == 0 {
                        trial.depart_days =
                            (trial.depart_days + sign * step).clamp(0.0, cfg.window_days);
                    } else {
                        let (lo, hi) = bounds[k - 1];
                        trial.legs[k - 1] = (trial.legs[k - 1] + sign * step).clamp(lo, hi);
                    }
                    let s = score(&trial);
                    if s < best_score {
                        best_score = s;
                        best = trial;
                        improved = true;
                    }
                }
            }
        }
        step *= 0.5;
    }
    best
}

/// A tour refined to mission grade: every leg is a continuous full-fidelity
/// n-body trajectory hitting its patch body at the patch epoch.
#[derive(Clone)]
pub struct RefinedTour {
    /// Dense samples across all legs, ready to render/animate.
    pub traj: Vec<(Epoch, ScState)>,
    /// Real flyby parameters recomputed from the corrected leg velocities.
    pub flybys: Vec<Flyby>,
    pub vinf_dep_kms: f64,
    pub vinf_arr_kms: f64,
    pub assist_dv_kms: f64,
    /// Total deep-space-maneuver Δv actually flown, km/s.
    pub dsm_dv_kms: f64,
    /// Largest per-leg targeting residual after correction, km.
    pub worst_miss_km: f64,
    /// The genome as refined — the corrector may have moved the departure
    /// epoch and the leg flight times, so this is the schedule that was flown.
    pub genome: Genome,
    /// Patch epochs of the refined schedule: departure, each flyby, arrival.
    pub epochs: Vec<Epoch>,
}

/// Multi-leg shooting: differential-correct each leg under the full n-body
/// dynamics so it lands on its patch body at the patch epoch. The arrival
/// body's own gravity is excluded per leg (the flyby hyperbola inside its SOI
/// is the flyby model's job — standard patched-n-body formulation); every
/// other body pulls.
///
/// Unlike the earlier version, the patch *schedule* is not frozen: the
/// departure epoch and leg flight times are first re-optimized on the conic
/// model (see [`optimize_patch_epochs`]).
///
/// A leg carrying a deep-space maneuver is flown as one residual — coast from
/// the SOI to the maneuver point, apply the genome's burn there, coast on to
/// the node — with the *departure* velocity as the corrected variable, the
/// same parameterization [`solve_leg`] uses. The maneuver point is therefore
/// the one the real dynamics put the spacecraft at, and the Δv reported is
/// the genome's burn exactly, under n-body flight.
pub fn refine_tour(
    eph: &Ephemeris,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
) -> Option<RefinedTour> {
    let mu = BodyId::Sun.gm();
    // Let the corrector move the flyby epochs before committing to paths.
    let g = optimize_patch_epochs(eph, cfg, epoch0, g);
    let (seq, epochs, states) = tour_nodes(eph, cfg, epoch0, &g);
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    // Initial guesses: the same conic arcs (DSM included) the scout scored.
    let mut arcs: Vec<LegArc> = Vec::with_capacity(g.legs.len());
    for (i, tof) in g.legs.iter().enumerate() {
        arcs.push(solve_leg(
            states[i].pos_km,
            states[i + 1].pos_km,
            tof * DAY_S,
            mu,
            states[i].vel_km_s,
            g.leg_dsm(i),
            g.leg_branch(i),
        )?);
    }

    let dyn_cfg = DynamicsConfig {
        rel_tol: 1e-9,
        ..Default::default()
    };

    let mut traj: Vec<(Epoch, ScState)> = Vec::new();
    let mut end_vels: Vec<[f64; 3]> = Vec::new();
    let mut start_vels: Vec<[f64; 3]> = Vec::new();
    let mut worst_miss = 0.0f64;
    let mut dsm_total = 0.0f64;

    for i in 0..g.legs.len() {
        let body_from = seq[i];
        let body_to = seq[i + 1];
        // This leg feels every body except its arrival body.
        let mut leg_dyn = dyn_cfg;
        for (k, b) in crate::bodies::ALL_BODIES.iter().enumerate() {
            leg_dyn.perturbers[k] = *b != body_to;
        }
        // Launch from the start body's SOI along the leg's initial velocity
        // direction relative to the body (fixed for the whole correction, so
        // the Jacobian stays clean).
        let vrel0 = sub(arcs[i].v_start, states[i].vel_km_s);
        let vrn = norm(vrel0).max(1e-6);
        let soi = soi_km(eph, body_from, epochs[i]).min(2.0e6);
        let start_pos = [
            states[i].pos_km[0] + vrel0[0] / vrn * soi,
            states[i].pos_km[1] + vrel0[1] / vrn * soi,
            states[i].pos_km[2] + vrel0[2] / vrn * soi,
        ];
        let target_pos = states[i + 1].pos_km;

        // The corrected variable is the SOI *departure* velocity, for both
        // leg kinds — exactly as in the conic model, where the genome fixes
        // the burn and the departure is the dependent unknown. So the whole
        // leg (coast, burn, coast) is one residual, and the maneuver point is
        // wherever the n-body coast from the SOI actually puts it. The old
        // code corrected the post-burn velocity while coasting from the SOI
        // toward a maneuver point the conic seed had propagated from the body
        // *center* — two different points, so the refined burn could jump
        // even when the conic one was tiny.
        let leg_dv = arcs[i].dsm.map(|(t1, _, _)| {
            let d = g.leg_dsm(i).copied().unwrap_or([0.0; 4]);
            (Duration::from_seconds(t1), [d[1], d[2], d[3]])
        });
        let leg_tof = epochs[i + 1] - epochs[i];

        // Fly the whole leg for a trial departure velocity, at `samples`
        // density: `None` if the integrator ran out of steps anywhere, since
        // a partial endpoint says nothing about where the leg arrives.
        let fly = |v: [f64; 3], samples: usize| -> Option<(Vec<(Epoch, ScState)>, [f64; 3])> {
            let start = ScState { pos: start_pos, vel: v };
            let Some((dt1, dv)) = leg_dv else {
                let leg =
                    dynamics::propagate_checked(eph, &leg_dyn, epochs[i], start, leg_tof, samples, None);
                return leg.complete.then_some((leg.points, [0.0; 3]));
            };
            let coast =
                dynamics::propagate_checked(eph, &leg_dyn, epochs[i], start, dt1, samples, None);
            if !coast.complete {
                return None;
            }
            let (e_m, s_m) = *coast.points.last().unwrap();
            // The burn is the genome's vector, applied here and nowhere else.
            let after = ScState {
                pos: s_m.pos,
                vel: [s_m.vel[0] + dv[0], s_m.vel[1] + dv[1], s_m.vel[2] + dv[2]],
            };
            let post = dynamics::propagate_checked(
                eph,
                &leg_dyn,
                e_m,
                after,
                epochs[i + 1] - e_m,
                samples,
                None,
            );
            if !post.complete {
                return None;
            }
            let mut pts = coast.points;
            pts.extend_from_slice(&post.points);
            Some((pts, dv))
        };

        // Truncation is a hard stop for the whole correction. The flag lets
        // the shared Newton driver keep its plain residual signature.
        let truncated = std::cell::Cell::new(false);
        let shoot = |v: [f64; 3]| -> [f64; 3] {
            match fly(v, 2) {
                Some((pts, _)) => sub(pts.last().unwrap().1.pos, target_pos),
                None => {
                    truncated.set(true);
                    [0.0; 3]
                }
            }
        };
        let (v, miss) = newton_shoot(&shoot, arcs[i].v_start, 5, 300.0);
        if truncated.get() {
            return None;
        }
        worst_miss = worst_miss.max(miss);

        // Dense pass for rendering + the arrival velocity.
        let (dense, dv) = fly(v, 240)?;
        // The burn charged is the burn flown — the genome vector itself.
        dsm_total += norm(dv);
        // Report the departure velocity that was actually corrected onto the
        // node, not the conic seed it started from.
        start_vels.push(v);
        end_vels.push(dense.last().unwrap().1.vel);
        traj.extend_from_slice(&dense);
    }
    // Real flyby parameters from corrected velocities.
    let mut flybys = Vec::new();
    let mut assist_dv = 0.0;
    for k in 1..seq.len() - 1 {
        let body = seq[k];
        let vin = sub(end_vels[k - 1], states[k].vel_km_s);
        let vout = sub(start_vels[k], states[k].vel_km_s);
        let (nin, nout) = (norm(vin), norm(vout));
        let mut dv = (nin - nout).abs();
        let vinf = 0.5 * (nin + nout);
        let mu_b = body.gm();
        let rp_min = 1.1 * body.radius_km();
        let delta_max = 2.0 * (mu_b / (mu_b + rp_min * vinf * vinf)).asin();
        let cos_turn =
            ((vin[0] * vout[0] + vin[1] * vout[1] + vin[2] * vout[2]) / (nin * nout))
                .clamp(-1.0, 1.0);
        let turn = cos_turn.acos();
        let periapsis_alt_km = if turn <= delta_max && turn > 1e-6 {
            let sh = (turn / 2.0).sin();
            (mu_b * (1.0 - sh) / (sh * vinf * vinf) - body.radius_km()).max(0.0)
        } else {
            0.0
        };
        if turn > delta_max {
            dv += 2.0 * vinf * ((turn - delta_max) / 2.0).sin();
        }
        assist_dv += dv;
        flybys.push(Flyby {
            body,
            epoch: epochs[k],
            vinf_kms: vinf,
            dv_kms: dv,
            periapsis_alt_km,
        });
    }

    Some(RefinedTour {
        vinf_dep_kms: norm(sub(start_vels[0], states[0].vel_km_s)),
        vinf_arr_kms: norm(sub(
            *end_vels.last().unwrap(),
            states.last().unwrap().vel_km_s,
        )),
        assist_dv_kms: assist_dv,
        dsm_dv_kms: dsm_total,
        worst_miss_km: worst_miss,
        flybys,
        traj,
        genome: g,
        epochs,
    })
}

/// Candidate assist routes for a target: direct plus every sequence of
/// length 1–3 from an alphabet of useful assist bodies (inner planets, plus
/// Mars for Jupiter-and-beyond, plus Jupiter itself for Saturn-and-beyond).
/// Fixed enumeration order — the screening is deterministic.
pub fn candidate_routes(target: BodyId) -> Vec<Vec<BodyId>> {
    let mut alphabet = vec![BodyId::Venus, BodyId::Earth];
    let a_t = semi_major_km(target);
    if a_t > 1.3 * semi_major_km(BodyId::Earth) {
        alphabet.push(BodyId::Mars);
    }
    if a_t > 1.3 * semi_major_km(BodyId::Jupiter) {
        alphabet.push(BodyId::Jupiter);
    }
    alphabet.retain(|b| *b != target);
    let mut routes: Vec<Vec<BodyId>> = vec![Vec::new()];
    for &a in &alphabet {
        routes.push(vec![a]);
        for &b in &alphabet {
            routes.push(vec![a, b]);
            for &c in &alphabet {
                routes.push(vec![a, b, c]);
            }
        }
    }
    routes
}

pub fn route_name(route: &[BodyId]) -> String {
    if route.is_empty() {
        "direct".into()
    } else {
        route.iter().map(|b| &b.name()[..1]).collect::<Vec<_>>().join("-")
    }
}

/// Step budgets for the three phases of automatic route discovery.
#[derive(Clone, Copy)]
pub struct AutoBudget {
    /// Per-route screening steps (scaled by route length internally).
    pub screen: usize,
    /// Steps for each of the surviving routes (and their second seeds).
    pub refine: usize,
    /// Final-phase steps on the winner (usize::MAX + a live flag for GUI).
    pub polish: usize,
}

/// Automatic route discovery: screen every candidate route briefly, deepen
/// the best few, then polish the winner. Deterministic (fixed order, fixed
/// seeds). `keep_running` is polled between steps.
pub fn auto_search(
    eph: &Ephemeris,
    cfg: &SolverConfig,
    epoch0: Epoch,
    keep_running: &dyn Fn() -> bool,
    status: &mut dyn FnMut(String),
    on_best: &mut dyn FnMut(&Solution),
    budget: AutoBudget,
) -> u64 {
    let routes = candidate_routes(cfg.target);
    // One shared ephemeris table for every route. Each Search would otherwise
    // build its own (a full threaded SPK sampling pass) — ~40 of them. Sizing
    // the span to the longest route's max TOF is behaviour-preserving: the
    // sample step and t0 are unchanged, so a shorter route reads exactly the
    // same sample values, just from a longer table.
    let max_tof = routes
        .iter()
        .map(|r| {
            let mut c = cfg.clone();
            c.route = r.clone();
            c.max_total_tof_days()
        })
        .fold(0.0f64, f64::max);
    let dyn_cfg = solver_dynamics();
    let shared_eph = Arc::new(eph.cached_span(
        epoch0 - Duration::from_days(1.0),
        epoch0 + Duration::from_days(cfg.window_days + max_tof + 2.0),
        &dyn_cfg.perturbers,
    ));
    let mut evals = 0u64;
    let mut best_score = f64::INFINITY;
    let mut pool: Vec<(f64, Search)> = Vec::new();

    let drive = |search: &mut Search,
                     steps: usize,
                     local_best: &mut f64,
                     evals: &mut u64,
                     best_score: &mut f64,
                     on_best: &mut dyn FnMut(&Solution),
                     keep_running: &dyn Fn() -> bool| {
        for _ in 0..steps {
            if !keep_running() {
                return false;
            }
            let (e, (sc, g)) = search.step(eph);
            *evals += e;
            if sc < *local_best {
                *local_best = sc;
                if sc < *best_score {
                    *best_score = sc;
                    on_best(&search.solution_for(eph, &g));
                }
            }
        }
        true
    };

    for (i, route) in routes.iter().enumerate() {
        if !keep_running() {
            return evals;
        }
        status(format!(
            "screening {} ({}/{})",
            route_name(route),
            i + 1,
            routes.len()
        ));
        let mut c = cfg.clone();
        c.route = route.clone();
        c.auto_route = false;
        let mut search = Search::new(eph, c, epoch0, Some(shared_eph.clone()));
        let mut local = f64::INFINITY;
        // Longer routes have a bigger timing space — give them proportionally
        // more screening budget or multi-leg tours never show their value.
        let screen_budget = budget.screen * (1 + route.len());
        if !drive(&mut search, screen_budget, &mut local, &mut evals, &mut best_score, on_best, keep_running) {
            return evals;
        }
        pool.push((local, search));
    }

    pool.sort_by(|a, b| a.0.total_cmp(&b.0));
    pool.truncate(5);
    // Refine survivors; each also gets a fresh second-seed attempt, because
    // beam searches on rugged tour landscapes are seed-sensitive.
    let mut refined: Vec<(f64, Search)> = Vec::new();
    for (mut local, mut search) in pool {
        if !keep_running() {
            return evals;
        }
        status(format!("refining {}", route_name(&search.cfg.route)));
        if !drive(&mut search, budget.refine, &mut local, &mut evals, &mut best_score, on_best, keep_running) {
            return evals;
        }
        let mut c2 = search.cfg.clone();
        c2.seed = c2.seed.wrapping_add(1000);
        let mut alt = Search::new(eph, c2, epoch0, Some(shared_eph.clone()));
        let mut alt_local = f64::INFINITY;
        if !drive(&mut alt, budget.refine, &mut alt_local, &mut evals, &mut best_score, on_best, keep_running) {
            return evals;
        }
        if alt_local < local {
            refined.push((alt_local, alt));
        } else {
            refined.push((local, search));
        }
    }
    let mut pool = refined;

    pool.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut local, mut winner) = pool.swap_remove(0);
    status(format!("polishing {}", route_name(&winner.cfg.route)));
    drive(&mut winner, budget.polish, &mut local, &mut evals, &mut best_score, on_best, keep_running);
    evals
}

/// Re-evaluate a saved mission genome (departure baked in) with dense
/// sampling — used to restore the accepted mission after an app restart.
pub fn evaluate_saved(
    eph: &Ephemeris,
    cfg: &SolverConfig,
    depart: Epoch,
    g: &Genome,
) -> Solution {
    let dyn_cfg = solver_dynamics();
    evaluate(eph, &dyn_cfg, cfg, depart, g, 600)
}

/// The deterministic worker. Runs until `shared.running` clears.
pub fn solver_thread(
    eph: Arc<Ephemeris>,
    cfg: SolverConfig,
    epoch0: Epoch,
    prebuilt: Option<Arc<Ephemeris>>,
    shared: Arc<Shared>,
    ctx: egui::Context,
) {
    if cfg.auto_route {
        let running = || shared.running.load(Ordering::Relaxed);
        let shared2 = shared.clone();
        let ctx2 = ctx.clone();
        let mut status = |s: String| {
            *shared2.status.lock().unwrap() = s;
            ctx2.request_repaint();
        };
        let shared3 = shared.clone();
        let ctx3 = ctx.clone();
        let mut on_best = |sol: &Solution| {
            let mut best = shared3.best.lock().unwrap();
            if best.as_ref().is_none_or(|b| sol.score < b.score) {
                *best = Some(sol.clone());
                drop(best);
                ctx3.request_repaint();
            }
            shared3.steps.fetch_add(1, Ordering::Relaxed);
        };
        let evals = auto_search(
            &eph, &cfg, epoch0, &running, &mut status, &mut on_best,
            AutoBudget { screen: 120, refine: 600, polish: usize::MAX },
        );
        shared.evals.fetch_add(evals, Ordering::Relaxed);
        return;
    }
    let mut search = Search::new(&eph, cfg, epoch0, prebuilt);
    while shared.running.load(Ordering::Relaxed) {
        let (evals, (top_score, top_g)) = search.step(&eph);
        shared.evals.fetch_add(evals, Ordering::Relaxed);

        let mut best = shared.best.lock().unwrap();
        if best.as_ref().is_none_or(|b| top_score < b.score) {
            *best = Some(search.solution_for(&eph, &top_g));
            drop(best);
            ctx.request_repaint();
        }
        shared.steps.fetch_add(1, Ordering::Relaxed);
    }
}
