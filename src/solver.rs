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
}

impl Genome {
    pub fn total_tof_days(&self) -> f64 {
        self.legs.iter().sum()
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

fn evaluate(
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
    let traj = dynamics::propagate_thrusted(
        eph,
        dyn_cfg,
        depart,
        s0,
        Duration::from_seconds(total_s),
        n_samples,
        Some(&thrust),
    );
    let (arrive, sf) = *traj.last().unwrap();
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
    // Launcher C3 cap and propellant budget, charged steeply past the limit.
    let c3_over = (vinf_dep * vinf_dep - cfg.launcher.c3_max()).max(0.0);
    let prop_over = (thrust_dv - cfg.engine.max_dv_kms()).max(0.0);
    // Hardware violations are charged far above any miss penalty: a candidate
    // that "arrives" on a launcher or tank that doesn't exist must never
    // outrank a feasible candidate that still has distance to close.
    let score = cfg.objective.score(vinf_dep, arrival_dv, g.legs[0])
        + thrust_dv * 0.25 // propellant is cheaper than launch/capture Δv
        + excess_miss * 1e-6
        + miss_km * 3e-8
        + c3_over * 5.0
        + prop_over * 50.0;

    Solution {
        genome: g.clone(),
        score,
        vinf_dep_kms: vinf_dep,
        vinf_arr_kms: v_rel,
        arrival_dv_kms: arrival_dv,
        thrust_dv_kms: thrust_dv,
        assist_dv_kms: 0.0,
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
/// standard first-order tour-scouting formulation (STOUR/GALLOP class);
/// PN-dynamics refinement of a chosen tour is future work.
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
        flybys: Vec::new(),
        miss_km: f64::INFINITY,
        depart,
        arrive: *epochs.last().unwrap(),
        traj: Vec::new(),
    };

    // Solve each leg. Long legs also try one-rev solutions and keep whichever
    // conic needs the least velocity at the leg's start node (deterministic).
    let mut legs_v: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(g.legs.len());
    for (i, tof) in g.legs.iter().enumerate() {
        let (r1, r2) = (states[i].pos_km, states[i + 1].pos_km);
        let tof_s = tof * DAY_S;
        let mut options: Vec<([f64; 3], [f64; 3])> = Vec::new();
        options.extend(lambert_rev(r1, r2, tof_s, mu, 0, false));
        if *tof > 550.0 {
            options.extend(lambert_rev(r1, r2, tof_s, mu, 1, false));
            options.extend(lambert_rev(r1, r2, tof_s, mu, 1, true));
        }
        let vref = states[i].vel_km_s;
        let best = options.into_iter().min_by(|a, b| {
            let cost = |o: &([f64; 3], [f64; 3])| {
                let d = [o.0[0] - vref[0], o.0[1] - vref[1], o.0[2] - vref[2]];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            };
            cost(a).total_cmp(&cost(b))
        });
        match best {
            Some(v) => legs_v.push(v),
            None => return bad(100.0 * (i + 1) as f64),
        }
    }

    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    let vinf_dep = norm(sub(legs_v[0].0, states[0].vel_km_s));
    let vinf_arr = norm(sub(
        legs_v.last().unwrap().1,
        states.last().unwrap().vel_km_s,
    ));

    // Flyby feasibility at each intermediate body.
    let mut flybys = Vec::new();
    let mut assist_dv = 0.0;
    for k in 1..seq.len() - 1 {
        let body = seq[k];
        let vin = sub(legs_v[k - 1].1, states[k].vel_km_s);
        let vout = sub(legs_v[k].0, states[k].vel_km_s);
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
    let score = cfg.objective.score(vinf_dep, arrival_dv, total_tof) + assist_dv;

    // Sample each Lambert conic for rendering.
    let per_leg = (n_samples / g.legs.len().max(1)).max(8);
    let mut traj = Vec::with_capacity(per_leg * g.legs.len() + 1);
    for (i, tof) in g.legs.iter().enumerate() {
        let (r0, v0) = (states[i].pos_km, legs_v[i].0);
        for j in 0..per_leg {
            let dt = tof * DAY_S * j as f64 / per_leg as f64;
            let (r, v) = kepler_universal(r0, v0, dt, mu);
            traj.push((
                epochs[i] + Duration::from_seconds(dt),
                ScState { pos: r, vel: v },
            ));
        }
    }
    traj.push((
        *epochs.last().unwrap(),
        ScState {
            pos: states.last().unwrap().pos_km,
            vel: legs_v.last().unwrap().1,
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
    let traj = dynamics::propagate(
        eph,
        dyn_cfg,
        depart,
        s0,
        Duration::from_seconds(g.legs[0] * DAY_S),
        n_samples,
    );
    let (arrive, sf) = *traj.last().unwrap();
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
    let score = cfg.objective.score(vinf_dep, arrival_dv, g.legs[0])
        + excess * 1e-6
        + miss_km * 3e-8;

    Solution {
        genome: g.clone(),
        score,
        vinf_dep_kms: vinf_dep,
        vinf_arr_kms: vinf_arr,
        arrival_dv_kms: arrival_dv,
        thrust_dv_kms: 0.0,
        assist_dv_kms: 0.0,
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
}

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
        let mut options: Vec<[f64; 3]> = Vec::new();
        let mut push = |o: Option<([f64; 3], [f64; 3])>| {
            if let Some((v1, _)) = o {
                options.push([
                    v1[0] - earth.vel_km_s[0],
                    v1[1] - earth.vel_km_s[1],
                    v1[2] - earth.vel_km_s[2],
                ]);
            }
        };
        push(lambert_rev(earth.pos_km, tgt.pos_km, tof_days * DAY_S, mu, 0, false));
        if tof_days > 550.0 {
            push(lambert_rev(earth.pos_km, tgt.pos_km, tof_days * DAY_S, mu, 1, false));
            push(lambert_rev(earth.pos_km, tgt.pos_km, tof_days * DAY_S, mu, 1, true));
        }
        options.into_iter().min_by(|a, b| {
            let n = |v: &[f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            n(a).total_cmp(&n(b))
        })
    }

    fn random_genome(&mut self) -> Genome {
        let sep = self.cfg.engine != Engine::Ballistic && self.cfg.route.is_empty();
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
        Genome {
            depart_days,
            legs,
            vinf_dep,
            thrust,
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
                    // Tours: fine-nudge one leg — flyby feasibility is very
                    // sensitive to timing, so small edits matter.
                    let i = self.rng.below(g.legs.len());
                    let (lo, hi) = self.leg_bounds[i];
                    g.legs[i] = (g.legs[i] + self.rng.sym() * 4.0).clamp(lo, hi);
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
    let shoot = |vinf: [f64; 3]| -> [f64; 3] {
        let g = Genome {
            depart_days: 0.0,
            legs: vec![tof.to_seconds() / DAY_S],
            vinf_dep: vinf,
            thrust: Vec::new(),
        };
        let cfg = SolverConfig {
            route: Vec::new(),
            ..Default::default()
        };
        let sol = evaluate_direct(eph, dyn_cfg, &cfg, depart, &g, 2);
        let sf = sol.traj.last().unwrap().1;
        [
            sf.pos[0] - tgt.pos_km[0],
            sf.pos[1] - tgt.pos_km[1],
            sf.pos[2] - tgt.pos_km[2],
        ]
    };
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    let mut vinf = vinf0;
    let mut f = shoot(vinf);
    for _ in 0..9 {
        if norm(f) < 500.0 {
            break;
        }
        // Finite-difference Jacobian dF/dv∞, column per component.
        const H: f64 = 1e-4;
        let mut jac = [[0.0f64; 3]; 3];
        for c in 0..3 {
            let mut vp = vinf;
            vp[c] += H;
            let fp = shoot(vp);
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
        let scale = if dvn > 1.0 { 1.0 / dvn } else { 1.0 };
        for c in 0..3 {
            vinf[c] += dv[c] * scale;
        }
        f = shoot(vinf);
    }
    (vinf, norm(f))
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
    /// Largest per-leg targeting residual after correction, km.
    pub worst_miss_km: f64,
}

/// Multi-leg shooting: differential-correct each Lambert leg under the full
/// n-body dynamics so it lands on its patch body at the patch epoch. The
/// arrival body's own gravity is excluded per leg (the flyby hyperbola inside
/// its SOI is the flyby model's job — standard patched-n-body formulation);
/// every other body pulls. Patch positions/times stay fixed at the scouted
/// solution — this polishes the *paths*, not the schedule.
pub fn refine_tour(
    eph: &Ephemeris,
    cfg: &SolverConfig,
    epoch0: Epoch,
    g: &Genome,
) -> Option<RefinedTour> {
    let mu = BodyId::Sun.gm();
    let (seq, epochs, states) = tour_nodes(eph, cfg, epoch0, g);
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    // Initial guesses: the same Lambert arcs the scout used.
    let mut legs_v: Vec<[f64; 3]> = Vec::with_capacity(g.legs.len());
    for (i, tof) in g.legs.iter().enumerate() {
        let (r1, r2) = (states[i].pos_km, states[i + 1].pos_km);
        let tof_s = tof * DAY_S;
        let mut options: Vec<([f64; 3], [f64; 3])> = Vec::new();
        options.extend(lambert_rev(r1, r2, tof_s, mu, 0, false));
        if *tof > 550.0 {
            options.extend(lambert_rev(r1, r2, tof_s, mu, 1, false));
            options.extend(lambert_rev(r1, r2, tof_s, mu, 1, true));
        }
        let vref = states[i].vel_km_s;
        let best = options.into_iter().min_by(|a, b| {
            let cost = |o: &([f64; 3], [f64; 3])| {
                let d = sub(o.0, vref);
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            };
            cost(a).total_cmp(&cost(b))
        })?;
        legs_v.push(best.0);
    }

    let dyn_cfg = DynamicsConfig {
        rel_tol: 1e-9,
        ..Default::default()
    };

    let mut traj: Vec<(Epoch, ScState)> = Vec::new();
    let mut end_vels: Vec<[f64; 3]> = Vec::new();
    let mut start_vels: Vec<[f64; 3]> = Vec::new();
    let mut worst_miss = 0.0f64;

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
        let vrel0 = sub(legs_v[i], states[i].vel_km_s);
        let vrn = norm(vrel0).max(1e-6);
        let soi = soi_km(eph, body_from, epochs[i]).min(2.0e6);
        let start_pos = [
            states[i].pos_km[0] + vrel0[0] / vrn * soi,
            states[i].pos_km[1] + vrel0[1] / vrn * soi,
            states[i].pos_km[2] + vrel0[2] / vrn * soi,
        ];
        let tof = epochs[i + 1] - epochs[i];
        let target_pos = states[i + 1].pos_km;

        let shoot = |v: [f64; 3]| -> [f64; 3] {
            let leg = dynamics::propagate(
                eph,
                &leg_dyn,
                epochs[i],
                ScState { pos: start_pos, vel: v },
                tof,
                2,
            );
            let sf = leg.last().unwrap().1;
            sub(sf.pos, target_pos)
        };

        let mut v = legs_v[i];
        let mut f = shoot(v);
        for _ in 0..5 {
            if norm(f) < 300.0 {
                break;
            }
            const H: f64 = 1e-4;
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
            let dvn = norm(dv);
            let scale = if dvn > 1.0 { 1.0 / dvn } else { 1.0 };
            for c in 0..3 {
                v[c] += dv[c] * scale;
            }
            f = shoot(v);
        }
        worst_miss = worst_miss.max(norm(f));

        // Dense pass for rendering + the arrival velocity.
        let dense = dynamics::propagate(
            eph,
            &leg_dyn,
            epochs[i],
            ScState { pos: start_pos, vel: v },
            tof,
            240,
        );
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
        worst_miss_km: worst_miss,
        flybys,
        traj,
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
