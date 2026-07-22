//! Spacecraft dynamics and propagation.
//!
//! Acceleration model: Newtonian point-mass attraction from Sun + planets
//! (+ Moon), plus the first post-Newtonian correction per body in the
//! Moyer/IERS "Schwarzschild" form:
//!
//!   a_pn = μ/(c²r³) · [ (4μ/r − v²) r⃗ + 4 (r⃗·v⃗) v⃗ ]
//!
//! with r⃗, v⃗ relative to each body. The Sun's term dominates and reproduces
//! Mercury's 43″/century perihelion advance; the EIH indirect/cross terms are
//! deliberately omitted (negligible for spacecraft work at this fidelity).
//!
//! Integrator: adaptive Dormand–Prince 5(4).

use crate::bodies::{BodyId, ALL_BODIES, C_KM_S};
use crate::ephemeris::Ephemeris;
use hifitime::{Duration, Epoch};

#[derive(Clone, Copy, Debug)]
pub struct ScState {
    /// Heliocentric ICRF position, km.
    pub pos: [f64; 3],
    /// km/s.
    pub vel: [f64; 3],
}

#[derive(Clone, Copy)]
pub struct DynamicsConfig {
    pub relativity: bool,
    /// Bodies contributing gravity.
    pub perturbers: [bool; ALL_BODIES.len()],
    pub rel_tol: f64,
    /// Integrator step budget. If a trajectory dives so close to a body that
    /// the adaptive step collapses, we stop rather than grind — callers see a
    /// truncated trajectory (and, in the solver, a terrible score).
    pub max_steps: usize,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            relativity: true,
            perturbers: [true; ALL_BODIES.len()],
            rel_tol: 1e-10,
            max_steps: 2_000_000,
        }
    }
}

pub fn acceleration(
    eph: &Ephemeris,
    cfg: &DynamicsConfig,
    epoch: Epoch,
    state: &ScState,
) -> [f64; 3] {
    let mut acc = [0.0f64; 3];
    for (i, body) in ALL_BODIES.iter().enumerate() {
        if !cfg.perturbers[i] {
            continue;
        }
        let bs = eph.state(*body, epoch);
        let r = [
            state.pos[0] - bs.pos_km[0],
            state.pos[1] - bs.pos_km[1],
            state.pos[2] - bs.pos_km[2],
        ];
        let v = [
            state.vel[0] - bs.vel_km_s[0],
            state.vel[1] - bs.vel_km_s[1],
            state.vel[2] - bs.vel_km_s[2],
        ];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let rn = r2.sqrt();
        if rn < 1.0 {
            continue; // inside the body; don't blow up
        }
        let mu = body.gm();
        let newton = -mu / (r2 * rn);
        for k in 0..3 {
            acc[k] += newton * r[k];
        }
        if cfg.relativity {
            let c2 = C_KM_S * C_KM_S;
            let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            let rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
            let f = mu / (c2 * r2 * rn);
            for k in 0..3 {
                acc[k] += f * ((4.0 * mu / rn - v2) * r[k] + 4.0 * rv * v[k]);
            }
        }
    }
    acc
}

/// Dormand–Prince 5(4) coefficients.
#[rustfmt::skip]
const A: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
];
const C: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
#[rustfmt::skip]
const B5: [f64; 7] = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0];
#[rustfmt::skip]
const B4: [f64; 7] = [5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0];

type Y = [f64; 6];

/// Piecewise-constant low-thrust profile. Throttle vectors are expressed in
/// the velocity frame — (along-track, orbit-normal, radial-ish) — so a
/// capture burn is simply along = −1 regardless of where the trajectory
/// points. |u| is clamped to 1 (full throttle).
pub struct Thrust<'a> {
    pub segs: &'a [[f64; 3]],
    /// Maximum thrust acceleration, km/s².
    pub accel_kms2: f64,
    /// Profile duration (the propagation span), seconds.
    pub total_s: f64,
}

fn thrust_accel(th: &Thrust, tau: f64, y: &Y) -> [f64; 3] {
    if th.segs.is_empty() || th.total_s <= 0.0 {
        return [0.0; 3];
    }
    let idx = (((tau / th.total_s) * th.segs.len() as f64) as usize).min(th.segs.len() - 1);
    let mut u = th.segs[idx];
    let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    if un < 1e-6 {
        return [0.0; 3];
    }
    if un > 1.0 {
        for c in &mut u {
            *c /= un;
        }
    }
    // Velocity frame: t̂ along heliocentric velocity, ĥ along angular
    // momentum, ŵ completes the triad.
    let (r, v) = ([y[0], y[1], y[2]], [y[3], y[4], y[5]]);
    let vn = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if vn < 1e-6 {
        return [0.0; 3];
    }
    let t_hat = [v[0] / vn, v[1] / vn, v[2] / vn];
    let h = [
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    ];
    let hn = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt().max(1e-6);
    let h_hat = [h[0] / hn, h[1] / hn, h[2] / hn];
    let w_hat = [
        t_hat[1] * h_hat[2] - t_hat[2] * h_hat[1],
        t_hat[2] * h_hat[0] - t_hat[0] * h_hat[2],
        t_hat[0] * h_hat[1] - t_hat[1] * h_hat[0],
    ];
    let a = th.accel_kms2;
    [
        a * (u[0] * t_hat[0] + u[1] * h_hat[0] + u[2] * w_hat[0]),
        a * (u[0] * t_hat[1] + u[1] * h_hat[1] + u[2] * w_hat[1]),
        a * (u[0] * t_hat[2] + u[1] * h_hat[2] + u[2] * w_hat[2]),
    ]
}

fn deriv(
    eph: &Ephemeris,
    cfg: &DynamicsConfig,
    epoch: Epoch,
    tau: f64,
    y: &Y,
    thrust: Option<&Thrust>,
) -> Y {
    let st = ScState {
        pos: [y[0], y[1], y[2]],
        vel: [y[3], y[4], y[5]],
    };
    let mut a = acceleration(eph, cfg, epoch, &st);
    if let Some(th) = thrust {
        let ta = thrust_accel(th, tau, y);
        a[0] += ta[0];
        a[1] += ta[1];
        a[2] += ta[2];
    }
    [y[3], y[4], y[5], a[0], a[1], a[2]]
}

/// Propagate from `epoch0` for `span`, returning sampled trajectory points
/// (epoch + state) at roughly `n_samples` even intervals plus the final state.
pub fn propagate(
    eph: &Ephemeris,
    cfg: &DynamicsConfig,
    epoch0: Epoch,
    state0: ScState,
    span: Duration,
    n_samples: usize,
) -> Vec<(Epoch, ScState)> {
    propagate_thrusted(eph, cfg, epoch0, state0, span, n_samples, None)
}

/// As `propagate`, with an optional continuous low-thrust profile.
pub fn propagate_thrusted(
    eph: &Ephemeris,
    cfg: &DynamicsConfig,
    epoch0: Epoch,
    state0: ScState,
    span: Duration,
    n_samples: usize,
    thrust: Option<&Thrust>,
) -> Vec<(Epoch, ScState)> {
    let total = span.to_seconds();
    let dir = total.signum();
    let total = total.abs();
    let sample_dt = (total / n_samples.max(1) as f64).max(1.0);

    let mut y: Y = [
        state0.pos[0], state0.pos[1], state0.pos[2],
        state0.vel[0], state0.vel[1], state0.vel[2],
    ];
    let mut t = 0.0f64;
    let mut h = (total / 1000.0).clamp(10.0, 86_400.0);
    let mut out = Vec::with_capacity(n_samples + 2);
    let push = |out: &mut Vec<(Epoch, ScState)>, t: f64, y: &Y| {
        out.push((
            epoch0 + Duration::from_seconds(dir * t),
            ScState {
                pos: [y[0], y[1], y[2]],
                vel: [y[3], y[4], y[5]],
            },
        ));
    };
    push(&mut out, 0.0, &y);
    let mut next_sample = sample_dt;

    let mut k = [[0.0f64; 6]; 7];
    let mut steps = 0usize;
    while t < total && steps < cfg.max_steps {
        steps += 1;
        if t + h > total {
            h = total - t;
        }
        let ep = |tau: f64| epoch0 + Duration::from_seconds(dir * tau);
        k[0] = deriv(eph, cfg, ep(t), t, &y, thrust);
        for stage in 0..6 {
            let mut ys = y;
            for j in 0..=stage {
                let a = A[stage][j];
                if a != 0.0 {
                    for m in 0..6 {
                        ys[m] += dir * h * a * k[j][m];
                    }
                }
            }
            k[stage + 1] = deriv(eph, cfg, ep(t + C[stage] * h), t + C[stage] * h, &ys, thrust);
        }
        let mut y5 = y;
        let mut y4 = y;
        for j in 0..7 {
            for m in 0..6 {
                y5[m] += dir * h * B5[j] * k[j][m];
                y4[m] += dir * h * B4[j] * k[j][m];
            }
        }
        // Error estimate, scaled.
        let mut err: f64 = 0.0;
        for m in 0..6 {
            let sc = cfg.rel_tol * y5[m].abs().max(1e3);
            err = err.max(((y5[m] - y4[m]) / sc).abs());
        }
        if err <= 1.0 {
            // Dense output: samples inside the step are cubic-Hermite
            // interpolated between the step's endpoint states — previously
            // samples were tagged with the sample epoch but carried the
            // step-end state (up to a whole step wrong), which made animated
            // playback lurch.
            let t_new = t + h;
            while next_sample <= t_new && next_sample <= total {
                let th = ((next_sample - t) / h).clamp(0.0, 1.0);
                let (t2, t3) = (th * th, th * th * th);
                let (h00, h10, h01, h11) = (
                    2.0 * t3 - 3.0 * t2 + 1.0,
                    t3 - 2.0 * t2 + th,
                    -2.0 * t3 + 3.0 * t2,
                    t3 - t2,
                );
                let mut ys: Y = [0.0; 6];
                for m in 0..3 {
                    ys[m] = h00 * y[m]
                        + h10 * h * dir * y[m + 3]
                        + h01 * y5[m]
                        + h11 * h * dir * y5[m + 3];
                    ys[m + 3] = y[m + 3] + (y5[m + 3] - y[m + 3]) * th;
                }
                push(&mut out, next_sample, &ys);
                next_sample += sample_dt;
            }
            t = t_new;
            y = y5;
        }
        let factor = if err > 0.0 {
            (0.9 * err.powf(-0.2)).clamp(0.2, 5.0)
        } else {
            5.0
        };
        h = (h * factor).clamp(1.0, 30.0 * 86_400.0);
    }
    push(&mut out, total, &y);
    out
}

/// Circular-orbit speed about the Sun at radius r (km) — handy for demo states.
pub fn circular_speed_kms(r_km: f64) -> f64 {
    (BodyId::Sun.gm() / r_km).sqrt()
}
