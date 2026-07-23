//! Physics validation. Run with `cargo test --release` (the perihelion test
//! integrates a century of Mercury's orbit).

use crate::bodies::{BodyId, ALL_BODIES, AU_KM};
use crate::dynamics::{self, DynamicsConfig, ScState};
use crate::ephemeris::Ephemeris;
use hifitime::{Duration, Epoch};

fn j2000() -> Epoch {
    Epoch::from_gregorian_utc_at_midnight(2000, 1, 1)
}

/// Guard for tests that need the real SPICE kernels in `data/`.
///
/// These are the project's load-bearing validation tests. Silently returning
/// when the kernels are missing makes them PASS vacuously, which has already
/// hidden a real regression — so absence is a hard failure by default. Set
/// `ORBITAL_ALLOW_SKIP_SPICE=1` only in environments that genuinely cannot
/// carry the 32 MB kernels; the test then skips and says so.
#[must_use]
fn require_kernels(label: &str, needle: &str) -> bool {
    if label.contains(needle) {
        return true;
    }
    if std::env::var("ORBITAL_ALLOW_SKIP_SPICE").as_deref() == Ok("1") {
        eprintln!(
            "SKIPPING: kernels absent (loaded {label:?}, need {needle:?}); \
             ORBITAL_ALLOW_SKIP_SPICE=1 is set"
        );
        return false;
    }
    panic!(
        "SPICE kernels absent: ephemeris loaded as {label:?}, need {needle:?}. \
         This validation test cannot run without data/de440s.bsp. Fetch the \
         kernels, or set ORBITAL_ALLOW_SKIP_SPICE=1 to skip (see docs/VALIDATION.md)."
    );
}

#[test]
fn earth_is_about_one_au_out() {
    let eph = Ephemeris::Kepler;
    for days in [0.0, 100.0, 5000.0, -5000.0] {
        let s = eph.state(BodyId::Earth, j2000() + Duration::from_days(days));
        let r = (s.pos_km[0].powi(2) + s.pos_km[1].powi(2) + s.pos_km[2].powi(2)).sqrt() / AU_KM;
        assert!((0.97..1.03).contains(&r), "r = {r} AU at {days} d");
        let v = (s.vel_km_s[0].powi(2) + s.vel_km_s[1].powi(2) + s.vel_km_s[2].powi(2)).sqrt();
        assert!((28.0..32.0).contains(&v), "v = {v} km/s at {days} d");
    }
}

#[test]
fn energy_conserved_over_a_decade() {
    // Sun-only Newtonian two-body: specific orbital energy must hold.
    let eph = Ephemeris::Kepler;
    let mut cfg = DynamicsConfig {
        relativity: false,
        ..Default::default()
    };
    cfg.perturbers = [false; ALL_BODIES.len()];
    cfg.perturbers[0] = true; // Sun only
    let mu = BodyId::Sun.gm();
    let r0 = 1.2 * AU_KM;
    let v0 = dynamics::circular_speed_kms(r0) * 1.1;
    let s0 = ScState {
        pos: [r0, 0.0, 0.0],
        vel: [0.0, v0, 0.0],
    };
    let energy = |s: &ScState| {
        let r = (s.pos[0].powi(2) + s.pos[1].powi(2) + s.pos[2].powi(2)).sqrt();
        let v2 = s.vel[0].powi(2) + s.vel[1].powi(2) + s.vel[2].powi(2);
        v2 / 2.0 - mu / r
    };
    let traj = dynamics::propagate(&eph, &cfg, j2000(), s0, Duration::from_days(3650.0), 100);
    let e0 = energy(&traj[0].1);
    let e1 = energy(&traj.last().unwrap().1);
    assert!(
        ((e1 - e0) / e0).abs() < 1e-8,
        "energy drift {:.3e}",
        (e1 - e0) / e0
    );
}

/// The signature GR test: a Mercury-like orbit must precess ~43″/century more
/// with the 1PN term than without.
#[test]
fn mercury_perihelion_advance() {
    let eph = Ephemeris::Kepler;
    let mut cfg = DynamicsConfig::default();
    cfg.perturbers = [false; ALL_BODIES.len()];
    cfg.perturbers[0] = true; // Sun only: isolates the relativistic advance
    cfg.rel_tol = 1e-12;

    // Mercury-like: a = 0.387 AU, e = 0.2056, planar.
    let mu = BodyId::Sun.gm();
    let a = 0.387_098 * AU_KM;
    let e = 0.205_63;
    let rp = a * (1.0 - e);
    let vp = (mu * (2.0 / rp - 1.0 / a)).sqrt();
    let s0 = ScState {
        pos: [rp, 0.0, 0.0],
        vel: [0.0, vp, 0.0],
    };

    // Laplace–Runge–Lenz vector angle gives the apsidal orientation.
    let lrl_angle = |s: &ScState| -> f64 {
        let r = [s.pos[0], s.pos[1], s.pos[2]];
        let v = [s.vel[0], s.vel[1], s.vel[2]];
        let rn = (r[0].powi(2) + r[1].powi(2) + r[2].powi(2)).sqrt();
        let h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ];
        let vxh = [
            v[1] * h[2] - v[2] * h[1],
            v[2] * h[0] - v[0] * h[2],
            v[0] * h[1] - v[1] * h[0],
        ];
        let ax = vxh[0] / mu - r[0] / rn;
        let ay = vxh[1] / mu - r[1] / rn;
        ay.atan2(ax)
    };

    let century = Duration::from_days(36_525.0);
    cfg.relativity = true;
    let with_gr = dynamics::propagate(&eph, &cfg, j2000(), s0, century, 10);
    cfg.relativity = false;
    let without = dynamics::propagate(&eph, &cfg, j2000(), s0, century, 10);

    let adv_rad = lrl_angle(&with_gr.last().unwrap().1) - lrl_angle(&without.last().unwrap().1);
    let adv_arcsec = adv_rad.to_degrees() * 3600.0;
    assert!(
        (adv_arcsec - 42.98).abs() < 2.0,
        "perihelion advance = {adv_arcsec:.2}\"/century, expected ~43"
    );
}

/// Requires data/de440s.bsp; checks the fallback tracks DE440 well.
#[test]
fn spice_and_kepler_agree() {
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "de440s.bsp") {
        return;
    }
    let kepler = Ephemeris::Kepler;
    let epoch = j2000() + Duration::from_days(3652.5);
    for body in [BodyId::Earth, BodyId::Mars, BodyId::Jupiter] {
        let a = eph.state(body, epoch);
        let b = kepler.state(body, epoch);
        let d = ((a.pos_km[0] - b.pos_km[0]).powi(2)
            + (a.pos_km[1] - b.pos_km[1]).powi(2)
            + (a.pos_km[2] - b.pos_km[2]).powi(2))
        .sqrt()
            / AU_KM;
        assert!(d < 0.1, "{}: DE440 vs Kepler differ by {d} AU", body.name());
    }
}

/// The besom rule applied to the propagator: identical inputs must produce
/// bit-identical trajectories, with whatever ephemeris is actually loaded.
#[test]
fn propagation_is_bit_reproducible() {
    let (eph, _) = Ephemeris::load();
    let cfg = DynamicsConfig::default();
    let s0 = ScState {
        pos: [1.1 * AU_KM, 0.0, 0.0],
        vel: [0.0, 32.0, 3.0],
    };
    let run = || dynamics::propagate(&eph, &cfg, j2000(), s0, Duration::from_days(900.0), 500);
    let (a, b) = (run(), run());
    assert_eq!(a.len(), b.len());
    for (pa, pb) in a.iter().zip(&b) {
        assert_eq!(pa.0, pb.0);
        assert_eq!(pa.1.pos.map(f64::to_bits), pb.1.pos.map(f64::to_bits));
        assert_eq!(pa.1.vel.map(f64::to_bits), pb.1.vel.map(f64::to_bits));
    }
}

/// Every cataloged body returns a sane state: moons sit near their parent,
/// heliocentric bodies sit at plausible radii.
#[test]
fn all_bodies_have_sane_states() {
    let (eph, _) = Ephemeris::load();
    let epoch = j2000() + Duration::from_days(9000.0);
    let rmag = |p: [f64; 3]| (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
    for body in ALL_BODIES {
        let s = eph.state(body, epoch);
        if let Some(parent) = body.parent() {
            let p = eph.state(parent, epoch);
            let d = rmag([
                s.pos_km[0] - p.pos_km[0],
                s.pos_km[1] - p.pos_km[1],
                s.pos_km[2] - p.pos_km[2],
            ]);
            assert!(
                d > 1e4 && d < 3e6,
                "{} is {d:.0} km from {}", body.name(), parent.name()
            );
        } else if body != BodyId::Sun {
            let r = rmag(s.pos_km) / AU_KM;
            let (lo, hi) = match body {
                BodyId::Pluto => (29.0, 50.0),
                BodyId::Ceres | BodyId::Vesta | BodyId::Pallas | BodyId::Hygiea => (2.0, 3.6),
                _ => (0.3, 31.0),
            };
            assert!((lo..hi).contains(&r), "{} at r = {r:.2} AU", body.name());
        }
    }
}

/// The solver inherits the determinism rule: same seed + config, bit-identical
/// search trajectory. Also sanity-checks the search actually converges toward
/// its target rather than wandering.
#[test]
fn solver_is_deterministic_and_converges() {
    use crate::solver::{Search, SolverConfig};
    let (eph, _) = Ephemeris::load();
    let cfg = SolverConfig::default();
    let epoch0 = j2000() + Duration::from_days(9000.0);

    let run = || {
        let mut s = Search::new(&eph, cfg.clone(), epoch0, None);
        let mut first = f64::NAN;
        let mut last = f64::NAN;
        for i in 0..8 {
            let (_, (score, _)) = s.step(&eph);
            if i == 0 {
                first = score;
            }
            last = score;
        }
        (first, last)
    };
    let (a_first, a_last) = run();
    let (b_first, b_last) = run();
    assert_eq!(a_first.to_bits(), b_first.to_bits(), "search not deterministic");
    assert_eq!(a_last.to_bits(), b_last.to_bits(), "search not deterministic");
    assert!(a_last <= a_first, "beam search got worse: {a_first} -> {a_last}");
}

#[test]
fn bench_ephemeris_query_cost() {
    let (eph, label) = Ephemeris::load();
    let e0 = j2000() + Duration::from_days(9000.0);
    let t = std::time::Instant::now();
    let n = 2000;
    let mut acc = 0.0;
    for i in 0..n {
        let s = eph.state(BodyId::Earth, e0 + Duration::from_seconds(i as f64 * 3600.0));
        acc += s.pos_km[0];
    }
    println!(
        "{label}: {n} Earth queries in {:.2}s ({:.2} ms/query, checksum {acc:.0})",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() * 1e3 / n as f64
    );
}

#[test]
fn bench_cached_span_build() {
    let (eph, _) = Ephemeris::load();
    let t = std::time::Instant::now();
    let mask = crate::solver::solver_dynamics().perturbers;
    let start = j2000() + Duration::from_days(9000.0);
    let table = eph.cached_span(start, start + Duration::from_days(16.0 * 365.25), &mask);
    println!("16-year table built in {:.2}s", t.elapsed().as_secs_f64());
    let s = table.state(BodyId::Mars, start + Duration::from_days(500.0));
    let r = eph.state(BodyId::Mars, start + Duration::from_days(500.0));
    let d = ((s.pos_km[0]-r.pos_km[0]).powi(2)+(s.pos_km[1]-r.pos_km[1]).powi(2)+(s.pos_km[2]-r.pos_km[2]).powi(2)).sqrt();
    println!("Mars interp error {d:.3} km");
    assert!(d < 50.0);
}

#[test]
fn bench_single_eval() {
    use crate::solver::{Search, SolverConfig};
    let (eph, _) = Ephemeris::load();
    let cfg = SolverConfig::default();
    let epoch0 = j2000() + Duration::from_days(9000.0);
    let t = std::time::Instant::now();
    let mut s = Search::new(&eph, cfg.clone(), epoch0, None);
    println!("Search::new (beam seed, {} evals): {:.2}s", cfg.beam_width, t.elapsed().as_secs_f64());
    for k in 0..3 {
        let t = std::time::Instant::now();
        let _ = s.step(&eph);
        println!("step {k} ({} evals): {:.2}s", cfg.beam_width * cfg.mutations + 1, t.elapsed().as_secs_f64());
    }
}

#[test]
fn bench_propagate_profiles() {
    // Isolate where eval time goes: one long propagate at solver fidelity,
    // once with a healthy Mars-transfer v∞ and once with a near-zero v∞
    // (the pathological "loiter near Earth" case).
    let (eph, _) = Ephemeris::load();
    let dyn_cfg = crate::solver::solver_dynamics();
    let epoch0 = j2000() + Duration::from_days(9000.0);
    let span_end = epoch0 + Duration::from_days(1502.0);
    let fast = eph.cached_span(epoch0 - Duration::from_days(1.0), span_end, &dyn_cfg.perturbers);
    let earth = fast.state(BodyId::Earth, epoch0);
    for (label, vinf) in [("mars-like v∞=3.5", [3.0, 1.5, 0.5]), ("pathological v∞≈0", [0.02, 0.0, 0.0])] {
        let s0 = ScState {
            pos: [earth.pos_km[0] + 925_000.0, earth.pos_km[1], earth.pos_km[2]],
            vel: [
                earth.vel_km_s[0] + vinf[0],
                earth.vel_km_s[1] + vinf[1],
                earth.vel_km_s[2] + vinf[2],
            ],
        };
        let t = std::time::Instant::now();
        let traj = dynamics::propagate(&fast, &dyn_cfg, epoch0, s0, Duration::from_days(600.0), 40);
        let reached = (traj.last().unwrap().0 - epoch0).to_seconds() / 86400.0;
        println!("{label}: {:.2}s, reached day {reached:.0}/600, {} samples", t.elapsed().as_secs_f64(), traj.len());
    }
}

#[test]
fn bench_eval_by_candidate() {
    let (eph, _) = Ephemeris::load();
    let dyn_cfg = crate::solver::solver_dynamics();
    let epoch0 = j2000() + Duration::from_days(9000.0);
    let span_end = epoch0 + Duration::from_days(1502.0);
    let fast = eph.cached_span(epoch0 - Duration::from_days(1.0), span_end, &dyn_cfg.perturbers);
    // Same flavor of xorshift stream the search uses (seed 7).
    let mut state: u64 = 7;
    let mut next = move || {
        let mut x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut unit = move || (next() >> 11) as f64 / (1u64 << 53) as f64;
    for k in 0..12 {
        let depart_days = unit() * 900.0;
        let tof_days = 60.0 + unit() * 540.0;
        let vinf = [
            unit() * 10.0 - 5.0,
            unit() * 10.0 - 5.0,
            unit() * 4.0 - 2.0,
        ];
        let depart = epoch0 + Duration::from_seconds(depart_days * 86400.0);
        let earth = fast.state(BodyId::Earth, depart);
        let vmag = (vinf[0] * vinf[0] + vinf[1] * vinf[1] + vinf[2] * vinf[2])
            .sqrt()
            .max(1e-6);
        let s0 = ScState {
            pos: [
                earth.pos_km[0] + vinf[0] / vmag * 925_000.0,
                earth.pos_km[1] + vinf[1] / vmag * 925_000.0,
                earth.pos_km[2] + vinf[2] / vmag * 925_000.0,
            ],
            vel: [
                earth.vel_km_s[0] + vinf[0],
                earth.vel_km_s[1] + vinf[1],
                earth.vel_km_s[2] + vinf[2],
            ],
        };
        let t = std::time::Instant::now();
        let traj = dynamics::propagate(
            &fast,
            &dyn_cfg,
            depart,
            s0,
            Duration::from_seconds(tof_days * 86400.0),
            40,
        );
        println!(
            "cand {k}: depart {depart_days:5.0}d tof {tof_days:4.0}d vinf [{:+.1} {:+.1} {:+.1}] -> {:.2}s ({} pts)",
            vinf[0], vinf[1], vinf[2],
            t.elapsed().as_secs_f64(),
            traj.len()
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Lambert solver sanity: the returned conic must conserve energy and
/// angular momentum between its endpoints, and a near-Hohmann geometry
/// (178° transfer at half-ellipse TOF) must come out near the analytic
/// Hohmann speeds.
#[test]
fn lambert_conserves_and_matches_hohmann() {
    use crate::solver::lambert;
    let mu = BodyId::Sun.gm();
    let r1v = [1.0 * AU_KM, 0.0, 0.0];

    // General geometry: 120°, 200 days.
    let th = 120f64.to_radians();
    let r2n_g = 1.4 * AU_KM;
    let r2v = [r2n_g * th.cos(), r2n_g * th.sin(), 0.0];
    let (v1, v2) = lambert(r1v, r2v, 200.0 * 86_400.0, mu).expect("lambert 120°");
    let energy = |r: [f64; 3], v: [f64; 3]| {
        let rn = (r[0].powi(2) + r[1].powi(2) + r[2].powi(2)).sqrt();
        (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)) / 2.0 - mu / rn
    };
    let hz = |r: [f64; 3], v: [f64; 3]| r[0] * v[1] - r[1] * v[0];
    let e1 = energy(r1v, v1);
    let e2 = energy(r2v, v2);
    assert!(((e1 - e2) / e1).abs() < 1e-9, "energy mismatch {e1} vs {e2}");
    assert!(
        ((hz(r1v, v1) - hz(r2v, v2)) / hz(r1v, v1)).abs() < 1e-9,
        "ang. momentum mismatch"
    );

    // Near-Hohmann: 178°, half-ellipse TOF, speeds ≈ vis-viva at both ends.
    let r2n = 1.523_7 * AU_KM;
    let a = 0.5 * (1.0 * AU_KM + r2n);
    let tof = std::f64::consts::PI * (a * a * a / mu).sqrt();
    let th = 178f64.to_radians();
    let (v1, v2) = lambert(r1v, [r2n * th.cos(), r2n * th.sin(), 0.0], tof, mu)
        .expect("lambert 178°");
    let v1n = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
    let v2n = (v2[0].powi(2) + v2[1].powi(2) + v2[2].powi(2)).sqrt();
    let vis_viva = |r: f64| (mu * (2.0 / r - 1.0 / a)).sqrt();
    assert!((v1n - vis_viva(1.0 * AU_KM)).abs() < 0.3, "v1 {v1n} vs {}", vis_viva(1.0 * AU_KM));
    assert!((v2n - vis_viva(r2n)).abs() < 0.3, "v2 {v2n} vs {}", vis_viva(r2n));
}

/// The headline validation: searching the 2020 window must rediscover the
/// Mars 2020 (Perseverance) trajectory — launched 2020-07-30, TOF 203 d,
/// C3 = 14.49 km²/s². We accept the window to within a few days and ~15% C3.
#[test]
fn rediscovers_mars_2020_trajectory() {
    use crate::solver::{Search, SolverConfig};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let cfg = SolverConfig::default();
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2020, 1, 1);
    let mut s = Search::new(&eph, cfg, epoch0, None);
    let mut top = None;
    for _ in 0..150 {
        let (_, t) = s.step(&eph);
        top = Some(t);
    }
    let (_, g) = top.unwrap();
    let sol = s.solution_for(&eph, &g);
    let (y, m, d, ..) = sol.depart.to_gregorian_utc();
    let c3 = sol.vinf_dep_kms * sol.vinf_dep_kms;
    println!(
        "found depart {y:04}-{m:02}-{d:02}, TOF {:.0} d, C3 {c3:.1} km2/s2, arr v∞ {:.2}",
        g.total_tof_days(), sol.vinf_arr_kms
    );
    assert_eq!(y, 2020);
    assert!(m == 7 || m == 8, "wrong window: month {m}");
    assert!((150.0..260.0).contains(&g.total_tof_days()), "TOF {}", g.total_tof_days());
    assert!((11.0..18.0).contains(&c3), "C3 {c3}");
}

/// Fresh ballistic seeds must not be sampled *conditional* on low C3: an
/// explicit uniform channel has to survive alongside the launch-feasible one,
/// or timing regions with good arrival v∞ / short TOF but moderate C3 become
/// unreachable (see issue #12). Also checks the feasible-draw rate is recorded
/// rather than the distribution being silently replaced.
#[test]
fn fresh_seeds_keep_a_uniform_timing_channel() {
    use crate::solver::{Search, SolverConfig};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let cfg = SolverConfig::default();
    let cap = cfg.launcher.c3_max();
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2020, 1, 1);
    let s = Search::new(&eph, cfg, epoch0, None);
    let stats = s.seed_draw_stats();
    println!("{stats:?} feasible rate {:.3}", stats.feasible_rate());

    // The rate is measured over untouched uniform draws, so it must be well
    // under 1 — Lambert aims at arbitrary (departure, TOF) pairs mostly bust
    // any real launcher cap.
    assert!(stats.draws > 0, "no timing draws recorded");
    assert!(stats.feasible_rate() < 0.75, "rate {}", stats.feasible_rate());
    assert!(stats.uniform_genomes > 0, "uniform seed channel is empty");
    assert!(stats.feasible_genomes > 0, "feasible seed channel is empty");

    // ...and the uniform channel must actually reach past the cap in the beam.
    let over = s
        .beam_genomes()
        .iter()
        .filter(|g| {
            let v = g.vinf_dep;
            v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > cap
        })
        .count();
    assert!(over > 0, "every fresh seed landed under the C3 cap: {over}");
}

/// RNG consumption during seeding must not depend on Lambert reachability:
/// the feasible-channel batch is always drawn in full, so the number of timing
/// draws is a pure function of the config (docs/DETERMINISM.md).
#[test]
fn seed_rng_consumption_is_data_independent() {
    use crate::solver::{Search, SolverConfig, SEED_DRAWS};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2020, 1, 1);
    let mut expected = None;
    // Very different launcher caps => very different feasible-draw rates; the
    // draw *count* must be identical either way.
    for launcher in crate::solver::Launcher::ALL {
        let mut cfg = SolverConfig::default();
        cfg.launcher = launcher;
        let width = cfg.beam_width;
        let s = Search::new(&eph, cfg, epoch0, None);
        let draws = s.seed_draw_stats().draws;
        assert_eq!(draws, width * SEED_DRAWS, "{}", launcher.label());
        if let Some(e) = expected {
            assert_eq!(draws, e, "RNG consumption shifted with launcher cap");
        }
        expected = Some(draws);
    }
}

/// Universal-variable Kepler propagation must round-trip: forward dt then
/// backward dt returns the initial state.
#[test]
fn kepler_universal_round_trip() {
    use crate::solver::kepler_universal;
    let mu = BodyId::Sun.gm();
    let r0 = [1.1 * AU_KM, 0.2 * AU_KM, 0.05 * AU_KM];
    let v0 = [-5.0, 28.0, 1.0];
    let dt = 250.0 * 86_400.0;
    let (r1, v1) = kepler_universal(r0, v0, dt, mu);
    let (r2, v2) = kepler_universal(r1, v1, -dt, mu);
    for k in 0..3 {
        assert!((r2[k] - r0[k]).abs() < 1.0, "pos {k}: {} vs {}", r2[k], r0[k]);
        assert!((v2[k] - v0[k]).abs() < 1e-6, "vel {k}");
    }
}

/// Multi-rev Lambert: adding one transfer-ellipse period to the TOF and
/// requesting revs=1 must recover (nearly) the same conic as the single-rev
/// solution — it is the same ellipse flown once more around.
#[test]
fn lambert_multirev_recovers_same_ellipse() {
    use crate::solver::{lambert, lambert_rev};
    let mu = BodyId::Sun.gm();
    let r1v = [1.0 * AU_KM, 0.0, 0.0];
    let th = 120f64.to_radians();
    let r2v = [1.4 * AU_KM * th.cos(), 1.4 * AU_KM * th.sin(), 0.0];
    let tof = 200.0 * 86_400.0;
    let (v1, _) = lambert(r1v, r2v, tof, mu).expect("single rev");
    // Period of that transfer ellipse.
    let r1n = 1.0 * AU_KM;
    let v1n2 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    let a = 1.0 / (2.0 / r1n - v1n2 / mu);
    let period = 2.0 * std::f64::consts::PI * (a * a * a / mu).sqrt();
    // The m=1 TOF curve is U-shaped: two conics solve it, and the re-flown
    // original ellipse is one of them — on whichever branch. Accept either.
    let candidates: Vec<[f64; 3]> = [false, true]
        .iter()
        .filter_map(|hb| lambert_rev(r1v, r2v, tof + period, mu, 1, *hb).map(|(w1, _)| w1))
        .collect();
    assert!(!candidates.is_empty(), "no multi-rev solution");
    let matched = candidates.iter().any(|w1| {
        (0..3).all(|k| (w1[k] - v1[k]).abs() < 0.05)
    });
    assert!(matched, "no branch recovered the original ellipse: {candidates:?} vs {v1:?}");
}

/// `lambert_best` must actually find multi-revolution transfers, not just the
/// direct arc. Construct a known m-rev transfer by flying a real ellipse: take
/// the single-rev solution's conic and ask for the *same* endpoints m extra
/// periods later. A solver that only ever tries revs=0 cannot solve that TOF
/// at all (the geometry needs > 2π of sweep), so recovering the ellipse for
/// every m is proof the sweep is real.
#[test]
fn lambert_best_finds_known_multirev_transfers() {
    use crate::solver::{lambert, lambert_best};
    let mu = BodyId::Sun.gm();
    let r1v = [1.0 * AU_KM, 0.0, 0.0];
    let th = 120f64.to_radians();
    let r2v = [1.4 * AU_KM * th.cos(), 1.4 * AU_KM * th.sin(), 0.0];
    let tof = 200.0 * 86_400.0;
    let (v1, _) = lambert(r1v, r2v, tof, mu).expect("single rev");
    let v1n2 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    let a = 1.0 / (2.0 / AU_KM - v1n2 / mu);
    let period = 2.0 * std::f64::consts::PI * (a * a * a / mu).sqrt();

    // The known ellipse leaves r1 with exactly v1. Ask for it back by making
    // v1 the reference velocity: only an m-rev sweep can return it, because
    // the m-rev arc *is* this ellipse flown m extra times around.
    for m in 1..=4 {
        let t = tof + m as f64 * period;
        let (w1, _) = lambert_best(r1v, r2v, t, mu, v1)
            .unwrap_or_else(|| panic!("m={m}: lambert_best found no arc"));
        for k in 0..3 {
            assert!(
                (w1[k] - v1[k]).abs() < 0.05,
                "m={m}: did not recover the known ellipse on axis {k}: {} vs {}",
                w1[k],
                v1[k]
            );
        }
        // The arc must be a real conic through both endpoints.
        let (rf, _) = crate::solver::kepler_universal(r1v, w1, t, mu);
        for k in 0..3 {
            assert!(
                (rf[k] - r2v[k]).abs() < 1e5,
                "m={m}: arc misses the arrival point on axis {k}: {} vs {}",
                rf[k],
                r2v[k]
            );
        }
        // Sanity that this is genuinely the multi-rev branch and not the
        // single-rev arc in disguise: the 0-rev solution for the same TOF is
        // a much slower, much bigger ellipse.
        let (v0, _) = crate::solver::lambert_rev(r1v, r2v, t, mu, 0, false)
            .expect("0-rev arc exists too");
        let sma = |v: [f64; 3]| {
            1.0 / (2.0 / AU_KM - (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / mu)
        };
        assert!(
            sma(v0) > 1.2 * sma(w1),
            "m={m}: 0-rev arc was not distinguishably larger ({} vs {})",
            sma(v0),
            sma(w1)
        );
    }
}

/// The revolution ceiling must come from geometry, not a hardcoded TOF: a leg
/// far too short for even one extra revolution admits none, while a very long
/// one admits several.
#[test]
fn multirev_ceiling_scales_with_time_of_flight() {
    use crate::solver::lambert_best;
    let mu = BodyId::Sun.gm();
    let r1v = [1.0 * AU_KM, 0.0, 0.0];
    let r2v = [0.0, 1.2 * AU_KM, 0.0];
    // 100 days is well under one transfer period — only the direct arc exists.
    let short = lambert_best(r1v, r2v, 100.0 * 86_400.0, mu, [0.0; 3]);
    assert!(short.is_some(), "direct arc should still solve");
    // The old hardcoded gate was tof > 550 d. A 500-day leg is below it but
    // is more than a transfer period, so multi-rev arcs do exist there — this
    // is exactly what the magic constant used to hide.
    let mid = 500.0 * 86_400.0;
    let multi_exists = (1..=4).any(|m| {
        [false, true]
            .iter()
            .any(|hb| crate::solver::lambert_rev(r1v, r2v, mid, mu, m, *hb).is_some())
    });
    assert!(multi_exists, "no multi-rev arc below the old 550-day gate");
    // And lambert_best must be picking the cheapest of everything available:
    // never worse (w.r.t. its vref cost) than the plain single-rev arc.
    let vref = [0.0, 30.0, 0.0];
    let best = lambert_best(r1v, r2v, mid, mu, vref).expect("some arc");
    let cost = |v: [f64; 3]| {
        (0..3).map(|k| (v[k] - vref[k]).powi(2)).sum::<f64>()
    };
    if let Some((v0, _)) = crate::solver::lambert_rev(r1v, r2v, mid, mu, 0, false) {
        assert!(
            cost(best.0) <= cost(v0) + 1e-9,
            "lambert_best returned a worse arc than plain single-rev"
        );
    }
}

/// A DSM-enabled tour must score no worse than the same tour flown ballistic.
/// The parameterization is built so an inert node (zero Δv) is bit-for-bit the
/// ballistic leg, and the search only ever turns one on if it pays — so the
/// beam with DSMs available must never end up behind the beam without them.
#[test]
fn dsm_tour_scores_no_worse_than_ballistic() {
    use crate::solver::{Genome, Search, SolverConfig};
    let (eph, _) = Ephemeris::load();
    let mut cfg = SolverConfig::default();
    cfg.target = BodyId::Jupiter;
    cfg.route = vec![BodyId::Venus, BodyId::Earth];
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2028, 1, 1);

    let run = |strip: bool| -> (f64, Genome) {
        let mut s = Search::new(&eph, cfg.clone(), epoch0, None);
        let mut top = None;
        for _ in 0..60 {
            let (_, t) = s.step(&eph);
            top = Some(t);
        }
        let (score, mut g) = top.unwrap();
        if strip {
            // Score the very same schedule with the maneuvers removed.
            g.dsm.clear();
            return (s.solution_for(&eph, &g).score, g);
        }
        (score, g)
    };

    let (with_dsm, g) = run(false);
    assert!(with_dsm < 1e3, "no feasible tour found, score {with_dsm}");

    // 1. An inert DSM node is exactly the ballistic leg — no drift, no cost.
    let mut inert = g.clone();
    inert.dsm = vec![[0.5, 0.0, 0.0, 0.0]; inert.legs.len()];
    let mut bare = g.clone();
    bare.dsm.clear();
    let s_inert = crate::solver::evaluate_saved(&eph, &cfg, epoch0, &inert).score;
    let s_bare = crate::solver::evaluate_saved(&eph, &cfg, epoch0, &bare).score;
    assert_eq!(
        s_inert, s_bare,
        "an inert DSM node changed the score ({s_inert} vs {s_bare})"
    );

    // 2. Whatever the search settled on, it beats the same schedule stripped
    //    of its maneuvers — a DSM is only ever kept when it pays for itself.
    let stripped = crate::solver::evaluate_saved(&eph, &cfg, epoch0, &bare).score;
    assert!(
        with_dsm <= stripped + 1e-6,
        "DSM tour scored worse than ballistic: {with_dsm} vs {stripped}"
    );

    // 3. The DSM Δv the solution reports is real and charged, not free.
    let sol = crate::solver::evaluate_saved(&eph, &cfg, epoch0, &g);
    assert!(sol.dsm_dv_kms >= 0.0 && sol.dsm_dv_kms < 20.0, "DSM Δv {}", sol.dsm_dv_kms);
}

/// The corrector must be allowed to move patch epochs, and doing so must not
/// make the tour worse under the conic model it optimizes against.
#[test]
fn patch_epoch_optimization_improves_schedule() {
    use crate::solver::{optimize_patch_epochs, Search, SolverConfig};
    let (eph, _) = Ephemeris::load();
    let mut cfg = SolverConfig::default();
    cfg.target = BodyId::Mars;
    cfg.route = vec![BodyId::Venus];
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2028, 1, 1);
    let mut s = Search::new(&eph, cfg.clone(), epoch0, None);
    let mut top = None;
    for _ in 0..30 {
        let (_, t) = s.step(&eph);
        top = Some(t);
    }
    let (before, g) = top.unwrap();
    let tuned = optimize_patch_epochs(&eph, &cfg, epoch0, &g);
    let after = crate::solver::evaluate_saved(&eph, &cfg, epoch0, &tuned).score;
    assert!(after <= before + 1e-9, "schedule tuning made it worse: {before} -> {after}");
    // Epochs are genuinely free now — the tuner should have moved *something*
    // off a randomly-sampled beam schedule.
    let moved = (tuned.depart_days - g.depart_days).abs() > 1e-9
        || tuned
            .legs
            .iter()
            .zip(&g.legs)
            .any(|(a, b)| (a - b).abs() > 1e-9);
    assert!(moved, "patch epochs stayed frozen");
}

/// Tour evaluation smoke: a VEEGA search must produce feasible Lambert legs
/// (score ≪ the failure sentinel) and one Flyby record per route body.
#[test]
fn veega_tour_search_is_sane() {
    use crate::solver::{Search, SolverConfig};
    let (eph, _) = Ephemeris::load();
    let mut cfg = SolverConfig::default();
    cfg.target = BodyId::Jupiter;
    cfg.route = vec![BodyId::Venus, BodyId::Earth, BodyId::Earth];
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2028, 1, 1);
    let mut s = Search::new(&eph, cfg, epoch0, None);
    let mut top = None;
    for _ in 0..40 {
        let (_, t) = s.step(&eph);
        top = Some(t);
    }
    let (score, g) = top.unwrap();
    let sol = s.solution_for(&eph, &g);
    assert!(score < 1e3, "no feasible tour found, score {score}");
    assert_eq!(sol.flybys.len(), 3);
    assert!(sol.vinf_dep_kms > 0.5 && sol.vinf_dep_kms < 15.0);
    // Patched-conic legs land exactly on the target by construction.
    assert_eq!(sol.miss_km, 0.0);
    // Issue #16: every flyby must sit at or above the body-safe periapsis
    // floor — the scout is no longer allowed to buy an unphysical grazing or
    // sub-surface pass for free.
    for f in &sol.flybys {
        let floor_alt = crate::solver::min_flyby_periapsis_km(f.body) - f.body.radius_km();
        assert!(
            f.periapsis_alt_km >= floor_alt - 1.0,
            "{} flyby periapsis {:.0} km below safe floor {:.0} km",
            f.body.name(),
            f.periapsis_alt_km,
            floor_alt
        );
    }
}

/// Issue #21: the J2 oblateness term must match the closed form, act only
/// near a body, and vanish when disabled. Earth's pole is ~ICRF +Z, giving a
/// clean geometry to check: in the equatorial plane J2 pushes inward and
/// slightly toward the equator; on the spin axis it pushes inward, doubled.
#[test]
fn j2_oblateness_matches_closed_form() {
    let eph = Ephemeris::Kepler;
    let epoch = j2000();
    let mut cfg = DynamicsConfig {
        relativity: false,
        oblateness: true,
        ..Default::default()
    };
    // Earth only, so we isolate its J2 (and its point mass).
    cfg.perturbers = [false; ALL_BODIES.len()];
    let earth_idx = ALL_BODIES.iter().position(|b| *b == BodyId::Earth).unwrap();
    cfg.perturbers[earth_idx] = true;
    let e = eph.state(BodyId::Earth, epoch);
    let (j2, r_eq, _, _) = BodyId::Earth.j2_params().unwrap();
    let mu = BodyId::Earth.gm();
    // A point 3 equatorial radii out, in the equatorial plane (+x from Earth).
    let r = 3.0 * r_eq;
    let sc = ScState { pos: [e.pos_km[0] + r, e.pos_km[1], e.pos_km[2]], vel: [0.0; 3] };
    let a = dynamics::acceleration(&eph, &cfg, epoch, &sc);
    // Subtract the point-mass part to isolate J2 (both along -x here).
    let a_point = -mu / (r * r);
    let a_j2_x = a[0] - a_point;
    // Closed form on the equator (u = r̂·p̂ = 0): a_J2 = -(3/2)J2 μ R²/r⁴ · r̂.
    let expect = -1.5 * j2 * mu * r_eq * r_eq / r.powi(4);
    assert!(
        (a_j2_x - expect).abs() < 1e-6 * expect.abs(),
        "equatorial J2 {a_j2_x:e} vs {expect:e}"
    );
    // On the spin axis (+z), u = 1: the z-coefficient is (3 - 5u²) = -2, so
    // a_J2_z = k·u·(3-5u²) = -2k = -2·expect — outward, twice the equatorial
    // magnitude (Vallado: the K-component carries 3-5(z/r)², not 1-5(z/r)²).
    let sc_pole = ScState { pos: [e.pos_km[0], e.pos_km[1], e.pos_km[2] + r], vel: [0.0; 3] };
    let ap = dynamics::acceleration(&eph, &cfg, epoch, &sc_pole);
    let a_j2_z = ap[2] - (-mu / (r * r));
    assert!((a_j2_z + 2.0 * expect).abs() < 1e-6 * expect.abs(), "polar J2 {a_j2_z:e}");
    // Far away (200 R_eq > the 100 R_eq gate) the J2 term is not applied:
    // the residual over point mass is far below the equatorial J2 signal.
    let far = 200.0 * r_eq;
    let sc_far = ScState { pos: [e.pos_km[0] + far, e.pos_km[1], e.pos_km[2]], vel: [0.0; 3] };
    let af = dynamics::acceleration(&eph, &cfg, epoch, &sc_far);
    let far_point = -mu / (far * far);
    assert!(
        (af[0] - far_point).abs() < 1e-9 * far_point.abs(),
        "J2 leaked past the gate"
    );
    // Disabled: pure point mass (bit-unchanged path — only rounding differs).
    cfg.oblateness = false;
    let a_off = dynamics::acceleration(&eph, &cfg, epoch, &sc);
    assert!(
        (a_off[0] - a_point).abs() < 1e-12 * a_point.abs(),
        "J2 applied with flag off"
    );
}

/// Issue #19: solar radiation pressure must push directly away from the Sun
/// with the analytic 1/r² cannonball magnitude, and be off when disabled.
#[test]
fn srp_pushes_away_from_sun_with_inverse_square_magnitude() {
    let eph = Ephemeris::Kepler;
    let epoch = j2000();
    // Isolate SRP: no gravity, no relativity.
    let mut cfg = DynamicsConfig {
        relativity: false,
        srp_cr_area_mass: 1.3 * 0.015,
        ..Default::default()
    };
    cfg.perturbers = [false; ALL_BODIES.len()];
    // Spacecraft 1 AU from the Sun along +x (states are heliocentric).
    let at = |r_km: f64| ScState { pos: [r_km, 0.0, 0.0], vel: [0.0; 3] };
    let a1 = dynamics::acceleration(&eph, &cfg, epoch, &at(AU_KM));
    // Analytic: Cr·(A/m)·P_1AU·(AU/r)² ×1e-3, along +x, away from Sun.
    let expect = 1.3 * 0.015 * 4.5344e-6 * 1e-3;
    assert!((a1[0] - expect).abs() < 1e-3 * expect, "SRP mag {} vs {expect}", a1[0]);
    assert!(a1[0] > 0.0, "SRP must push away from the Sun (+x)");
    assert!(a1[1].abs() < 1e-14 && a1[2].abs() < 1e-14, "SRP not radial");
    // Inverse-square: at 2 AU the magnitude is a quarter.
    let a2 = dynamics::acceleration(&eph, &cfg, epoch, &at(2.0 * AU_KM));
    assert!((a2[0] - a1[0] / 4.0).abs() < 1e-3 * a1[0], "not 1/r²: {} vs {}", a2[0], a1[0] / 4.0);
    // Disabled by default.
    cfg.srp_cr_area_mass = 0.0;
    let off = dynamics::acceleration(&eph, &cfg, epoch, &at(AU_KM));
    assert_eq!(off, [0.0; 3], "SRP leaked with coefficient zero");
}

/// Issue #18: capture Δv must carry a finite-burn margin that grows with burn
/// size, so a high-v∞ arrival is charged more than the impulsive ideal and
/// isn't preferred only because the loss was omitted.
#[test]
fn capture_dv_carries_finite_burn_margin() {
    use crate::solver::{arrival_dv_kms, MissionType};
    // Impulsive ideal for a Mars orbit capture (1.5 R, e=0.95), by hand.
    let mu = BodyId::Mars.gm();
    let rp = 1.5 * BodyId::Mars.radius_km();
    let vinf = 2.6;
    let ideal = (vinf * vinf + 2.0 * mu / rp).sqrt() - (mu / rp * 1.95).sqrt();
    let charged = arrival_dv_kms(BodyId::Mars, vinf, MissionType::Orbit);
    // Charged strictly more than ideal, within the 3–15% margin band.
    assert!(charged > ideal * 1.03 - 1e-9, "no margin: {charged} vs {ideal}");
    assert!(charged < ideal * 1.16, "margin too large: {charged} vs {ideal}");
    // The margin fraction grows with burn size: a fast (high-v∞) arrival is
    // charged a larger *fraction* than a gentle one.
    let gentle = 1.0;
    let fast = 5.0;
    let frac = |vinf: f64| {
        let ideal = (vinf * vinf + 2.0 * mu / rp).sqrt() - (mu / rp * 1.95).sqrt();
        arrival_dv_kms(BodyId::Mars, vinf, MissionType::Orbit) / ideal - 1.0
    };
    assert!(frac(fast) > frac(gentle), "margin not monotone in burn size");
}

/// Issue #17: a high-declination launch asymptote must derate the usable C3,
/// while an in-plane (low-DLA) departure keeps the full cap. The derate only
/// bites trajectories pushing near the launcher's capability.
#[test]
fn high_dla_departure_loses_c3_capability() {
    use crate::solver::{dla_c3_derate, Launcher};
    // In-ecliptic-ish, low-declination v∞: no penalty.
    let low = dla_c3_derate([5.0, 0.3, 0.0]);
    assert!((low - 1.0).abs() < 1e-9, "low-DLA derate {low}");
    // Same magnitude but thrown far out of plane: strictly derated.
    let high = dla_c3_derate([0.0, 0.0, 5.0]);
    assert!(high < 0.6 && high > 0.3, "polar-DLA derate {high}");
    // A departure just under the raw Falcon Heavy cap becomes infeasible once
    // it is thrown to a polar asymptote (the derated cap falls below it).
    let raw_cap = Launcher::FalconHeavy.c3_max().sqrt(); // ~7.75 km/s
    let vmag = raw_cap * 0.95;
    let polar = [0.0, 0.0, vmag];
    let derated_cap = (Launcher::FalconHeavy.c3_max() * dla_c3_derate(polar)).sqrt();
    assert!(
        vmag > derated_cap,
        "polar {vmag:.2} should exceed derated cap {derated_cap:.2}"
    );
}

/// Issue #16: the safe-periapsis floor must respect each body's hazard —
/// atmospheric bodies cleared above their air, gas giants well above cloud
/// tops, airless bodies only grazed.
#[test]
fn flyby_periapsis_floor_respects_body_hazards() {
    let alt = |b: BodyId| crate::solver::min_flyby_periapsis_km(b) - b.radius_km();
    // Atmospheric: a couple hundred km of clearance, not a graze.
    assert!(alt(BodyId::Earth) >= 150.0, "Earth floor {:.0} km", alt(BodyId::Earth));
    assert!(alt(BodyId::Venus) >= 150.0);
    assert!(alt(BodyId::Mars) >= 150.0);
    // Gas giant: floor is a large fraction of the (huge) radius.
    assert!(
        crate::solver::min_flyby_periapsis_km(BodyId::Jupiter) > 1.2 * BodyId::Jupiter.radius_km()
    );
    // Airless: low but strictly above the surface.
    let moon = alt(BodyId::Moon);
    assert!(moon > 0.0 && moon < 0.2 * BodyId::Moon.radius_km(), "Moon floor {moon:.0} km");
}

/// Two-phase pipeline: beam search scouts to ~1e5 km, then the differential
/// corrector must drive the full-fidelity miss to km-scale.
#[test]
fn differential_correction_hits_target() {
    use crate::solver::{differential_correct, Search, SolverConfig};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let cfg = SolverConfig::default();
    let epoch0 = j2000() + Duration::from_days(9000.0);
    let mut s = Search::new(&eph, cfg.clone(), epoch0, None);
    let mut top = None;
    for _ in 0..60 {
        let (_, t) = s.step(&eph);
        top = Some(t);
    }
    let (_, g) = top.unwrap();
    let sol = s.solution_for(&eph, &g);
    let full = crate::solver::solver_dynamics(); // full catalog, search tol
    let (_, miss) = differential_correct(
        &eph,
        &full,
        sol.depart,
        Duration::from_days(g.total_tof_days()),
        cfg.target,
        g.vinf_dep,
    );
    assert!(
        miss < 5_000.0,
        "corrector left {miss:.0} km miss (started ~{:.0})",
        sol.miss_km
    );
}

/// Multi-leg shooting: a scouted Venus-assist tour to Mars must refine into
/// continuous n-body legs that each hit their patch body to km-scale.
#[test]
fn tour_refines_to_mission_grade() {
    use crate::solver::{refine_tour, Search, SolverConfig};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let mut cfg = SolverConfig::default();
    cfg.target = BodyId::Mars;
    cfg.route = vec![BodyId::Venus];
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2028, 1, 1);
    let mut s = Search::new(&eph, cfg.clone(), epoch0, None);
    let mut top = None;
    for _ in 0..60 {
        let (_, t) = s.step(&eph);
        top = Some(t);
    }
    let (score, g) = top.unwrap();
    assert!(score < 1e3, "no feasible scout tour, score {score}");
    let rt = refine_tour(&eph, &cfg, epoch0, &g).expect("refinement failed");
    assert!(
        rt.worst_miss_km < 5_000.0,
        "worst leg miss {:.0} km after shooting",
        rt.worst_miss_km
    );
    assert_eq!(rt.flybys.len(), 1);
    assert!(rt.traj.len() > 400, "dense trajectory expected");
    // Issue #16: the final leg is B-plane-targeted onto the mission periapsis,
    // the same fidelity a direct transfer gets — a real captured periapsis and
    // a real insertion Δv, not the idealized center-targeted 1.5·R capture.
    let berr = rt.arrival_bplane_err_km.expect("arrival B-plane targeted");
    assert!(berr < 50.0, "arrival B-plane error {berr:.1} km");
    let alt = rt.arrival_periapsis_alt_km.expect("arrival periapsis");
    let want = 0.5 * BodyId::Mars.radius_km(); // 1.5 R orbit ⇒ 0.5 R altitude
    assert!(
        (alt - want).abs() < 50.0,
        "arrival periapsis alt {alt:.0} km, wanted ~{want:.0}"
    );
    let dv = rt.arrival_dv_kms.expect("arrival insertion Δv");
    assert!(dv > 0.3 && dv < 5.0, "arrival insertion Δv {dv:.2} km/s");
}

/// GPU porkchop must run headless without tripping wgpu validation (a shader
/// or layout error panics the whole app when clicked in the UI).
#[test]
fn gpu_porkchop_computes() {
    use eframe::egui_wgpu::wgpu;
    // Tiny block_on: busy-poll with a no-op waker (no async runtime in deps).
    fn block_on<F: std::future::Future>(mut fut: F) -> F::Output {
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        fn noop_raw() -> RawWaker {
            fn clone(_: *const ()) -> RawWaker { noop_raw() }
            fn noop(_: *const ()) {}
            RawWaker::new(std::ptr::null(), &RawWakerVTable::new(clone, noop, noop, noop))
        }
        let waker = unsafe { Waker::from_raw(noop_raw()) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::yield_now();
        }
    }
    let instance = wgpu::Instance::default();
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Some(a) => a,
        None => {
            eprintln!("no GPU adapter; skipping");
            return;
        }
    };
    let (device, queue) =
        block_on(adapter.request_device(&Default::default(), None)).expect("device");
    device.on_uncaptured_error(Box::new(|e| panic!("wgpu error: {e}")));
    let eph = Ephemeris::Kepler;
    let pc = crate::porkchop::compute(
        &device,
        &queue,
        &eph,
        BodyId::Mars,
        crate::porkchop::GridSpec {
            start: j2000() + Duration::from_days(9000.0),
            window_days: 900.0,
            tof_min_days: 60.0,
            tof_max_days: 600.0,
        },
    )
    .expect("gpu compute failed");
    let valid = pc.grid.iter().filter(|c| crate::porkchop::Porkchop::is_valid(**c)).count();
    println!("porkchop: {}/{} cells solved", valid, pc.grid.len());
    assert!(valid > pc.grid.len() / 4, "too few Lambert solutions: {valid}");
    // Spot-check one good cell against the CPU f64 Lambert.
    let (i, j) = (crate::porkchop::NX / 2, crate::porkchop::NY / 2);
    let [vd, _] = pc.cell(i, j);
    assert!(vd > 0.1 && vd < 50.0, "midgrid v∞ dep {vd}");

    // Every cell the shader claims to have solved must be a cell the f64 CPU
    // solver also solves, to the same conic (issue #2: the shader used to omit
    // the TOF bracket test and fabricate solutions over whole regions).
    let mut checked = 0;
    let mut agree = 0;
    let mut fabricated = Vec::new();
    // Deterministic sample of the grid: a stride coprime with the cell count.
    let n = pc.grid.len();
    for k in 0..600 {
        let idx = (k * 2411) % n;
        let (i, j) = (idx % crate::porkchop::NX, idx / crate::porkchop::NX);
        let cell = pc.cell(i, j);
        if !crate::porkchop::Porkchop::is_valid(cell) {
            continue;
        }
        checked += 1;
        let (r1, r2, tof_s) = pc.cell_lambert_inputs(&eph, BodyId::Mars, i, j);
        let mu = BodyId::Sun.gm();
        match crate::solver::lambert_rev(r1, r2, tof_s, mu, 0, false) {
            None => fabricated.push((i, j, cell[0])),
            Some((v1, _)) => {
                // Compare speeds only: v∞ folds in Earth's velocity, so match
                // the departure speed of the conic itself (f32 shader vs f64).
                let cpu = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
                let e = eph.state(
                    BodyId::Earth,
                    pc.start + Duration::from_days(pc.cell_time(i, j).0),
                );
                let gpu_v1 = [
                    v1[0] - e.vel_km_s[0],
                    v1[1] - e.vel_km_s[1],
                    v1[2] - e.vel_km_s[2],
                ];
                let cpu_vinf =
                    (gpu_v1[0].powi(2) + gpu_v1[1].powi(2) + gpu_v1[2].powi(2)).sqrt();
                assert!(cpu > 0.0);
                // f32 bisection on a 1e8-km-scale problem: ~1e-3 relative.
                let tol = 0.02 * cpu_vinf.max(1.0);
                assert!(
                    (cpu_vinf - cell[0] as f64).abs() < tol,
                    "cell ({i},{j}): GPU v∞ {} vs CPU {cpu_vinf}",
                    cell[0]
                );
                agree += 1;
            }
        }
    }
    println!("cross-checked {checked} valid cells, {agree} agree");
    assert!(checked > 50, "sample found too few valid cells: {checked}");
    // A handful of cells may straddle the bracket edge where f32 and f64
    // disagree on reachability; wholesale fabrication must be gone.
    assert!(
        fabricated.len() * 100 <= checked,
        "{} of {checked} GPU cells have no CPU solution: {:?}",
        fabricated.len(),
        &fabricated[..fabricated.len().min(8)]
    );
}

/// Low-thrust physics: a year of full along-track NEXT-C thrust must raise
/// the orbit's energy and add Δv ≈ a·t (within geometry effects).
#[test]
fn low_thrust_changes_orbit_as_expected() {
    use crate::dynamics::{propagate, propagate_thrusted, Thrust};
    let eph = Ephemeris::Kepler;
    let cfg = DynamicsConfig::default();
    let epoch = j2000();
    let r0 = 1.0 * AU_KM;
    let s0 = ScState {
        pos: [r0, 0.0, 0.0],
        vel: [0.0, dynamics::circular_speed_kms(r0), 0.0],
    };
    let year = Duration::from_days(365.25);
    let accel = crate::solver::Engine::NextC.accel_kms2();
    let segs = [[1.0, 0.0, 0.0]; 4];
    let thrust = Thrust {
        segs: &segs,
        accel_kms2: accel,
        total_s: year.to_seconds(),
    };
    let coast = propagate(&eph, &cfg, epoch, s0, year, 4);
    let burn = propagate_thrusted(&eph, &cfg, epoch, s0, year, 4, Some(&thrust));
    let energy = |s: &ScState| {
        let r = (s.pos[0].powi(2) + s.pos[1].powi(2) + s.pos[2].powi(2)).sqrt();
        (s.vel[0].powi(2) + s.vel[1].powi(2) + s.vel[2].powi(2)) / 2.0 - BodyId::Sun.gm() / r
    };
    let de = energy(&burn.last().unwrap().1) - energy(&coast.last().unwrap().1);
    assert!(de > 0.0, "prograde thrust must raise orbital energy");
    let dv_expected = accel * year.to_seconds();
    // Semi-major axis growth implies the burn actually delivered km/s-scale Δv.
    assert!(
        dv_expected > 5.0 && dv_expected < 15.0,
        "NEXT-C pair should deliver ~10 km/s/yr, got {dv_expected}"
    );
}

/// SEP search smoke: a NEXT-C Pluto orbiter search must produce candidates
/// that actually burn propellant and stay within the modeled budget.
#[test]
fn sep_pluto_search_uses_thrust() {
    use crate::solver::{Engine, Search, SolverConfig};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let mut cfg = SolverConfig::default();
    cfg.target = BodyId::Pluto;
    cfg.engine = Engine::NextC;
    cfg.auto_route = false;
    cfg.beam_width = 6;
    cfg.mutations = 4;
    cfg.scale_tof_to_target();
    cfg.tof_max_days = cfg.tof_max_days.min(15_000.0);
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2030, 1, 1);
    let mut s = Search::new(&eph, cfg.clone(), epoch0, None);
    let mut top = None;
    for _ in 0..5 {
        let (_, t) = s.step(&eph);
        top = Some(t);
    }
    let (score, g) = top.unwrap();
    let sol = s.solution_for(&eph, &g);
    assert!(score.is_finite());
    assert!(
        sol.thrust_dv_kms > 0.5,
        "SEP candidate is not thrusting ({:.2} km/s)",
        sol.thrust_dv_kms
    );
    assert!(sol.thrust_dv_kms < cfg.engine.max_dv_kms() * 1.5);
}

/// Issue #1: the launcher C3 cap must bind on *every* evaluator, not just the
/// low-thrust one. Issue #14: and it binds as a *hard* feasibility bound, not
/// a soft weighted penalty — a transfer needing C3 ≈ 200 km²/s² is simply not
/// flyable on a Falcon Heavy (C3 ≤ 60) and cannot buy its way past the cap.
#[test]
fn launcher_c3_cap_binds_on_direct_transfers() {
    use crate::solver::{evaluate_saved, Genome, Launcher, SolverConfig};
    let eph = Ephemeris::Kepler;
    let depart = j2000() + Duration::from_days(9000.0);
    // C3 = 200, over every cap. Kept in-plane (zero z ⇒ DLA = 0) so this test
    // isolates the raw C3 cap; the declination derate is exercised separately
    // by `high_dla_departure_loses_c3_capability` (issue #17).
    let g = Genome {
        depart_days: 0.0,
        legs: vec![250.0],
        vinf_dep: [10.0, 10.0, 0.0],
        thrust: Vec::new(),
        dsm: Vec::new(),
        branch: Vec::new(),
    };
    let score_for = |l: Launcher| {
        let cfg = SolverConfig {
            launcher: l,
            ..Default::default()
        };
        evaluate_saved(&eph, &cfg, depart, &g).score
    };
    let fh = score_for(Launcher::FalconHeavy);
    let kick = score_for(Launcher::KickStage);
    // C3 = 200 is over *both* caps, so both are demoted outright — no
    // exchange rate against the objective, just the infeasible offset plus
    // the km/s v∞ shortfall so the search still has a gradient home.
    assert!(
        fh > 1e8 && kick > 1e8,
        "over-cap candidates were not ruled infeasible: FH {fh}, kick {kick}"
    );
    // The tighter launcher is the further from flyable, and the gap is the
    // difference in v∞ capability in km/s — same unit as every other term.
    let expected = 130.0f64.sqrt() - 60.0f64.sqrt();
    assert!(
        (fh - kick - expected).abs() < 1e-6,
        "C3 shortfall is not the km/s v∞ deficit: FH {fh}, kick {kick}"
    );
    // And the over-cap solution must lose outright to a plausible in-cap one.
    let cheap = Genome {
        vinf_dep: [2.0, 2.0, 1.0],
        ..g.clone()
    };
    let cfg = SolverConfig {
        launcher: Launcher::FalconHeavy,
        ..Default::default()
    };
    assert!(
        evaluate_saved(&eph, &cfg, depart, &cheap).score
            < evaluate_saved(&eph, &cfg, depart, &g).score,
        "over-cap candidate outranked an in-cap one"
    );
}

/// Issue #1, tour branch: the same cap must bind on `evaluate_tour` — and,
/// per issue #14, bind hard there too.
#[test]
fn launcher_c3_cap_binds_on_tours() {
    use crate::solver::{evaluate_saved, Genome, Launcher, SolverConfig};
    let eph = Ephemeris::Kepler;
    let epoch0 = j2000() + Duration::from_days(9000.0);
    let base = |l: Launcher| SolverConfig {
        launcher: l,
        target: BodyId::Mars,
        route: vec![BodyId::Venus],
        ..Default::default()
    };
    // Scan a grid for a tour whose departure C3 lands between the two caps,
    // where only the tighter launcher is violated.
    let mut found = false;
    for d in 0..40 {
        for leg in [150.0f64, 200.0, 260.0, 320.0] {
            let g = Genome {
                depart_days: d as f64 * 10.0,
                legs: vec![leg, leg * 1.2],
                vinf_dep: [0.0; 3],
                thrust: Vec::new(),
            dsm: Vec::new(),
        branch: Vec::new(),
            };
            let fh = evaluate_saved(&eph, &base(Launcher::FalconHeavy), epoch0, &g);
            let kick = evaluate_saved(&eph, &base(Launcher::KickStage), epoch0, &g);
            let c3 = fh.vinf_dep_kms * fh.vinf_dep_kms;
            // Gate on the launcher that *can* fly it: the tour has to be a
            // real solved trajectory, and one whose C3 falls between the two
            // caps so exactly one launcher is violated.
            if kick.score >= 1e4 || !(60.0..130.0).contains(&c3) {
                continue;
            }
            found = true;
            assert!(
                fh.score > 1e8,
                "over-cap tour was not ruled infeasible: C3 {c3}, FH {}",
                fh.score
            );
            assert!(
                kick.score < 1e4,
                "in-cap tour was penalized: C3 {c3}, kick {}",
                kick.score
            );
            // What is left past the bound is the v∞ shortfall in km/s against
            // the DLA-derated Falcon Heavy cap (issue #17): the effective cap
            // depends on the departure asymptote's declination.
            let cap = 60.0 * crate::solver::dla_c3_derate(fh.genome.vinf_dep);
            let expected = 1e9 + kick.score + (c3.sqrt() - cap.sqrt());
            assert!(
                (fh.score - expected).abs() < 1e-6,
                "tour C3 shortfall wrong: C3 {c3}, FH {}, expected {expected}",
                fh.score
            );
        }
    }
    assert!(found, "no tour with C3 between the two launcher caps");
}

/// Issue #3: when the integrator runs out of steps, the propagation must say
/// so rather than tagging its partial state with the full arrival epoch.
#[test]
fn step_exhaustion_is_reported() {
    let eph = Ephemeris::Kepler;
    let cfg = DynamicsConfig {
        max_steps: 5,
        ..Default::default()
    };
    let r0 = AU_KM;
    let s0 = ScState {
        pos: [r0, 0.0, 0.0],
        vel: [0.0, dynamics::circular_speed_kms(r0), 0.0],
    };
    let span = Duration::from_days(4000.0);
    let start = j2000();
    let p = dynamics::propagate_checked(&eph, &cfg, start, s0, span, 50, None);
    assert!(!p.complete, "5 steps cannot cover 4000 days");
    assert!(p.fraction < 1.0, "fraction {} should be short", p.fraction);
    let (last, _) = *p.points.last().unwrap();
    assert!(
        last < start + span,
        "truncated arc still tagged with the full arrival epoch"
    );

    // A generous budget over the same span completes and lands on the epoch.
    let ok = dynamics::propagate_checked(
        &eph,
        &DynamicsConfig::default(),
        start,
        s0,
        span,
        50,
        None,
    );
    assert!(ok.complete);
    assert!((ok.fraction - 1.0).abs() < 1e-12);
    assert!((ok.points.last().unwrap().0 - (start + span)).to_seconds().abs() < 1e-3);
}

/// Issue #3, solver side: a truncated flight must be scored infeasible, never
/// ranked as an arrival.
#[test]
fn truncated_flights_score_infeasible() {
    use crate::solver::{evaluate, solver_dynamics, Genome, SolverConfig};
    let eph = Ephemeris::Kepler;
    let cfg = SolverConfig::default();
    let depart = j2000() + Duration::from_days(9000.0);
    let g = Genome {
        depart_days: 0.0,
        legs: vec![250.0],
        vinf_dep: [1.0, 3.0, 0.2],
        thrust: Vec::new(),
        dsm: Vec::new(),
        branch: Vec::new(),
    };
    let full = solver_dynamics();
    let good = evaluate(&eph, &full, &cfg, depart, &g, 32);
    assert!(good.score < 1e6, "baseline should be feasible: {}", good.score);

    let starved = DynamicsConfig {
        max_steps: 5,
        ..full
    };
    let bad = evaluate(&eph, &starved, &cfg, depart, &g, 32);
    assert!(
        bad.score >= 1e6 && bad.score > good.score,
        "truncated flight scored {} (feasible baseline {})",
        bad.score,
        good.score
    );
    assert!(bad.miss_km.is_infinite(), "truncated flight claimed a miss distance");
    assert!(
        bad.arrive < depart + Duration::from_days(250.0),
        "truncated flight claimed the full arrival epoch"
    );
}

/// Issue #4: `differential_correct` must shoot at the body it was given, not
/// at `SolverConfig::default()`'s Mars.
#[test]
fn differential_correct_honors_target() {
    use crate::solver::{differential_correct, solver_dynamics};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let full = solver_dynamics();
    let depart = j2000() + Duration::from_days(9000.0);
    let tof = Duration::from_days(150.0);
    let arrive = depart + tof;
    // Aim roughly inward toward Venus and let the corrector close the miss.
    let earth = eph.state(BodyId::Earth, depart);
    let ev = earth.vel_km_s;
    let en = (ev[0].powi(2) + ev[1].powi(2) + ev[2].powi(2)).sqrt();
    let v0 = [-ev[0] / en * 3.0, -ev[1] / en * 3.0, -ev[2] / en * 3.0];
    let (vinf, miss) =
        differential_correct(&eph, &full, depart, tof, BodyId::Venus, v0);
    assert!(miss < 5_000.0, "corrector left {miss:.0} km miss to Venus");

    // Independent check: propagating the corrected v∞ must end near Venus,
    // and nowhere near Mars — proving the target argument was honored.
    const EARTH_SOI_KM: f64 = 925_000.0;
    let vn = (vinf[0].powi(2) + vinf[1].powi(2) + vinf[2].powi(2)).sqrt();
    let s0 = ScState {
        pos: [
            earth.pos_km[0] + vinf[0] / vn * EARTH_SOI_KM,
            earth.pos_km[1] + vinf[1] / vn * EARTH_SOI_KM,
            earth.pos_km[2] + vinf[2] / vn * EARTH_SOI_KM,
        ],
        vel: [ev[0] + vinf[0], ev[1] + vinf[1], ev[2] + vinf[2]],
    };
    let traj = dynamics::propagate(&eph, &full, depart, s0, tof, 8);
    let sf = traj.last().unwrap().1;
    let dist = |b: BodyId| {
        let s = eph.state(b, arrive);
        ((sf.pos[0] - s.pos_km[0]).powi(2)
            + (sf.pos[1] - s.pos_km[1]).powi(2)
            + (sf.pos[2] - s.pos_km[2]).powi(2))
        .sqrt()
    };
    assert!(dist(BodyId::Venus) < 5_000.0, "not at Venus: {}", dist(BodyId::Venus));
    assert!(dist(BodyId::Mars) > 1e7, "suspiciously close to Mars too");
}

/// Issue #9: the mission file must survive a write/read round trip, keys and
/// all, with an explicit version stamp.
#[test]
fn mission_file_round_trips() {
    use crate::mission;
    use crate::solver::{evaluate_saved, Engine, Genome, Launcher, MissionType, SolverConfig};
    let eph = Ephemeris::Kepler;
    let epoch0 = j2000() + Duration::from_days(9000.0);
    let cfg = SolverConfig {
        target: BodyId::Mars,
        route: vec![BodyId::Venus],
        mission: MissionType::Orbit,
        engine: Engine::NextC,
        launcher: Launcher::Sls,
        auto_route: false,
        ..Default::default()
    };
    let g = Genome {
        depart_days: 30.0,
        legs: vec![180.0, 220.0],
        vinf_dep: [0.1, -0.2, 0.05],
        thrust: vec![[0.3, 0.4, 0.5]],
        dsm: vec![[0.5, 0.1, 0.2, 0.3]],
        branch: vec![0],
    };
    let sol = evaluate_saved(&eph, &cfg, epoch0, &g);
    let text = mission::serialize(&sol, &cfg);
    assert!(
        text.starts_with(&format!("version={}\n", mission::FORMAT_VERSION)),
        "missing version stamp: {text}"
    );
    let (cfg2, g2, depart) = mission::deserialize(&text).expect("round trip must parse");
    assert_eq!(cfg2.target, cfg.target);
    assert_eq!(cfg2.route, cfg.route);
    assert!(cfg2.mission == cfg.mission);
    assert!(cfg2.engine == cfg.engine);
    assert!(cfg2.launcher == cfg.launcher);
    assert_eq!(g2.legs, g.legs);
    assert_eq!(g2.vinf_dep, g.vinf_dep);
    assert_eq!(g2.thrust, g.thrust);
    assert_eq!(g2.dsm, g.dsm);
    assert_eq!(g2.branch, g.branch);
    assert!((depart.to_tdb_seconds() - sol.depart.to_tdb_seconds()).abs() < 1e-3);
}

/// Issue #9: a v1 file — no `version=` line, mission written as the UI
/// `label()` string — must still load. This is the regression the version
/// field exists to prevent: renaming a label may not orphan saved missions.
#[test]
fn mission_file_loads_legacy_label_keys() {
    use crate::mission;
    use crate::solver::{Engine, Launcher, MissionType};
    let legacy = format!(
        "target=Mars\nmission={}\nengine=next\nlauncher=sls\nroute=Venus\n\
         depart_tdb_s=1e8\nlegs=1.8e2,2.2e2\nvinf=1e-1,-2e-1,5e-2\n\
         thrust=3e-1:4e-1:5e-1\ndsm=5e-1:1e-1:2e-1:3e-1\n",
        MissionType::Orbit.label()
    );
    let (cfg, g, _) = mission::deserialize(&legacy).expect("v1 file must still load");
    assert!(cfg.mission == MissionType::Orbit);
    assert!(cfg.engine == Engine::NextC);
    assert!(cfg.launcher == Launcher::Sls);
    assert_eq!(cfg.route, vec![BodyId::Venus]);
    assert_eq!(g.legs, vec![180.0, 220.0]);
    // A pre-v3 file has no branch list: every leg flies branch 0, the
    // cheapest-departure arc those versions always picked.
    assert!(g.branch.is_empty(), "legacy file invented branch selectors");
}

/// #11: `newton_shoot` must return the best iterate it saw, not the last one.
///
/// The residual here is deliberately non-monotone in the way an n-body arc is:
/// it drops to a good value and then, once the walk passes a threshold, jumps
/// back up. A corrector that returns its final iterate reports the bad value;
/// one that keeps the best reports the good one.
#[test]
fn newton_shoot_keeps_best_iterate() {
    // The residual walks *through* its optimum: Newton converges toward x=0.5,
    // but every evaluation past x=0.45 returns a large value — a stand-in for
    // the Lambert branch flip a tour leg walks into just as it closes in. An
    // unguarded corrector accepts that step and reports the post-flip residual;
    // one that backtracks refuses it and reports the good iterate it already had.
    let shoot = |v: [f64; 3]| -> [f64; 3] {
        if v[0] > 0.45 {
            return [8000.0, 0.0, 0.0];
        }
        [(v[0] - 0.5) * 1000.0, v[1] * 1000.0, v[2] * 1000.0]
    };
    let (best_v, best_norm) = super::solver::newton_shoot(&shoot, [0.0, 0.0, 0.0], 20, 1.0);
    assert!(
        best_norm < 500.0,
        "corrector returned {best_norm} km; it should keep the best iterate it found"
    );
    // And the returned vector must be the one that produced that residual.
    let f = shoot(best_v);
    let n = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
    assert!(
        (n - best_norm).abs() < 1.0,
        "returned vector residual {n} km disagrees with reported {best_norm} km"
    );
}

/// #11: a step that makes the residual worse must be rejected, not accepted.
#[test]
fn newton_shoot_rejects_worsening_steps() {
    let shoot = |v: [f64; 3]| -> [f64; 3] {
        // Flat gradient in x with a large constant offset: Newton wants a huge
        // step, but every trial point is worse than the start.
        [1000.0 + v[0].abs() * 1e6, v[1] * 1e-9, v[2] * 1e-9]
    };
    let (_, n) = super::solver::newton_shoot(&shoot, [0.0, 0.0, 0.0], 20, 1.0);
    assert!(
        n <= 1000.0 + 1e-6,
        "corrector wandered uphill to {n} km from a 1000 km start"
    );
}

/// Issue #13, part 1: the DSM genome vector must *be* the maneuver.
///
/// Under the old parameterization the genome kicked the leg's *departure*
/// velocity and the Δv charged was a later Lambert mismatch, so the number
/// the optimizer paid for had no fixed relationship to the number it chose —
/// the departure perturbation was effectively free. Now the genome vector is
/// the burn applied at the maneuver point, and the charge is its magnitude.
#[test]
fn dsm_genome_vector_is_the_charged_burn() {
    use crate::solver::{evaluate_saved, Genome, SolverConfig};
    let (eph, _) = Ephemeris::load();
    let cfg = SolverConfig {
        target: BodyId::Mars,
        route: vec![BodyId::Venus],
        auto_route: false,
        ..Default::default()
    };
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2028, 1, 1);

    // Several distinct burns, so a coincidental match can't carry the test.
    for burn in [
        [0.05f64, -0.03, 0.02],
        [0.4, 0.1, -0.2],
        [-0.15, 0.25, 0.05],
    ] {
        let g = Genome {
            depart_days: 20.0,
            legs: vec![160.0, 240.0],
            vinf_dep: [0.0; 3],
            thrust: Vec::new(),
            dsm: vec![
                [0.4, burn[0], burn[1], burn[2]],
                [0.5, 0.0, 0.0, 0.0],
            ],
            branch: vec![0, 0],
        };
        let sol = evaluate_saved(&eph, &cfg, epoch0, &g);
        // A solved tour, feasible or not: `bad()` marks an unsolvable one
        // with an infinite miss and would report no Δv at all.
        assert!(sol.miss_km.is_finite(), "tour did not solve, score {}", sol.score);
        let want = (burn[0] * burn[0] + burn[1] * burn[1] + burn[2] * burn[2]).sqrt();
        assert!(
            (sol.dsm_dv_kms - want).abs() < 1e-9,
            "charged Δv {} is not the genome burn {want} ({burn:?})",
            sol.dsm_dv_kms
        );
    }
}

/// Issue #13, part 2: degeneracy to the ballistic leg must be *continuous*,
/// not a single coincidence at exactly zero.
///
/// The old code bypassed the DSM path entirely at `dv == 0`; for any nonzero
/// kick, however tiny, it re-ran `lambert_best` on the second sub-arc, which
/// could land on a different revolution count or branch and jump the arrival
/// velocity and the charged Δv by a finite amount. So an arbitrarily small
/// maneuver could change the score arbitrarily much.
///
/// Sweep the burn magnitude across nine orders of magnitude on a long,
/// multi-revolution geometry — the regime where the branch set is richest and
/// a flip is most likely — and require every observable to converge to the
/// ballistic leg at a rate bounded by the burn itself.
#[test]
fn dsm_converges_continuously_to_ballistic() {
    use crate::solver::{evaluate_saved, Genome, SolverConfig};
    let (eph, _) = Ephemeris::load();
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2028, 1, 1);

    // Long legs against inner-planet nodes: TOF spans several transfer
    // periods, so `max_revs` admits multiple families and both branches of
    // each. This is precisely where the old second Lambert could flip.
    // These specific geometries were found by sweeping legs, departures and
    // burn positions against the *old* code and keeping the ones where its
    // second `lambert_best` demonstrably flipped branch under a vanishing
    // kick. They are the regression, not decoration.
    let cases: [(BodyId, Vec<BodyId>, Vec<f64>, f64); 4] = [
        (BodyId::Mars, vec![BodyId::Venus], vec![900.0, 1170.0], 0.45),
        (BodyId::Jupiter, vec![BodyId::Venus, BodyId::Earth], vec![900.0, 1170.0, 1440.0], 0.45),
        (BodyId::Jupiter, vec![BodyId::Venus, BodyId::Earth], vec![180.0, 420.0, 700.0], 0.45),
        (BodyId::Saturn, vec![BodyId::Earth], vec![640.0, 900.0], 0.6),
    ];

    let mut checked = 0;
    for (target, route, legs, frac) in cases {
        let cfg = SolverConfig {
            target,
            route: route.clone(),
            auto_route: false,
            ..Default::default()
        };
        let n = legs.len();
        let base = Genome {
            depart_days: 0.0,
            legs,
            vinf_dep: [0.0; 3],
            thrust: Vec::new(),
            dsm: Vec::new(),
            branch: vec![0; n],
        };
        let ballistic = evaluate_saved(&eph, &cfg, epoch0, &base);
        if ballistic.score >= 1e4 {
            continue; // geometry with no conic solution: nothing to compare
        }

        // A fixed unit direction, scaled down decade by decade.
        let dir = {
            let d = [0.6f64, -0.7, 0.39];
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            [d[0] / n, d[1] / n, d[2] / n]
        };
        // (deviation / burn) at each magnitude: a *rate*, so the check is
        // scale-free and does not have to guess this geometry's sensitivity.
        let mut rates: Vec<(f64, f64)> = Vec::new();
        for k in 0..10 {
            let mag = 1e-3 * 10f64.powi(-(k as i32)); // 1e-3 … 1e-12
            let mut g = base.clone();
            g.dsm = vec![[0.5, 0.0, 0.0, 0.0]; n];
            g.dsm[0] = [frac, dir[0] * mag, dir[1] * mag, dir[2] * mag];
            let sol = evaluate_saved(&eph, &cfg, epoch0, &g);
            assert!(
                sol.score < 1e4,
                "{target:?}: a {mag:e} km/s burn made the leg unsolvable"
            );

            // The charge is the burn: it vanishes with it.
            assert!(
                (sol.dsm_dv_kms - mag).abs() <= 1e-12 + 1e-9 * mag,
                "{target:?}: {mag:e} km/s burn charged {}",
                sol.dsm_dv_kms
            );

            // How far the leg moved: score plus both v∞ endpoints, i.e. the
            // whole observable state of the transfer.
            let dev = (sol.score - ballistic.score).abs()
                + (sol.vinf_dep_kms - ballistic.vinf_dep_kms).abs()
                + (sol.vinf_arr_kms - ballistic.vinf_arr_kms).abs();
            // Below this the deviation is floating-point noise on an O(1)
            // score, not a signal about the parameterization.
            if dev > 1e-9 {
                rates.push((mag, dev / mag));
            }
            checked += 1;
        }

        // Lipschitz continuity, stated without assuming the constant: a
        // smooth leg has a bounded deviation *rate*, so shrinking the burn by
        // nine decades must not inflate dev/burn. A branch or revolution flip
        // is a finite jump — dev stops shrinking while the burn keeps going,
        // so the rate grows without bound and this catches it by orders of
        // magnitude, not by a hair. The 100x slack absorbs the genuine
        // conditioning of a long multi-rev leg.
        if let Some((mag0, rate0)) = rates.first().copied() {
            for (mag, rate) in &rates {
                assert!(
                    *rate <= rate0 * 100.0,
                    "{target:?}: deviation rate blew up from {rate0:e}/km/s at \
                     {mag0:e} to {rate:e}/km/s at {mag:e} — the leg does not \
                     converge continuously to the ballistic arc"
                );
            }
        }
    }
    assert!(checked >= 20, "not enough solvable geometries swept ({checked})");
}

/// Issue #14, part 2: alternative Lambert branches must be *visible* to the
/// caller, not silently swallowed by a greedy local pick, and the per-leg
/// branch gene has to actually change the trajectory that gets scored.
#[test]
fn lambert_branches_are_exposed_and_selectable() {
    use crate::solver::{evaluate_saved, lambert_best, lambert_candidates, Genome, SolverConfig};
    let mu = BodyId::Sun.gm();
    // A near-circular 1 AU → 1.5 AU geometry over a long TOF: several
    // revolution counts fit, each with a low and a high branch.
    let r1 = [1.496e8, 0.0, 0.0];
    let r2 = [0.0, 2.244e8, 0.0];
    let tof = 1400.0 * 86_400.0;
    let vref = [0.0, 29.8, 0.0];
    let c = lambert_candidates(r1, r2, tof, mu, vref);
    assert!(c.len() > 2, "geometry offered only {} arc(s)", c.len());

    // Ordered cheapest-departure first, and index 0 is exactly `lambert_best`.
    let cost = |v: [f64; 3]| {
        let d = [v[0] - vref[0], v[1] - vref[1], v[2] - vref[2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    for w in c.windows(2) {
        assert!(cost(w[0].0) <= cost(w[1].0), "candidates are not sorted");
    }
    assert_eq!(lambert_best(r1, r2, tof, mu, vref).unwrap().0, c[0].0);
    // The alternatives are genuinely distinct trajectories.
    assert!(
        c.iter().skip(1).any(|a| a.0 != c[0].0),
        "every candidate was the same arc"
    );

    // And the gene reaches the score: branch 0 is the historical behaviour,
    // some other branch flies something else.
    let (eph, _) = Ephemeris::load();
    let cfg = SolverConfig {
        target: BodyId::Jupiter,
        route: vec![BodyId::Earth],
        auto_route: false,
        ..Default::default()
    };
    let epoch0 = Epoch::from_gregorian_utc_at_midnight(2028, 1, 1);
    let base = Genome {
        depart_days: 30.0,
        legs: vec![700.0, 900.0],
        vinf_dep: [0.0; 3],
        thrust: Vec::new(),
        dsm: Vec::new(),
        branch: vec![0, 0],
    };
    let s0 = evaluate_saved(&eph, &cfg, epoch0, &base);
    // An empty branch list must mean the same thing as all-zeros.
    let mut unset = base.clone();
    unset.branch = Vec::new();
    assert_eq!(
        s0.score,
        evaluate_saved(&eph, &cfg, epoch0, &unset).score,
        "an unset branch list is not branch 0"
    );
    let differs = (1..4).any(|k| {
        let mut g = base.clone();
        g.branch = vec![k, 0];
        evaluate_saved(&eph, &cfg, epoch0, &g).score != s0.score
    });
    assert!(differs, "the branch gene never changed the scored trajectory");
}

/// Restore path (issue #15 follow-up): reopening a mission re-runs
/// `correct_direct` from the persisted (already-corrected) v∞. That second
/// pass must stay converged, not drift back up the focusing fold — otherwise
/// a reopen would degrade the trajectory it just restored.
#[test]
fn bplane_correction_is_idempotent_on_reload() {
    use crate::solver::{correct_direct, solver_dynamics, MissionType};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let depart = Epoch::from_tdb_seconds(8.476940205951389e8);
    let tof = Duration::from_days(2.8631775643736773e2);
    let full = solver_dynamics();
    let first = correct_direct(
        &eph,
        &full,
        depart,
        tof,
        BodyId::Mars,
        MissionType::Orbit,
        [-1.550225727753165, 2.260312907214572, 1.8088770400273846],
    );
    // Re-correct from the persisted v∞, exactly as boot restore does.
    let second = correct_direct(
        &eph,
        &full,
        depart,
        tof,
        BodyId::Mars,
        MissionType::Orbit,
        first.vinf_dep,
    );
    let a = first.periapsis_alt_km.expect("first pass has periapsis");
    let b = second.periapsis_alt_km.expect("reload pass has periapsis");
    assert!(
        (a - b).abs() < 20.0,
        "reload drifted periapsis {a:.0} -> {b:.0} km"
    );
    assert!(
        second.bplane_err_km.unwrap() < 50.0,
        "reload left {:.1} km B-plane error",
        second.bplane_err_km.unwrap()
    );
}

/// Issue #15: position-at-fixed-epoch targeting folds near the planet
/// (gravitational focusing), so stage one alone stalls at ~9,500 km on this
/// saved 2026 Mars window. The B-plane stage must converge through that floor
/// onto the mission's periapsis geometry.
#[test]
fn bplane_correction_beats_focusing_floor() {
    use crate::solver::{correct_direct, solver_dynamics, MissionType};
    let (eph, label) = Ephemeris::load();
    if !require_kernels(&label, "SPICE") {
        return;
    }
    let depart = Epoch::from_tdb_seconds(8.476940205951389e8);
    let tof = Duration::from_days(2.8631775643736773e2);
    let vinf0 = [
        -1.550225727753165e0,
        2.260312907214572e0,
        1.8088770400273846e0,
    ];
    let full = solver_dynamics();
    let c = correct_direct(&eph, &full, depart, tof, BodyId::Mars, MissionType::Orbit, vinf0);
    // Stage one stalls at the focusing floor — that is the bug being pinned.
    assert!(
        c.center_miss_km > 5_000.0 && c.center_miss_km < 20_000.0,
        "stage-one floor moved: {:.0} km (formulation changed?)",
        c.center_miss_km
    );
    let berr = c.bplane_err_km.expect("arc reached the SOI; B-plane stage must run");
    assert!(berr < 50.0, "B-plane error {berr:.1} km");
    let alt = c.periapsis_alt_km.expect("periapsis defined");
    // Orbit profile targets rp = 1.5 R, i.e. altitude = 0.5 R ≈ 1695 km.
    let want = 0.5 * BodyId::Mars.radius_km();
    assert!(
        (alt - want).abs() < 100.0,
        "periapsis altitude {alt:.0} km, wanted ~{want:.0}"
    );
    let dv = c.arrival_dv_kms.expect("insertion dv defined");
    assert!(dv > 0.3 && dv < 3.0, "insertion dv {dv:.2} km/s");
    println!(
        "stage1 miss {:.0} km | B-plane err {berr:.2} km | periapsis alt {alt:.0} km | v inf arr {:.2} | insertion dv {dv:.2}",
        c.center_miss_km,
        c.vinf_arr_kms.unwrap()
    );
}

