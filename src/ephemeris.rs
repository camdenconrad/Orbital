//! Body state source: JPL SPICE kernels via ANISE when available, otherwise
//! Keplerian approximations.
//!
//! Every `.bsp` in `data/` is loaded (sorted by name, so load order — and
//! therefore any segment-priority effects — is deterministic). `de440s.bsp`
//! covers Sun/planets/Moon/Pluto; drop in satellite kernels (`jup365.bsp`,
//! `sat441.bsp`, `nep097.bsp`) or asteroid SPKs for mission-grade moon and
//! asteroid states. Bodies a loaded kernel can't answer for fall back to the
//! Keplerian model per body (probed once at load, so there is no per-frame
//! error handling and no log spam).
//!
//! All states are heliocentric ICRF/J2000 equatorial. Units: km, km/s. Epochs: TDB.
//!
//! Determinism: state() is a pure function of (loaded kernels, body, epoch).
//! No caching, no threading, no wall clock.

use crate::bodies::{BodyId, ALL_BODIES, AU_KM, CENTURY_D, DAY_S};
use hifitime::Epoch;

#[derive(Clone, Copy, Debug, Default)]
pub struct StateVec {
    pub pos_km: [f64; 3],
    pub vel_km_s: [f64; 3],
}

pub enum Ephemeris {
    #[cfg(feature = "spice")]
    Spice {
        almanac: Box<anise::almanac::Almanac>,
        /// Which bodies the loaded kernels can actually answer for.
        covered: [bool; ALL_BODIES.len()],
    },
    Kepler,
    /// Pre-sampled span with cubic-Hermite interpolation — the solver's fast
    /// path. Bodies outside the table fall back to the Keplerian model.
    /// Deterministic: the table is a pure function of (source, span, step).
    Cached(CachedSpan),
}

pub struct CachedSpan {
    t0_tdb_s: f64,
    /// Per body: (sample step seconds, sampled states), or None if not tabulated.
    /// Steps are per-body: fast movers sample densely, outer planets coarsely.
    tables: Vec<Option<(f64, Vec<StateVec>)>>,
}

impl Ephemeris {
    /// Load every .bsp under data/ (sorted); fall back to Keplerian elements.
    pub fn load() -> (Self, String) {
        #[cfg(feature = "spice")]
        {
            let mut kernels: Vec<std::path::PathBuf> = std::fs::read_dir("data")
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|x| x == "bsp"))
                        .collect()
                })
                .unwrap_or_default();
            kernels.sort();
            if !kernels.is_empty() {
                let mut almanac = anise::almanac::Almanac::default();
                let mut loaded = Vec::new();
                for k in &kernels {
                    match almanac.load(&k.to_string_lossy()) {
                        Ok(a) => {
                            almanac = a;
                            loaded.push(k.file_name().unwrap().to_string_lossy().into_owned());
                        }
                        Err(e) => eprintln!("failed to load {}: {e}", k.display()),
                    }
                }
                if !loaded.is_empty() {
                    // Probe coverage per body at one representative epoch.
                    let probe = Epoch::from_gregorian_utc_at_midnight(2026, 1, 1);
                    let mut covered = [false; ALL_BODIES.len()];
                    for (i, b) in ALL_BODIES.iter().enumerate() {
                        covered[i] = try_spice(&almanac, *b, probe).is_some();
                    }
                    let n_cov = covered.iter().filter(|c| **c).count();
                    return (
                        Ephemeris::Spice { almanac: Box::new(almanac), covered },
                        format!("SPICE: {} ({n_cov}/{} bodies)", loaded.join("+"), ALL_BODIES.len()),
                    );
                }
            }
        }
        (
            Ephemeris::Kepler,
            "Keplerian approximation (place de440s.bsp in data/ for DE440)".into(),
        )
    }

    /// Heliocentric ICRF/J2000 state of `body` at `epoch` (TDB).
    pub fn state(&self, body: BodyId, epoch: Epoch) -> StateVec {
        match self {
            #[cfg(feature = "spice")]
            Ephemeris::Spice { almanac, covered } => {
                if covered[body.index()] {
                    if let Some(s) = try_spice(almanac, body, epoch) {
                        return s;
                    }
                }
                kepler_state(self, body, epoch)
            }
            Ephemeris::Kepler => kepler_state(self, body, epoch),
            Ephemeris::Cached(span) => {
                match &span.tables[body.index()] {
                    Some((step_s, table)) => span.interp(*step_s, table, epoch),
                    None => kepler_state(self, body, epoch),
                }
            }
        }
    }

    /// Tabulate `mask`ed bodies over [start, end] for fast interpolated
    /// queries. Queries outside the span clamp to its ends — callers pad the
    /// span. Sample steps adapt per body (~period/2000, clamped to
    /// [0.25 d, 16 d]) and the tables build in parallel; both are safe for
    /// determinism because each table's *values* depend only on
    /// (source, span, step), never on thread timing.
    pub fn cached_span(
        &self,
        start: Epoch,
        end: Epoch,
        mask: &[bool; ALL_BODIES.len()],
    ) -> Ephemeris {
        let t0 = start.to_tdb_seconds();
        let span_s = end.to_tdb_seconds() - t0;
        let step_for = |b: BodyId| -> f64 {
            if b == BodyId::Sun {
                16.0 * DAY_S
            } else {
                (b.period_days().abs() * DAY_S / 2000.0).clamp(0.25 * DAY_S, 16.0 * DAY_S)
            }
        };
        // Query source: with the full kernel set loaded, every translate pays
        // to search the big satellite kernels' segment lists. The solver mask
        // is Sun/planets/Moon — all in de440s — so a slim almanac with just
        // that file answers ~40x faster. Values are identical (same DE440 data).
        #[cfg(feature = "spice")]
        let slim: Option<Ephemeris> = {
            let path = "data/de440s.bsp";
            if matches!(self, Ephemeris::Spice { .. }) && std::path::Path::new(path).exists() {
                anise::almanac::Almanac::default().load(path).ok().map(|almanac| {
                    let probe = Epoch::from_gregorian_utc_at_midnight(2026, 1, 1);
                    let mut covered = [false; ALL_BODIES.len()];
                    for (i, b) in ALL_BODIES.iter().enumerate() {
                        covered[i] = try_spice(&almanac, *b, probe).is_some();
                    }
                    Ephemeris::Spice { almanac: Box::new(almanac), covered }
                })
            } else {
                None
            }
        };
        #[cfg(not(feature = "spice"))]
        let slim: Option<Ephemeris> = None;
        let source: &Ephemeris = slim.as_ref().unwrap_or(self);

        let mut tables: Vec<Option<(f64, Vec<StateVec>)>> = vec![None; ALL_BODIES.len()];
        std::thread::scope(|scope| {
            let handles: Vec<_> = ALL_BODIES
                .iter()
                .zip(mask)
                .map(|(b, m)| {
                    // Moons are never tabulated: their periods are days, so a
                    // shared-span table either under-samples them into Hermite
                    // garbage (Io: 4 samples/orbit) or blows up memory. The
                    // Cached fallback composes parent-from-table + analytic
                    // circle instead — smooth at any zoom.
                    (*m && b.parent().is_none()).then(|| {
                        scope.spawn(move || {
                            let step_s = step_for(*b);
                            let n = ((span_s / step_s).ceil() as usize).max(1) + 1;
                            let table: Vec<StateVec> = (0..n)
                                .map(|i| {
                                    source.state(
                                        *b,
                                        start + hifitime::Duration::from_seconds(i as f64 * step_s),
                                    )
                                })
                                .collect();
                            (step_s, table)
                        })
                    })
                })
                .collect();
            for (slot, h) in tables.iter_mut().zip(handles) {
                if let Some(h) = h {
                    *slot = Some(h.join().expect("table build panicked"));
                }
            }
        });
        Ephemeris::Cached(CachedSpan { t0_tdb_s: t0, tables })
    }
}

impl CachedSpan {
    fn interp(&self, step_s: f64, table: &[StateVec], epoch: Epoch) -> StateVec {
        let x = (epoch.to_tdb_seconds() - self.t0_tdb_s) / step_s;
        let i = (x.floor() as isize).clamp(0, table.len() as isize - 2) as usize;
        let t = (x - i as f64).clamp(0.0, 1.0);
        let (a, b) = (&table[i], &table[i + 1]);
        let h = step_s;
        // Cubic Hermite with exact endpoint velocities.
        let (t2, t3) = (t * t, t * t * t);
        let (h00, h10, h01, h11) = (
            2.0 * t3 - 3.0 * t2 + 1.0,
            t3 - 2.0 * t2 + t,
            -2.0 * t3 + 3.0 * t2,
            t3 - t2,
        );
        let (d00, d10, d01, d11) = (
            (6.0 * t2 - 6.0 * t) / h,
            3.0 * t2 - 4.0 * t + 1.0,
            (-6.0 * t2 + 6.0 * t) / h,
            3.0 * t2 - 2.0 * t,
        );
        let mut out = StateVec::default();
        for k in 0..3 {
            out.pos_km[k] = h00 * a.pos_km[k]
                + h10 * h * a.vel_km_s[k]
                + h01 * b.pos_km[k]
                + h11 * h * b.vel_km_s[k];
            out.vel_km_s[k] = d00 * a.pos_km[k]
                + d10 * a.vel_km_s[k]
                + d01 * b.pos_km[k]
                + d11 * b.vel_km_s[k];
        }
        out
    }
}

#[cfg(feature = "spice")]
fn try_spice(
    almanac: &anise::almanac::Almanac,
    body: BodyId,
    epoch: Epoch,
) -> Option<StateVec> {
    use anise::constants::frames::SUN_J2000;
    use anise::constants::orientations::J2000;
    use anise::frames::Frame;
    let target = Frame::new(body.naif_id(), J2000);
    almanac
        .translate(target, SUN_J2000, epoch, None)
        .ok()
        .map(|s| StateVec {
            pos_km: [s.radius_km.x, s.radius_km.y, s.radius_km.z],
            vel_km_s: [s.velocity_km_s.x, s.velocity_km_s.y, s.velocity_km_s.z],
        })
}

/// JPL approximate elements: a(au), e, I, L, varpi, Omega (deg) + rates per Julian century.
/// Planets + Pluto: E.M. Standish, "Keplerian Elements for Approximate Positions
/// of the Major Planets" (1800–2050). Asteroids: JPL SBDB osculating elements
/// near J2000 with only the mean-longitude rate carried (fallback fidelity).
#[rustfmt::skip]
const ELEMENTS: [(BodyId, [f64; 6], [f64; 6]); 13] = [
    (BodyId::Mercury, [0.387_099_27, 0.205_635_93,  7.004_979_02, 252.250_323_50,  77.457_796_28,  48.330_765_93],
                      [0.000_000_37, 0.000_019_06, -0.005_947_49, 149_472.674_111_75, 0.160_476_89, -0.125_340_81]),
    (BodyId::Venus,   [0.723_335_66, 0.006_776_72,  3.394_676_05, 181.979_099_50, 131.602_467_18,  76.679_842_55],
                      [0.000_003_90, -0.000_041_07, -0.000_788_90, 58_517.815_387_29, 0.002_683_29, -0.277_694_18]),
    (BodyId::Earth,   [1.000_002_61, 0.016_711_23, -0.000_015_31, 100.464_571_66, 102.937_681_93, 0.0],
                      [0.000_005_62, -0.000_043_92, -0.012_946_68, 35_999.372_449_81, 0.323_273_64, 0.0]),
    (BodyId::Mars,    [1.523_710_34, 0.093_394_10,  1.849_691_42,  -4.553_432_05, -23.943_629_59,  49.559_538_91],
                      [0.000_018_47, 0.000_078_82, -0.008_131_31, 19_140.302_684_99, 0.444_410_88, -0.292_573_43]),
    (BodyId::Jupiter, [5.202_887_00, 0.048_386_24,  1.304_396_95,  34.396_440_51,  14.728_479_83, 100.473_909_09],
                      [-0.000_116_07, -0.000_132_53, -0.001_837_14, 3_034.746_127_75, 0.212_526_68, 0.204_691_06]),
    (BodyId::Saturn,  [9.536_675_94, 0.053_861_79,  2.485_991_87,  49.954_244_23,  92.598_878_31, 113.662_424_48],
                      [-0.001_250_60, -0.000_509_91, 0.001_936_09, 1_222.493_622_01, -0.418_972_16, -0.288_677_94]),
    (BodyId::Uranus,  [19.189_164_64, 0.047_257_44, 0.772_637_83, 313.238_104_51, 170.954_276_30,  74.016_925_03],
                      [-0.001_961_76, -0.000_043_97, -0.002_429_39, 428.482_027_85, 0.408_052_81, 0.042_405_89]),
    (BodyId::Neptune, [30.069_922_76, 0.008_590_48, 1.770_043_47, -55.120_029_69,  44.964_762_27, 131.784_225_74],
                      [0.000_262_91, 0.000_051_05, 0.000_353_72, 218.459_453_25, -0.322_414_64, -0.005_086_64]),
    (BodyId::Pluto,   [39.482_116_75, 0.248_827_30, 17.140_012_06, 238.929_038_33, 224.068_916_29, 110.303_936_84],
                      [-0.000_315_96, 0.000_051_70, 0.000_048_18, 145.207_805_15, -0.040_629_42, -0.011_834_82]),
    (BodyId::Ceres,   [2.767_5, 0.075_8, 10.594, 186.06, 153.9, 80.31],
                      [0.0, 0.0, 0.0, 7_812.3, 0.0, 0.0]),
    (BodyId::Vesta,   [2.361_5, 0.088_7, 7.14, 350.4, 254.54, 103.81],
                      [0.0, 0.0, 0.0, 9_919.5, 0.0, 0.0]),
    (BodyId::Pallas,  [2.773_0, 0.229_9, 34.83, 163.7, 123.08, 173.08],
                      [0.0, 0.0, 0.0, 7_799.2, 0.0, 0.0]),
    (BodyId::Hygiea,  [3.142_1, 0.112_5, 3.84, 75.0, 235.5, 283.2],
                      [0.0, 0.0, 0.0, 6_477.3, 0.0, 0.0]),
];

/// Obliquity of the ecliptic at J2000, for rotating ecliptic elements into
/// the equatorial ICRF frame the DE kernels use.
const OBLIQUITY_RAD: f64 = 0.409_092_804_222_329_25; // 23.43928 deg

fn kepler_state(eph: &Ephemeris, body: BodyId, epoch: Epoch) -> StateVec {
    if body == BodyId::Sun {
        return StateVec::default();
    }
    // Moons: circular orbit about the parent in the ecliptic plane. Crude —
    // right period, radius, and direction (Triton retrograde), arbitrary
    // phase and zero inclination. Fine for perturbation magnitudes; drop in a
    // satellite SPK for flyby-grade geometry.
    if let Some(parent) = body.parent() {
        let p = eph.state(parent, epoch);
        let t = epoch.to_tdb_seconds();
        let n = 2.0 * std::f64::consts::PI / (body.period_days() * DAY_S);
        let (s, c) = (n * t).sin_cos();
        let r = body.moon_orbit_radius_km();
        let v = n.abs() * r;
        let dir = n.signum();
        let pos = ecl_to_eq([r * c, r * s, 0.0]);
        let vel = ecl_to_eq([-v * s * dir, v * c * dir, 0.0]);
        return StateVec {
            pos_km: [p.pos_km[0] + pos[0], p.pos_km[1] + pos[1], p.pos_km[2] + pos[2]],
            vel_km_s: [
                p.vel_km_s[0] + vel[0],
                p.vel_km_s[1] + vel[1],
                p.vel_km_s[2] + vel[2],
            ],
        };
    }

    let (_, el0, rate) = *ELEMENTS
        .iter()
        .find(|(b, _, _)| *b == body)
        .expect("no elements for body");

    // Julian centuries of TDB since J2000.
    let t = epoch.to_tdb_seconds() / (DAY_S * CENTURY_D);
    let a = (el0[0] + rate[0] * t) * AU_KM;
    let ecc = el0[1] + rate[1] * t;
    let inc = (el0[2] + rate[2] * t).to_radians();
    let mean_lon = (el0[3] + rate[3] * t).to_radians();
    let varpi = (el0[4] + rate[4] * t).to_radians();
    let raan = (el0[5] + rate[5] * t).to_radians();

    let argp = varpi - raan;
    let mean_anom = (mean_lon - varpi).rem_euclid(2.0 * std::f64::consts::PI);

    // Kepler's equation, Newton iteration.
    let mut ea = if ecc < 0.8 { mean_anom } else { std::f64::consts::PI };
    for _ in 0..12 {
        let f = ea - ecc * ea.sin() - mean_anom;
        ea -= f / (1.0 - ecc * ea.cos());
    }

    let mu = BodyId::Sun.gm();
    let (sin_ea, cos_ea) = ea.sin_cos();
    let r = a * (1.0 - ecc * cos_ea);
    // Perifocal position and velocity.
    let xp = a * (cos_ea - ecc);
    let yp = a * (1.0 - ecc * ecc).sqrt() * sin_ea;
    let vf = (mu * a).sqrt() / r;
    let vxp = -vf * sin_ea;
    let vyp = vf * (1.0 - ecc * ecc).sqrt() * cos_ea;

    let rot = |x: f64, y: f64| -> [f64; 3] {
        let (so, co) = raan.sin_cos();
        let (si, ci) = inc.sin_cos();
        let (sw, cw) = argp.sin_cos();
        let x1 = cw * x - sw * y;
        let y1 = sw * x + cw * y;
        let y2 = ci * y1;
        let z2 = si * y1;
        [co * x1 - so * y2, so * x1 + co * y2, z2]
    };

    StateVec {
        pos_km: ecl_to_eq(rot(xp, yp)),
        vel_km_s: ecl_to_eq(rot(vxp, vyp)),
    }
}

fn ecl_to_eq(v: [f64; 3]) -> [f64; 3] {
    let (s, c) = OBLIQUITY_RAD.sin_cos();
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}
