# Validation

How Orbital's physics and trajectory solver check out against reality.
Every claim below is enforced by a test in `src/tests.rs` (run
`cargo test --release`).

## Physics

| Check | Result | Test |
|---|---|---|
| Mercury perihelion advance (1PN term on vs off, 100 yr) | ~43″/century, matches GR's famous 42.98″ | `mercury_perihelion_advance` |
| Two-body energy conservation, 10 yr propagation | drift < 1e-8 relative | `energy_conserved_over_a_decade` |
| Keplerian fallback vs JPL DE440, J2000+10 yr | < 0.1 AU (fallback only; DE440 is the primary) | `spice_and_kepler_agree` |
| Cached-span interpolation vs direct DE440 query | 0.03 km (Mars, mid-span) | `bench_cached_span_build` |
| All 21 bodies return sane geometry (moons near parents, etc.) | pass | `all_bodies_have_sane_states` |

## Determinism

| Check | Result | Test |
|---|---|---|
| Propagation, two runs, bit-for-bit | identical (`f64::to_bits`) | `propagation_is_bit_reproducible` |
| Beam search, same seed, two runs | identical score sequence | `solver_is_deterministic_and_converges` |

## Trajectory solver vs real missions

**Mars 2020 (Perseverance).** Searching a 900-day window from 2020-01-01
with the min-total-Δv objective, the solver finds:

|  | Solver | Actual mission |
|---|---|---|
| Departure | 2020-07-29 | 2020-07-30 |
| Time of flight | 200 d | 203 d (arrive 2021-02-18) |
| Launch C3 | 14.3 km²/s² | 14.49 km²/s² |
| Arrival v∞ | 2.58 km/s | ~2.6 km/s |

One day off on departure, ~1% on launch energy. Enforced (with loose
tolerances) by `rediscovers_mars_2020_trajectory`.

**2026/27 Mars opportunity.** From 2026-07-20 the solver finds the type-II
transfer: depart 2026-11-04, TOF 309 d, C3 = 10.0 km²/s², arrival v∞
2.55 km/s — consistent with JPL Trajectory Browser values for that window.

**Lambert solver.** Conserves energy and angular momentum between endpoints
to 1e-9 relative; near-Hohmann geometry reproduces analytic vis-viva speeds
(`lambert_conserves_and_matches_hohmann`). Multi-rev solutions recover the
re-flown single-rev ellipse (`lambert_multirev_recovers_same_ellipse`);
two-body propagation round-trips to km/µm-per-s level
(`kepler_universal_round_trip`).

**Gravity-assist tours (patched-conic).** A VEEGA (Earth→Venus→Earth→Earth→
Jupiter) search from 2028 (orbit mission, 6 restarts × 1500 steps) finds:
depart 2029-11-07 at v∞ 3.30 km/s (C3 = 10.9 km²/s²), all three flybys
**unpowered** (Venus 9376 km, Earth 5380 km, Earth 1687 km altitudes),
Jupiter arrival v∞ 6.22 km/s, capture burn 1.01 km/s, TOF 5.7 yr. Total
post-launch Δv ≈ 1 km/s — Galileo-class (VEEGA, arrival v∞ ≈ 5.6, 6.1 yr).
Flybys respect the physical bending limit δ_max = 2·asin(μ/(μ + r_p·v∞²)) at
1.1 body radii; mismatch/turn deficits are charged as powered-flyby Δv
(`veega_tour_search_is_sane`).

**Autonomous route discovery.** With no route specified, the solver screens
every candidate sequence (direct + all 1–3 body combinations of a
target-appropriate assist alphabet, ~40 routes for Jupiter), refines the top
five (two seeds each), and polishes the winner — deterministically. For
Jupiter/2028 it discovered a **Mars→Earth→Earth** tour with every flyby
unpowered: depart 2029-08-02 (v∞ 5.20), Mars 6523 km, Earth 3998 km, Earth
10028 km, capture 0.94 km/s. Same-body legs are seeded near integer-year
resonances — the domain trick that makes resonant-return tours findable.
A manually specified VEEGA (`via=venus,earth,earth`, seed 12) still beats it
(score 4.31 vs 6.15); deeper screening budgets close that gap (`steps=`).

**Force model.** Point-mass Newtonian gravity from all catalogued bodies plus
the 1PN Schwarzschild correction per body (the Sun's term reproduces Mercury's
43″/century perihelion advance). Mission-grade propagation (the accepted-
trajectory corrector and tour refinement) additionally carries solar radiation
pressure as a cannonball term — Cr·(A/m) ≈ 0.0195 m²/kg, falling as 1/r² away
from the Sun — so B-plane targeting flies through the cruise perturbation
rather than below it. The coarse beam scout leaves SRP off to stay cheap; the
corrector reconverges on the target regardless (its whole job). Planetary
oblateness (J2) is not yet modeled — it matters only for very low flybys and
needs per-body pole orientation (tracked separately).

**Launch feasibility.** The launcher C3 cap is a hard bound (a candidate the
vehicle can't inject is ranked below every feasible one, not charged a
purchasable penalty). The usable cap additionally derates with the
declination of the launch asymptote (DLA): full capability for |DLA| ≤ the
launch-site latitude (28.5°, the Cape), falling as `cos(|DLA| − 28.5°)` toward
a polar asymptote (~0.5× at ±90°), since reaching a steeply inclined asymptote
costs a lofted/dogleg ascent or a plane change. DLA is `asin(v∞_z/|v∞|)` in the
ICRF equatorial frame the ephemeris uses. This bites only near-capability
trajectories — the validated low-C3 Mars/VEEGA solutions sit far under the cap
and are unchanged (Mars 2020 still rediscovers at 2020-07-30, C3 14.4).

**Mission types.** The arrival cost is mission-dependent: *flyby* (v∞ free),
*orbit* (capture at 1.5 radii into an e = 0.95 ellipse — gives ~1.06 km/s JOI
above after margin, vs ~0.6-0.9 km/s for real JOI into wider orbits), *land*
(propulsive descent for airless bodies, ~free aerodynamic entry for Venus/
Earth/Mars/Titan, unavailable for gas giants).

**Δv margins.** Propulsive arrival burns carry a finite-burn (gravity +
steering) margin rising with burn size across a 3–15% band, since a real
capture burn arcs over finite time rather than firing impulsively. Ballistic
missions additionally carry a 2% statistical TCM allocation on the
deterministic post-launch Δv for cruise navigation. Both are engineering
margins with cited real-world ranges, not derived arc integrals; they are
charged so the optimizer stops preferring high-v∞ arrivals that look cheap
only because the losses were omitted. The validated windows are unchanged
(Mars 2020 still rediscovers at 2020-07-30 within the 1% C3 tolerance).

**Multi-leg shooting (tours).** Accepting a tour launches background
multiple shooting: each Lambert leg is differentially corrected under the
full n-body dynamics (arrival body excluded per leg — its SOI hyperbola is
the flyby model's domain) until it hits its patch body at the patch epoch to
km-scale; flyby v∞/altitude are then recomputed from the corrected
velocities and the continuous n-body path replaces the conic sketch
(`tour_refines_to_mission_grade`).

**Two-phase refinement.** Accepted direct transfers are polished by a
Newton differential corrector (finite-difference Jacobian on v∞, full-
fidelity dynamics): the beam's ~1e5 km scouting miss converges to km-scale
in ≤5 iterations (`differential_correction_hits_target`) — the standard
scout→corrector structure of interplanetary design tools.

**GPU porkchop plots.** Lambert solved for a 256×192 (departure × TOF) grid
in one wgpu compute dispatch (f32 — screening precision). The GPU only
*proposes*: clicking a cell launches an f64 CPU refinement, so determinism
guarantees are unaffected.

## Method notes

- The search is a beam search over (departure, TOF, v∞) genomes — the
  architecture from 3DSolver — with every fresh candidate **Lambert-seeded**:
  its v∞ solves the two-body Earth→target boundary-value problem, so distinct
  launch windows compete globally instead of the beam collapsing into the
  first one it finds. Scoring propagates the full 1PN n-body dynamics.
- Search-grade dynamics: **all 21 bodies' gravity active** (same force model
  as final propagation), rel_tol 1e-8, injection at Earth's SOI along v∞.
  Accepted solutions re-propagate at rel_tol 1e-10.
- Tours are scouted patched-conic (Lambert legs + flyby bending limits),
  then refined to continuous n-body legs by multiple shooting on Accept.
  Patch times stay at the scouted schedule; re-optimizing the schedule under
  full dynamics is the remaining refinement axis.
- Known gaps: no explicit deep-space maneuvers (powered-flyby Δv stands in),
  retrograde/high-inclination transfers not searched.
- All 21 catalog bodies are now SPICE-covered: de440s + jup365 + sat441 +
  nep097 + codes_300ast (asteroids).

## Running the validation suite

`cargo test --release`. The validation tests need the real SPICE kernels in
`data/` (~32 MB); without them the ephemeris falls back to the analytic Kepler
model and the results mean nothing.

Those tests therefore **fail loudly** when the kernels are absent rather than
skipping — a silently-skipped `rediscovers_mars_2020_trajectory` is a green
suite that validates nothing, and that has already let a real regression
through. If an environment genuinely cannot carry the kernels, set

    ORBITAL_ALLOW_SKIP_SPICE=1

to downgrade the failure to a skip (each skipped test prints a `SKIPPING:`
line naming the ephemeris it actually loaded). Never set it in CI that is
meant to gate correctness.
