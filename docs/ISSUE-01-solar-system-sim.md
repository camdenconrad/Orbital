# Solar-system simulator with relativistic dynamics (trajectory-solver foundation)

## Goal
A 3D eframe/wgpu app that models the solar system from real ephemerides with
post-Newtonian spacecraft dynamics, exposing a propagation API a trajectory
solver can build on.

## Why
We want to design real transfers (Lambert/porkchop, low-thrust optimization,
flyby tours). Those need mission-grade body states and dynamics, not toy
Kepler orbits — get the foundation right once.

## In scope
- [ ] Ephemeris layer: JPL DE440s via ANISE when `data/de440s.bsp` is present;
      built-in JPL approximate Keplerian elements as an always-works fallback.
- [ ] Dynamics: Newtonian n-body point-mass acceleration on the spacecraft from
      Sun + 8 planets (+Moon), plus 1PN relativistic correction per body
      (Moyer-style Schwarzschild terms; dominated by the Sun).
- [ ] Propagator: adaptive Dormand–Prince 5(4) with per-step error control,
      dense trajectory output. Units km/s, epochs via hifitime (TDB).
- [ ] 3D view: wgpu paint callback inside egui — instanced planet spheres,
      sampled orbit trails, propagated spacecraft trajectory, arcball camera,
      log/true-scale toggle for body radii.
- [ ] Time controls: epoch display, play/pause, time-scale slider, jump-to-date.
- [ ] Demo spacecraft: editable heliocentric state vector, propagate + render.

## Out of scope (for this issue)
- The solvers themselves (Lambert, porkchop scans, low-thrust optimal control,
  B-plane targeting) — they consume this API in follow-up issues.
- Full EIH cross/indirect terms, asteroid perturbations, SRP, drag, moons
  other than Luna.
- Texturing/lighting polish; n-body integration of the planets themselves
  (they come from the ephemeris).

## Owner
@camden

## Acceptance
- With `data/de440s.bsp` present, Earth's position matches JPL Horizons to
  <1000 km at J2000+10 yr; fallback mode agrees with DE to <0.1 AU.
- Propagating Mercury-like test orbit for 100 yr shows ~43″/century perihelion
  advance vs the Newtonian-only run (validates the PN term).
- App runs at interactive framerate, camera + time controls work, spacecraft
  trajectory renders after propagation.

## Refs
- JPL approximate elements: Standish, "Keplerian Elements for Approximate
  Positions of the Major Planets" (1800–2050 table).
- Moyer, "Formulation for Observed and Computed Values of DSN Data Types" (PN terms).
- ANISE: https://github.com/nyx-space/anise
