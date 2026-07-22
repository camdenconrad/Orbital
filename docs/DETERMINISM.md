# Determinism

Orbital follows besom's rule: same inputs, bit-identical outputs.

- **Sim clock is integer ticks.** The UI epoch is derived (`epoch0 + ticks × 1 s`),
  never accumulated from wall-clock frame deltas. Frame rate, pauses, and host
  timing cannot change where anything is at tick N.
- **The propagator is a pure function.** Fixed Dormand–Prince 5(4) algorithm,
  sequential (no threads), no wall clock, no randomness, no hash-map iteration.
  Ephemeris queries are pure functions of (loaded kernels, body, epoch); kernels
  load in sorted filename order so segment priority is reproducible.
- **Guarded by test.** `propagation_is_bit_reproducible` compares two runs
  bit-for-bit (`f64::to_bits`).

Scope of the guarantee: bit-exact on a given binary + platform. Across
platforms, `sin`/`cos`/`powf` come from the system libm and may differ in the
last ulp; if cross-platform bit-exactness ever matters, swap in a software
libm and pin the target.

# Body catalog and ephemeris coverage

`data/*.bsp` are all loaded. `de440s.bsp` covers Sun, planets, Earth's Moon,
and the Pluto barycenter. The other cataloged bodies — Io, Europa, Ganymede,
Callisto, Titan, Triton, Ceres, Vesta, Pallas, Hygiea — fall back to Keplerian
models (asteroids: fixed J2000 elements; moons: circular orbits about their
parent with correct radius/period/direction, arbitrary phase). Their
*gravitational pull* on a propagated spacecraft is therefore correctly sized
everywhere, but their fallback geometry is not flyby-grade.

All satellite/asteroid kernels are present in `data/` (jup365, sat441,
nep097, codes_300ast) — coverage is probed at startup and shown in the UI
label. Fidelity is tiered by use: **rendering and search** read a pre-sampled
interpolation table (planets from DE440 to ~0.03 km; moons/asteroids from
analytic models — their gravity is correctly sized, phases approximate),
while the **accepted full-fidelity propagation** queries the kernels
directly, so final trajectories are mission-grade everywhere.
