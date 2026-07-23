# Orbital

A high-fidelity solar-system simulator and deterministic interplanetary trajectory solver, in a single native Rust app.

Orbital renders the solar system from real JPL ephemerides and propagates spacecraft under post-Newtonian n-body dynamics, then uses that same physics to design real transfers: Lambert/porkchop screening, a Lambert-seeded beam search over gravity-assist tours, low-thrust (SEP) profiles, and a two-phase corrector that polishes a scouted trajectory to mission grade — including B-plane targeting at arrival. The solver is validated against published mission data (it rediscovers the Mars 2020 launch to within a day and ~1% of launch energy) and the whole propagation chain is built to be bit-reproducible.

## What it does

- **DE440 ephemerides via ANISE**, with a Keplerian fallback when SPICE kernels aren't present — the app labels which fidelity mode it's running in.
- **1PN relativistic n-body propagation** using an adaptive Dormand–Prince 5(4) integrator.
- **Trajectory search**: Lambert/porkchop screening, beam search over multi-flyby gravity-assist tours, engine models (ballistic, NSTAR, NEXT, AEPS), and a headless CLI for route search independent of the GUI.
- **Mission-grade correction**: a two-phase differential corrector that refines a scouted trajectory under full n-body dynamics, with B-plane targeting at arrival.
- **GPU-accelerated porkchop plots** for launch-window screening.
- **Deterministic by construction**: integer sim ticks (no wall-clock-derived epoch), a pure propagator with no threads or randomness, and a test that verifies bit-identical output across runs. See [`docs/DETERMINISM.md`](docs/DETERMINISM.md).

## Architecture

| Path | What |
|---|---|
| `src/ephemeris.rs` | DE440 ephemeris access (via ANISE) with a Keplerian fallback |
| `src/dynamics.rs` | Newtonian n-body + 1PN relativistic propagator (DP5(4)) |
| `src/solver.rs` | Lambert solver, beam search, tour discovery, differential/B-plane correction |
| `src/porkchop.rs` | GPU porkchop-plot screening |
| `src/bodies.rs` | Body catalog and ephemeris coverage |
| `src/ui/`, `src/render.rs` | egui panels and the wgpu 3D view |
| `src/cli.rs` | Headless trajectory search |
| `docs/` | Validation methodology and determinism guarantees |

Rendering and search read a pre-sampled interpolation table (planets from DE440 to ~0.03 km; moons/asteroids from analytic models with correctly-sized gravity but approximate phase), while accepted full-fidelity propagation queries the SPICE kernels directly, so final trajectories are mission-grade throughout.

## Building

```sh
cargo build --release
cargo run --release            # launch the GUI
```

Requires a recent stable Rust toolchain; the app is pinned to the wgpu 22.1 / egui 0.29 line and all dependencies resolve from crates.io.

### Ephemerides (optional but recommended)

With no data present, Orbital runs on built-in JPL approximate Keplerian elements — everything works, but body states are lower fidelity. For DE440 accuracy, drop SPICE kernels into `data/` (gitignored; `de440s.bsp` is ~32 MB, from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/). Optional satellite kernels (`jup365.bsp`, `sat441.bsp`, `nep097.bsp`) add moon targets. Any `.bsp` in `data/` loads automatically at startup.

### Command line

```sh
# Direct/auto-route search to a target from a start date
cargo run --release -- search Mars 2026-07-20

# Constrain a gravity-assist route and search budget
cargo run --release -- search Jupiter 2028-01-01 via=venus,earth,earth steps=1500

# Write the winning trajectory to a mission file the GUI can open
cargo run --release -- search Mars 2026-11-01 via=direct save=mission.orbital

# Tour mode launches the GUI pre-seeded with a route search
cargo run --release -- tour Jupiter via=venus,earth,earth
```

`search` arguments (any order after the target): a `YYYY-MM-DD` start date, `via=body,body,…` (or `via=direct`), `mode=flyby|orbit|land`, `engine=ballistic|nstar|next|aeps`, `beam=N`, `steps=N`, `restarts=N`, `seed=N`, `save=<path>` (write a loadable mission file). A headless `save=` mission stores the scouted trajectory; opening it in the GUI re-runs the differential/B-plane correction, so both agree on the same mission.

## Status

The solver is validated against published mission data — it rediscovers the Mars 2020 launch to within one day of departure and ~1% of launch energy, and finds the 2026/27 type-II Mars transfer window consistent with the JPL Trajectory Browser. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the full check set (Lambert conservation, VEEGA tours vs Galileo-class values, autonomous route discovery) and [`docs/DETERMINISM.md`](docs/DETERMINISM.md) for the reproducibility guarantee and its scope (bit-exact per binary/platform; cross-platform libm differences are noted, not hidden).

## License

MIT — see [`LICENSE`](LICENSE).
