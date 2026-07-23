# Orbital

A solar-system simulator with relativistic dynamics and an interplanetary
trajectory solver, in a single native Rust app (eframe/egui + wgpu).

Orbital models the solar system from real JPL ephemerides with post-Newtonian
spacecraft dynamics, then designs real transfers on top of that foundation:
Lambert/porkchop screening, a Lambert-seeded beam search, gravity-assist tour
discovery, low-thrust (SEP) profiles, and a two-phase corrector that polishes a
scouted trajectory to mission grade under full n-body dynamics — including
B-plane targeting at arrival.

## Status

The solver is validated against published mission data — it rediscovers the
Mars 2020 launch to within one day of departure and ~1% of launch energy, and
finds the 2026/27 type-II Mars window consistent with the JPL Trajectory
Browser. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the full set of
checks (Lambert conservation, VEEGA tours vs Galileo-class values, autonomous
route discovery). Determinism guarantees are in
[`docs/DETERMINISM.md`](docs/DETERMINISM.md).

## Building

```sh
cargo build --release
cargo run --release            # launch the GUI
```

Requires a recent stable Rust toolchain. All dependencies resolve from
crates.io; the app is pinned to the wgpu 22.1 / egui 0.29 line.

### Ephemerides (optional but recommended)

With no data present, Orbital runs on built-in JPL approximate Keplerian
elements — everything works, but body states are lower fidelity and the app
labels itself as running on the *Keplerian approximation*.

For DE440 accuracy, drop SPICE kernels into `data/` (they are gitignored — a
`de440s.bsp` is ~32 MB):

```sh
mkdir -p data
# Fetch from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
#   de440s.bsp   (planets; required for DE440 mode)
# Optional satellite kernels for moon targets:
#   jup365.bsp, sat441.bsp, nep097.bsp
```

Any `.bsp` in `data/` is loaded automatically at startup.

## Command line

Beyond the GUI, Orbital exposes a headless search:

```sh
# Direct/auto-route search to a target from a start date
cargo run --release -- search Mars 2026-07-20

# Constrain a gravity-assist route and search budget
cargo run --release -- search Jupiter 2028-01-01 via=venus,earth,earth steps=1500

# Tour mode launches the GUI pre-seeded with a route search
cargo run --release -- tour Jupiter via=venus,earth,earth
```

`search` arguments (any order after the target): a `YYYY-MM-DD` start date,
`via=body,body,…` (or `via=direct`), `mode=flyby|orbit|land`,
`engine=ballistic|nstar|next|aeps`, `beam=N`, `steps=N`, `restarts=N`,
`seed=N`.

## Layout

| Path | What |
|---|---|
| `src/ephemeris.rs` | DE440 (via ANISE) with a Keplerian fallback |
| `src/dynamics.rs` | Newtonian n-body + 1PN relativistic propagator (DP5(4)) |
| `src/solver.rs` | Lambert, beam search, tours, differential/B-plane correction |
| `src/porkchop.rs` | GPU porkchop screening |
| `src/ui/`, `src/render.rs` | egui panels and the wgpu 3D view |
| `docs/` | validation, determinism, and design notes |

## License

MIT — see [`LICENSE`](LICENSE).
