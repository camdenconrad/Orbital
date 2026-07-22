//! GPU porkchop plots: Lambert's problem solved for every (departure, TOF)
//! cell of a launch-window grid in one compute dispatch.
//!
//! Division of labor: the RTX's thousands of f32 lanes screen hundreds of
//! thousands of independent Lambert solves (f32 is plenty for km/s-scale
//! screening); anything the user picks off the plot is re-evaluated on the
//! CPU in f64 by the ordinary solver, so the determinism guarantee is
//! untouched — the GPU only *proposes*.
//!
//! Body states are pre-sampled on the CPU (from the render table) and
//! interpolated linearly in-shader for arrival epochs.

use crate::bodies::{BodyId, DAY_S};
use crate::ephemeris::Ephemeris;
use eframe::egui_wgpu::wgpu;
use hifitime::{Duration, Epoch};

pub const NX: usize = 256; // departure axis
pub const NY: usize = 192; // TOF axis

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    nx: u32,
    ny: u32,
    n_span: u32,
    _pad0: u32,
    tof0_s: f32,
    dtof_s: f32,
    span_dt_s: f32,
    mu: f32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
struct BodyState {
    pos: [f32; 4],
    vel: [f32; 4],
}

/// The grid's launch-window extent.
#[derive(Clone, Copy)]
pub struct GridSpec {
    pub start: Epoch,
    pub window_days: f64,
    pub tof_min_days: f64,
    pub tof_max_days: f64,
}

/// One computed grid: v∞ at departure and arrival per cell, km/s.
pub struct Porkchop {
    pub start: Epoch,
    pub window_days: f64,
    pub tof_min_days: f64,
    pub tof_max_days: f64,
    /// Row-major [j * NX + i]: (v∞ dep, v∞ arr); ≥ 1e8 marks no solution.
    pub grid: Vec<[f32; 2]>,
}

impl Porkchop {
    pub fn cell(&self, i: usize, j: usize) -> [f32; 2] {
        self.grid[j * NX + i]
    }

    /// (depart_days offset, tof_days) for a cell.
    pub fn cell_time(&self, i: usize, j: usize) -> (f64, f64) {
        (
            self.window_days * i as f64 / (NX - 1) as f64,
            self.tof_min_days
                + (self.tof_max_days - self.tof_min_days) * j as f64 / (NY - 1) as f64,
        )
    }
}

const SHADER: &str = r#"
struct Params {
    nx: u32, ny: u32, n_span: u32, _pad0: u32,
    tof0_s: f32, dtof_s: f32, span_dt_s: f32, mu: f32,
};
struct BodyState { pos: vec4<f32>, vel: vec4<f32> };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> earth: array<BodyState>;   // per depart column
@group(0) @binding(2) var<storage, read> tgt_states: array<BodyState>;  // dense span
@group(0) @binding(3) var<storage, read_write> results: array<vec2<f32>>;

fn stumpff_c(z: f32) -> f32 {
    if z > 1e-6 { return (1.0 - cos(sqrt(z))) / z; }
    if z < -1e-6 { return (cosh(sqrt(-z)) - 1.0) / (-z); }
    return 0.5;
}
fn stumpff_s(z: f32) -> f32 {
    if z > 1e-6 { let sz = sqrt(z); return (sz - sin(sz)) / (sz * sz * sz); }
    if z < -1e-6 { let sz = sqrt(-z); return (sinh(sz) - sz) / (sz * sz * sz); }
    return 1.0 / 6.0;
}

// Single-rev prograde Lambert, bisection on z (mirrors the CPU f64 version).
fn lambert(r1: vec3<f32>, r2: vec3<f32>, tof: f32, mu: f32) -> mat2x3<f32> {
    let fail = mat2x3<f32>(vec3(1e12), vec3(1e12));
    let r1n = length(r1);
    let r2n = length(r2);
    let cosd = clamp(dot(r1, r2) / (r1n * r2n), -1.0, 1.0);
    var dtheta = acos(cosd);
    if r1.x * r2.y - r1.y * r2.x < 0.0 { dtheta = 6.2831853 - dtheta; }
    let a_coef = sin(dtheta) * sqrt(r1n * r2n / (1.0 - cosd));
    if abs(a_coef) < 1.0 { return fail; }

    let tgt = sqrt(mu) * tof;
    var z_lo = -157.9; // -16π²
    var z_hi = 39.47;  // (2π)² - ε
    // Walk z_lo up to the y>0 validity boundary.
    var y_lo = r1n + r2n + a_coef * (z_lo * stumpff_s(z_lo) - 1.0) / sqrt(stumpff_c(z_lo));
    if y_lo < 0.0 {
        var bad = z_lo;
        var good = z_hi;
        for (var k = 0; k < 40; k++) {
            let mid = 0.5 * (bad + good);
            let ym = r1n + r2n + a_coef * (mid * stumpff_s(mid) - 1.0) / sqrt(stumpff_c(mid));
            if ym > 0.0 { good = mid; } else { bad = mid; }
        }
        z_lo = good;
    }
    var z = 0.0;
    for (var k = 0; k < 60; k++) {
        z = 0.5 * (z_lo + z_hi);
        let c = stumpff_c(z);
        let s = stumpff_s(z);
        let yv = r1n + r2n + a_coef * (z * s - 1.0) / sqrt(c);
        var t: f32;
        if yv < 0.0 { t = -1e30; } else { t = pow(yv / c, 1.5) * s + a_coef * sqrt(yv); }
        if t < tgt { z_lo = z; } else { z_hi = z; }
    }
    let c = stumpff_c(z);
    let s = stumpff_s(z);
    let yv = r1n + r2n + a_coef * (z * s - 1.0) / sqrt(c);
    if yv <= 0.0 { return fail; }
    let f = 1.0 - yv / r1n;
    let g = a_coef * sqrt(yv / mu);
    let gdot = 1.0 - yv / r2n;
    if abs(g) < 1e-6 { return fail; }
    let v1 = (r2 - f * r1) / g;
    let v2 = (gdot * r2 - r1) / g;
    return mat2x3<f32>(v1, v2);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.nx || gid.y >= p.ny { return; }
    let e = earth[gid.x];
    let tof = p.tof0_s + f32(gid.y) * p.dtof_s;
    // Arrival epoch on the dense target span (earth column i departs at the
    // same span times scaled by column spacing — encoded via pos.w).
    let t_arr = e.pos.w + tof;
    let x = clamp(t_arr / p.span_dt_s, 0.0, f32(p.n_span - 2u));
    let i0 = u32(floor(x));
    let fr = x - floor(x);
    let t0 = tgt_states[i0];
    let t1 = tgt_states[i0 + 1u];
    let tp = mix(t0.pos.xyz, t1.pos.xyz, fr);
    let tv = mix(t0.vel.xyz, t1.vel.xyz, fr);

    let vs = lambert(e.pos.xyz, tp, tof, p.mu);
    let vinf_dep = length(vs[0] - e.vel.xyz);
    let vinf_arr = length(vs[1] - tv);
    results[gid.y * p.nx + gid.x] = vec2(vinf_dep, vinf_arr);
}
"#;

/// Compute a porkchop grid on the GPU. Blocking (the dispatch is tiny by GPU
/// standards — a frame's worth of work). Returns None if the GPU rejects
/// anything — errors are captured in a scope so a validation failure can
/// never take the app down (wgpu's default handler aborts the process).
pub fn compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    eph: &Ephemeris,
    target: BodyId,
    spec: GridSpec,
) -> Option<Porkchop> {
    use wgpu::util::DeviceExt;
    let GridSpec { start, window_days, tof_min_days, tof_max_days } = spec;
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    // CPU-side sampling (f64 → f32).
    let mut earth_states = Vec::with_capacity(NX);
    for i in 0..NX {
        let dt_days = window_days * i as f64 / (NX - 1) as f64;
        let e = start + Duration::from_days(dt_days);
        let s = eph.state(BodyId::Earth, e);
        earth_states.push(BodyState {
            // pos.w carries the departure offset (s) for the shader.
            pos: [
                s.pos_km[0] as f32,
                s.pos_km[1] as f32,
                s.pos_km[2] as f32,
                (dt_days * DAY_S) as f32,
            ],
            vel: [
                s.vel_km_s[0] as f32,
                s.vel_km_s[1] as f32,
                s.vel_km_s[2] as f32,
                0.0,
            ],
        });
    }
    let span_days = window_days + tof_max_days + 2.0;
    let span_dt_days = 0.5;
    let n_span = (span_days / span_dt_days).ceil() as usize + 2;
    let mut target_states = Vec::with_capacity(n_span);
    for k in 0..n_span {
        let e = start + Duration::from_days(k as f64 * span_dt_days);
        let s = eph.state(target, e);
        target_states.push(BodyState {
            pos: [s.pos_km[0] as f32, s.pos_km[1] as f32, s.pos_km[2] as f32, 0.0],
            vel: [s.vel_km_s[0] as f32, s.vel_km_s[1] as f32, s.vel_km_s[2] as f32, 0.0],
        });
    }

    let params = Params {
        nx: NX as u32,
        ny: NY as u32,
        n_span: n_span as u32,
        _pad0: 0,
        tof0_s: (tof_min_days * DAY_S) as f32,
        dtof_s: (((tof_max_days - tof_min_days) / (NY - 1) as f64) * DAY_S) as f32,
        span_dt_s: (span_dt_days * DAY_S) as f32,
        mu: BodyId::Sun.gm() as f32,
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("porkchop"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("porkchop"),
        layout: None,
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pc-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let earth_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pc-earth"),
        contents: bytemuck::cast_slice(&earth_states),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let target_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pc-target"),
        contents: bytemuck::cast_slice(&target_states),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (NX * NY * 8) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pc-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pc-read"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("porkchop"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: earth_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: target_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(NX.div_ceil(8) as u32, NY.div_ceil(8) as u32, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
    queue.submit([encoder.finish()]);

    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map channel").expect("map failed");
    let data = slice.get_mapped_range();
    let grid: Vec<[f32; 2]> = bytemuck::cast_slice::<u8, [f32; 2]>(&data).to_vec();
    drop(data);
    read_buf.unmap();

    // Surface any validation error as a soft failure. pop_error_scope is a
    // future; busy-poll it with a no-op waker (resolves after device.poll).
    let err = {
        use std::future::Future;
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        fn noop_raw() -> RawWaker {
            fn clone(_: *const ()) -> RawWaker {
                noop_raw()
            }
            fn noop(_: *const ()) {}
            RawWaker::new(std::ptr::null(), &RawWakerVTable::new(clone, noop, noop, noop))
        }
        let waker = unsafe { Waker::from_raw(noop_raw()) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(device.pop_error_scope());
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(e) => break e,
                Poll::Pending => device.poll(wgpu::Maintain::Wait),
            };
        }
    };
    if let Some(e) = err {
        eprintln!("porkchop GPU error: {e}");
        return None;
    }

    Some(Porkchop {
        start,
        window_days,
        tof_min_days,
        tof_max_days,
        grid,
    })
}
