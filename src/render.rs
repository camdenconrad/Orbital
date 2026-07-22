//! 3D scene rendering inside egui via the wgpu paint callback.
//!
//! World units are AU. Spheres are drawn instanced; trails and trajectories
//! are line strips. Requires eframe's `depth_buffer: 24`.

use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Vec3};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
pub const MAX_SPHERES: usize = 64;
pub const MAX_LINE_VERTS: usize = 200_000;
pub const MAX_TRI_VERTS: usize = 60_000;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
pub struct SphereInstance {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub emissive: f32,
}

// No padding: the attribute offsets below are generated sequentially, so any
// gap between pos and color would shift every color read by one float (this
// exact bug once made the orange trajectory render green).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

/// Per-frame scene description handed from the UI thread to the callback.
#[derive(Clone, Default)]
pub struct Scene {
    pub view_proj: Mat4,
    pub spheres: Vec<SphereInstance>,
    pub line_verts: Vec<LineVertex>,
    /// (start, count) ranges into `line_verts`, each drawn as a line strip.
    pub strips: Vec<(u32, u32)>,
    /// Camera-facing ribbon triangles (thick lines): a plain triangle list.
    pub tri_verts: Vec<LineVertex>,
}

pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.6,
            pitch: 0.5,
            distance: 8.0,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cp * cy, cp * sy, sp)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Z);
        let near = (self.distance * 1e-4).max(1e-8);
        let proj = Mat4::perspective_infinite_rh(60f32.to_radians(), aspect, near);
        proj * view
    }
}

const SHADER: &str = r#"
struct Camera { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> camera: Camera;

struct SphereOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) emissive: f32,
    @location(3) world: vec3<f32>,
};

@vertex
fn vs_sphere(
    @location(0) vpos: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec3<f32>,
    @location(4) emissive: f32,
) -> SphereOut {
    var out: SphereOut;
    let world = center + vpos * radius;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.normal = vpos;
    out.color = color;
    out.emissive = emissive;
    out.world = world;
    return out;
}

@fragment
fn fs_sphere(in: SphereOut) -> @location(0) vec4<f32> {
    // Light from the origin (the Sun).
    let to_sun = normalize(-in.world);
    let diffuse = max(dot(normalize(in.normal), to_sun), 0.0);
    let lit = in.color * (0.12 + 0.88 * diffuse);
    let final_color = mix(lit, in.color, in.emissive);
    return vec4<f32>(final_color, 1.0);
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_line(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> LineOut {
    var out: LineOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

pub struct SceneRenderer {
    sphere_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    tri_pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    camera_bind: wgpu::BindGroup,
    sphere_mesh: wgpu::Buffer,
    sphere_index: wgpu::Buffer,
    sphere_index_count: u32,
    instances: wgpu::Buffer,
    lines: wgpu::Buffer,
    tris: wgpu::Buffer,
    // Frame state written in prepare(), read in paint().
    instance_count: u32,
    tri_count: u32,
    strips: Vec<(u32, u32)>,
}

fn uv_sphere(rings: u32, sectors: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    for r in 0..=rings {
        let phi = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * s as f32 / sectors as f32;
            verts.push([
                phi.sin() * theta.cos(),
                phi.sin() * theta.sin(),
                phi.cos(),
            ]);
        }
    }
    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (verts, idx)
}

impl SceneRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        use wgpu::util::DeviceExt;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("orbital-scene"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });

        let (verts, idx) = uv_sphere(24, 36);
        let sphere_mesh = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere-verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sphere_index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere-idx"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sphere-instances"),
            size: (MAX_SPHERES * std::mem::size_of::<SphereInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lines = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line-verts"),
            size: (MAX_LINE_VERTS * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tris = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tri-verts"),
            size: (MAX_TRI_VERTS * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        };

        let sphere_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("spheres"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sphere",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<SphereInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32, 3 => Float32x3, 4 => Float32],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sphere",
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth.clone()),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_line",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_line",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: false,
                ..depth
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let tri_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ribbons"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_line",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_line",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            sphere_pipeline,
            line_pipeline,
            tri_pipeline,
            camera_buf,
            camera_bind,
            sphere_mesh,
            sphere_index,
            sphere_index_count: idx.len() as u32,
            instances,
            lines,
            tris,
            instance_count: 0,
            tri_count: 0,
            strips: Vec::new(),
        }
    }
}

pub struct SceneCallback {
    pub scene: Scene,
}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let r: &mut SceneRenderer = resources.get_mut().expect("renderer not registered");
        let cam = CameraUniform {
            view_proj: self.scene.view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&r.camera_buf, 0, bytemuck::bytes_of(&cam));
        let n = self.scene.spheres.len().min(MAX_SPHERES);
        if n > 0 {
            queue.write_buffer(
                &r.instances,
                0,
                bytemuck::cast_slice(&self.scene.spheres[..n]),
            );
        }
        r.instance_count = n as u32;
        let ln = self.scene.line_verts.len().min(MAX_LINE_VERTS);
        if ln > 0 {
            queue.write_buffer(&r.lines, 0, bytemuck::cast_slice(&self.scene.line_verts[..ln]));
        }
        r.strips = self
            .scene
            .strips
            .iter()
            .filter(|(s, c)| (*s + *c) as usize <= ln && *c >= 2)
            .copied()
            .collect();
        let tn = self.scene.tri_verts.len().min(MAX_TRI_VERTS);
        if tn > 0 {
            queue.write_buffer(&r.tris, 0, bytemuck::cast_slice(&self.scene.tri_verts[..tn]));
        }
        r.tri_count = (tn - tn % 3) as u32;
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::epaint::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let r: &SceneRenderer = resources.get().expect("renderer not registered");
        pass.set_bind_group(0, &r.camera_bind, &[]);

        pass.set_pipeline(&r.line_pipeline);
        pass.set_vertex_buffer(0, r.lines.slice(..));
        for &(start, count) in &r.strips {
            pass.draw(start..start + count, 0..1);
        }

        if r.tri_count > 0 {
            pass.set_pipeline(&r.tri_pipeline);
            pass.set_vertex_buffer(0, r.tris.slice(..));
            pass.draw(0..r.tri_count, 0..1);
        }

        if r.instance_count > 0 {
            pass.set_pipeline(&r.sphere_pipeline);
            pass.set_vertex_buffer(0, r.sphere_mesh.slice(..));
            pass.set_vertex_buffer(1, r.instances.slice(..));
            pass.set_index_buffer(r.sphere_index.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..r.sphere_index_count, 0, 0..r.instance_count);
        }
    }
}
