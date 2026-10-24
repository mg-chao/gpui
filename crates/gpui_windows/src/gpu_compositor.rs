use anyhow::{Context, Result};
use gpui::{GpuDamage, PaintGpuTexture, Scene};
use windows::Win32::{
    Foundation::RECT,
    Graphics::{Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, Direct3D11::*, Dxgi::Common::*},
};
#[cfg(debug_assertions)]
use windows::{
    Win32::Graphics::Direct3D::Fxc::{
        D3DCOMPILE_ENABLE_STRICTNESS, D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile,
    },
    core::{PCSTR, s},
};

use crate::gpu_extensions::{DirectXGpuService, DirectXGpuTexture};

pub(crate) struct TexturePipeline {
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    parameters: ID3D11Buffer,
    blend: ID3D11BlendState,
    linear: ID3D11SamplerState,
    nearest: ID3D11SamplerState,
}

impl TexturePipeline {
    pub(crate) fn new(device: &ID3D11Device) -> Result<Self> {
        #[cfg(debug_assertions)]
        let (compiled_vertex, compiled_fragment) = {
            let source = include_bytes!("gpu_texture.hlsl");
            let compile = |entry: PCSTR, target: PCSTR| -> Result<Vec<u8>> {
                let mut code = None;
                let mut errors = None;
                let result = unsafe {
                    D3DCompile(
                        source.as_ptr().cast(),
                        source.len(),
                        s!("gpui_gpu_texture"),
                        None,
                        None,
                        entry,
                        target,
                        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
                        0,
                        &mut code,
                        Some(&mut errors),
                    )
                };
                if let Err(error) = result {
                    let message = errors
                        .as_ref()
                        .map(|blob| unsafe {
                            String::from_utf8_lossy(std::slice::from_raw_parts(
                                blob.GetBufferPointer().cast(),
                                blob.GetBufferSize(),
                            ))
                            .into_owned()
                        })
                        .unwrap_or_default();
                    return Err(error).context(message);
                }
                let code = code.context("Shader compiler returned no bytecode")?;
                Ok(unsafe {
                    std::slice::from_raw_parts(code.GetBufferPointer().cast(), code.GetBufferSize())
                }
                .to_vec())
            };
            (
                compile(s!("vertex_main"), s!("vs_4_1"))?,
                compile(s!("fragment_main"), s!("ps_4_1"))?,
            )
        };
        #[cfg(debug_assertions)]
        let (vertex_bytes, fragment_bytes) =
            (compiled_vertex.as_slice(), compiled_fragment.as_slice());
        #[cfg(not(debug_assertions))]
        let (vertex_bytes, fragment_bytes) = (
            include_bytes!(concat!(env!("OUT_DIR"), "/gpu_texture_vs.cso")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/gpu_texture_ps.cso")).as_slice(),
        );
        let mut vertex = None;
        let mut fragment = None;
        let mut parameters = None;
        let mut blend = None;
        unsafe {
            device.CreateVertexShader(vertex_bytes, None, Some(&mut vertex))?;
            device.CreatePixelShader(fragment_bytes, None, Some(&mut fragment))?;
            device.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: 80,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut parameters),
            )?;
            let mut description = D3D11_BLEND_DESC::default();
            description.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D11_BLEND_ONE,
                DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D11_BLEND_OP_ADD,
                SrcBlendAlpha: D3D11_BLEND_ONE,
                DestBlendAlpha: D3D11_BLEND_INV_SRC_ALPHA,
                BlendOpAlpha: D3D11_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
            };
            device.CreateBlendState(&description, Some(&mut blend))?;
        }
        let sampler = |filter| -> Result<ID3D11SamplerState> {
            let mut state = None;
            unsafe {
                device.CreateSamplerState(
                    &D3D11_SAMPLER_DESC {
                        Filter: filter,
                        AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                        ComparisonFunc: D3D11_COMPARISON_NEVER,
                        MaxLOD: f32::MAX,
                        ..Default::default()
                    },
                    Some(&mut state),
                )
            }?;
            state.context("Creating texture sampler")
        };
        Ok(Self {
            vertex: vertex.context("Creating texture vertex shader")?,
            fragment: fragment.context("Creating texture fragment shader")?,
            parameters: parameters.context("Creating texture parameters")?,
            blend: blend.context("Creating texture blend state")?,
            linear: sampler(D3D11_FILTER_MIN_MAG_MIP_LINEAR)?,
            nearest: sampler(D3D11_FILTER_MIN_MAG_MIP_POINT)?,
        })
    }

    pub(crate) fn draw(
        &self,
        context: &ID3D11DeviceContext,
        service: &DirectXGpuService,
        texture: &PaintGpuTexture,
        viewport: [u32; 2],
    ) -> Result<()> {
        let native = texture
            .texture
            .as_any()
            .downcast_ref::<DirectXGpuTexture>()
            .context("GPU texture is not a D3D11 image")?;
        native.validate(service)?;
        let destination = texture.destination;
        let clip = texture.content_mask.bounds;
        let [a, b, c, d, tx, ty] = texture.options.transform;
        let parameters: [f32; 20] = [
            destination.origin.x.0,
            destination.origin.y.0,
            destination.size.width.0,
            destination.size.height.0,
            a,
            b,
            c,
            d,
            tx,
            ty,
            viewport[0] as f32,
            viewport[1] as f32,
            clip.origin.x.0,
            clip.origin.y.0,
            clip.origin.x.0 + clip.size.width.0,
            clip.origin.y.0 + clip.size.height.0,
            texture.options.opacity,
            0.0,
            0.0,
            0.0,
        ];
        unsafe {
            context.UpdateSubresource(&self.parameters, 0, None, parameters.as_ptr().cast(), 0, 0);
            context.IASetInputLayout(None);
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            context.VSSetShader(&self.vertex, None);
            context.PSSetShader(&self.fragment, None);
            context.VSSetConstantBuffers(0, Some(&[Some(self.parameters.clone())]));
            context.PSSetConstantBuffers(0, Some(&[Some(self.parameters.clone())]));
            context.PSSetShaderResources(0, Some(&[Some(native.shader_resource_view().clone())]));
            context.PSSetSamplers(
                0,
                Some(&[Some(if texture.options.linear_filtering {
                    self.linear.clone()
                } else {
                    self.nearest.clone()
                })]),
            );
            context.OMSetBlendState(&self.blend, None, u32::MAX);
            context.Draw(4, 0);
            context.PSSetShaderResources(0, Some(&[None]));
        }
        Ok(())
    }
}

pub(crate) struct RetainedTarget {
    pub texture: ID3D11Texture2D,
    pub view: ID3D11RenderTargetView,
}

pub(crate) struct ExtensionRenderer {
    pub pipeline: TexturePipeline,
    pub rasterizer: ID3D11RasterizerState,
    pub front: Option<RetainedTarget>,
    pub candidate: Option<RetainedTarget>,
    pub repair: Option<RECT>,
    pub damage: DamageTracker,
    pub occluded: bool,
    pub atlas_revision: u64,
}

impl ExtensionRenderer {
    pub(crate) fn new(device: &ID3D11Device) -> Result<Self> {
        let mut rasterizer = None;
        unsafe {
            device.CreateRasterizerState(
                &D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_SOLID,
                    CullMode: D3D11_CULL_NONE,
                    DepthClipEnable: true.into(),
                    ScissorEnable: true.into(),
                    MultisampleEnable: true.into(),
                    ..Default::default()
                },
                Some(&mut rasterizer),
            )
        }?;
        Ok(Self {
            pipeline: TexturePipeline::new(device)?,
            rasterizer: rasterizer.context("Creating extension compositor rasterizer")?,
            front: None,
            candidate: None,
            repair: None,
            damage: DamageTracker::default(),
            occluded: false,
            atlas_revision: 0,
        })
    }

    pub(crate) fn invalidate(&mut self) {
        self.front = None;
        self.candidate = None;
        self.repair = None;
        self.damage.invalidate();
        self.occluded = false;
    }

    pub(crate) fn ensure_targets(&mut self, device: &ID3D11Device, size: [u32; 2]) -> Result<()> {
        if self.front.is_none() {
            self.front = Some(RetainedTarget::new(device, size)?);
        }
        if self.candidate.is_none() {
            self.candidate = Some(RetainedTarget::new(device, size)?);
        }
        Ok(())
    }
}

pub(crate) fn copy_rectangle(
    context: &ID3D11DeviceContext,
    source: &ID3D11Texture2D,
    destination: &ID3D11Texture2D,
    rectangle: RECT,
) {
    let bounds = D3D11_BOX {
        left: rectangle.left as u32,
        top: rectangle.top as u32,
        front: 0,
        right: rectangle.right as u32,
        bottom: rectangle.bottom as u32,
        back: 1,
    };
    unsafe {
        context.CopySubresourceRegion(
            destination,
            0,
            bounds.left,
            bounds.top,
            0,
            source,
            0,
            Some(&bounds),
        );
    }
}

impl RetainedTarget {
    pub(crate) fn new(device: &ID3D11Device, size: [u32; 2]) -> Result<Self> {
        let mut texture = None;
        unsafe {
            device.CreateTexture2D(
                &D3D11_TEXTURE2D_DESC {
                    Width: size[0],
                    Height: size[1],
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
                    ..Default::default()
                },
                None,
                Some(&mut texture),
            )
        }?;
        let texture = texture.context("Creating retained window texture")?;
        let mut view = None;
        unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut view)) }?;
        Ok(Self {
            texture,
            view: view.context("Creating retained window target")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TextureStamp {
    order: u32,
    placement: [u32; 16],
    size: [u32; 2],
    revision: u64,
    resource: u64,
    service: u64,
}

impl TextureStamp {
    fn new(texture: &PaintGpuTexture) -> Self {
        let bounds = texture.destination;
        let clip = texture.content_mask.bounds;
        let mut placement = [0; 16];
        placement[15] = u32::from(texture.options.linear_filtering);
        for (target, value) in placement.iter_mut().zip([
            bounds.origin.x.0,
            bounds.origin.y.0,
            bounds.size.width.0,
            bounds.size.height.0,
            clip.origin.x.0,
            clip.origin.y.0,
            clip.size.width.0,
            clip.size.height.0,
            texture.options.transform[0],
            texture.options.transform[1],
            texture.options.transform[2],
            texture.options.transform[3],
            texture.options.transform[4],
            texture.options.transform[5],
            texture.options.opacity,
        ]) {
            *target = value.to_bits();
        }
        Self {
            order: texture.order,
            placement,
            size: texture.texture.size(),
            revision: texture.options.revision,
            resource: texture.texture.resource_id(),
            service: texture.texture.service_id(),
        }
    }
}

#[derive(Default)]
pub(crate) struct DamageTracker {
    previous: Option<(OrdinaryScene, Vec<TextureStamp>)>,
    atlas_changed: bool,
}

impl DamageTracker {
    pub(crate) fn invalidate(&mut self) {
        self.previous = None;
    }

    pub(crate) fn invalidate_atlas(&mut self) {
        self.atlas_changed = true;
    }

    pub(crate) fn damage(
        &self,
        scene: &Scene,
        size: [u32; 2],
        appearance: u64,
    ) -> Result<Vec<RECT>> {
        let full = || {
            vec![RECT {
                left: 0,
                top: 0,
                right: size[0] as i32,
                bottom: size[1] as i32,
            }]
        };
        let Some((ordinary, previous)) = &self.previous else {
            return Ok(full());
        };
        if appearance != ordinary.appearance
            || !scene.surfaces.is_empty()
            || ordinary.has_surfaces
            || previous.len() != scene.gpu_textures.len()
        {
            return Ok(full());
        }
        let mut damage = ordinary.damage(scene, size);
        if self.atlas_changed {
            // Atlas allocation IDs can be reused with different image contents.
            // Redraw sprite bounds even when their scene metadata stayed equal.
            OrdinaryScene::sprite_damage(
                &ordinary.monochrome_sprites,
                &ordinary.subpixel_sprites,
                &ordinary.polychrome_sprites,
                size,
                &mut damage,
            );
            OrdinaryScene::sprite_damage(
                &scene.monochrome_sprites,
                &scene.subpixel_sprites,
                &scene.polychrome_sprites,
                size,
                &mut damage,
            );
        }
        for (old, texture) in previous.iter().zip(&scene.gpu_textures) {
            let current = TextureStamp::new(texture);
            if *old == current {
                continue;
            }
            if old.placement != current.placement
                || old.size != current.size
                || old.order != current.order
                || old.service != current.service
            {
                return Ok(full());
            }
            let is_successor = texture.options.previous_revision == Some(old.revision);
            match (&texture.options.damage, is_successor) {
                (GpuDamage::None, true) => {}
                (GpuDamage::Rects(rectangles), true) => {
                    for rectangle in rectangles.iter() {
                        anyhow::ensure!(
                            rectangle.fits(texture.texture.size()),
                            "GPU texture damage exceeds the image extent"
                        );
                        if let Some(rectangle) = texture_rectangle(texture, *rectangle, size) {
                            damage.push(rectangle);
                        }
                    }
                }
                _ => {
                    let [width, height] = texture.texture.size();
                    if let Some(rectangle) = texture_rectangle(
                        texture,
                        gpui::GpuRect {
                            x: 0,
                            y: 0,
                            width,
                            height,
                        },
                        size,
                    ) {
                        damage.push(rectangle);
                    }
                }
            }
        }
        // A single union avoids repeated alpha blending over intersecting regions.
        Ok(union_damage(&damage).into_iter().collect())
    }

    pub(crate) fn commit(&mut self, scene: &Scene, appearance: u64) -> Result<()> {
        self.atlas_changed = false;
        if let Some((ordinary, textures)) = &mut self.previous {
            ordinary.update(scene, appearance);
            textures.clear();
            textures.extend(scene.gpu_textures.iter().map(TextureStamp::new));
        } else {
            self.previous = Some((
                OrdinaryScene::new(scene, appearance),
                scene.gpu_textures.iter().map(TextureStamp::new).collect(),
            ));
        }
        Ok(())
    }
}

fn union_damage(rectangles: &[RECT]) -> Option<RECT> {
    let mut result = *rectangles.first()?;
    for rectangle in &rectangles[1..] {
        result.left = result.left.min(rectangle.left);
        result.top = result.top.min(rectangle.top);
        result.right = result.right.max(rectangle.right);
        result.bottom = result.bottom.max(rectangle.bottom);
    }
    Some(result)
}

fn texture_rectangle(
    texture: &PaintGpuTexture,
    rectangle: gpui::GpuRect,
    viewport: [u32; 2],
) -> Option<RECT> {
    let size = texture.texture.size();
    let destination = texture.destination;
    let [a, b, c, d, tx, ty] = texture.options.transform;
    let support = u32::from(texture.options.linear_filtering);
    let left = rectangle.x.saturating_sub(support);
    let top = rectangle.y.saturating_sub(support);
    let right = rectangle
        .x
        .saturating_add(rectangle.width)
        .saturating_add(support)
        .min(size[0]);
    let bottom = rectangle
        .y
        .saturating_add(rectangle.height)
        .saturating_add(support)
        .min(size[1]);
    let mut minimum = [f32::INFINITY; 2];
    let mut maximum = [f32::NEG_INFINITY; 2];
    for (x, y) in [(left, top), (right, top), (left, bottom), (right, bottom)] {
        let x = x as f32 / size[0] as f32 * destination.size.width.0;
        let y = y as f32 / size[1] as f32 * destination.size.height.0;
        let point = [
            destination.origin.x.0 + a * x + c * y + tx,
            destination.origin.y.0 + b * x + d * y + ty,
        ];
        for axis in 0..2 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    let clip = texture.content_mask.bounds;
    let result = RECT {
        left: (minimum[0] - 1.0).floor().max(clip.origin.x.0).max(0.0) as i32,
        top: (minimum[1] - 1.0).floor().max(clip.origin.y.0).max(0.0) as i32,
        right: (maximum[0] + 1.0)
            .ceil()
            .min(clip.origin.x.0 + clip.size.width.0)
            .min(viewport[0] as f32)
            .ceil() as i32,
        bottom: (maximum[1] + 1.0)
            .ceil()
            .min(clip.origin.y.0 + clip.size.height.0)
            .min(viewport[1] as f32)
            .ceil() as i32,
    };
    (result.left < result.right && result.top < result.bottom).then_some(result)
}

#[derive(Clone)]
struct OrdinaryScene {
    appearance: u64,
    has_surfaces: bool,
    shadows: Vec<gpui::Shadow>,
    quads: Vec<gpui::Quad>,
    paths: Vec<gpui::Path<gpui::ScaledPixels>>,
    underlines: Vec<gpui::Underline>,
    monochrome_sprites: Vec<gpui::MonochromeSprite>,
    subpixel_sprites: Vec<gpui::SubpixelSprite>,
    polychrome_sprites: Vec<gpui::PolychromeSprite>,
}

impl OrdinaryScene {
    fn sprite_damage(
        monochrome: &[gpui::MonochromeSprite],
        subpixel: &[gpui::SubpixelSprite],
        polychrome: &[gpui::PolychromeSprite],
        viewport: [u32; 2],
        damage: &mut Vec<RECT>,
    ) {
        for item in monochrome {
            damage.extend(clipped_rectangle(
                transformed_bounds(item.bounds, item.transformation),
                item.content_mask.bounds,
                viewport,
            ));
        }
        for item in subpixel {
            damage.extend(clipped_rectangle(
                transformed_bounds(item.bounds, item.transformation),
                item.content_mask.bounds,
                viewport,
            ));
        }
        for item in polychrome {
            damage.extend(clipped_rectangle(
                item.bounds,
                item.content_mask.bounds,
                viewport,
            ));
        }
    }

    fn new(scene: &Scene, appearance: u64) -> Self {
        Self {
            appearance,
            has_surfaces: !scene.surfaces.is_empty(),
            shadows: scene.shadows.clone(),
            quads: scene.quads.clone(),
            paths: scene.paths.clone(),
            underlines: scene.underlines.clone(),
            monochrome_sprites: scene.monochrome_sprites.clone(),
            subpixel_sprites: scene.subpixel_sprites.clone(),
            polychrome_sprites: scene.polychrome_sprites.clone(),
        }
    }

    fn update(&mut self, scene: &Scene, appearance: u64) {
        self.appearance = appearance;
        self.has_surfaces = !scene.surfaces.is_empty();
        macro_rules! update {
            ($field:ident) => {
                if self.$field != scene.$field {
                    self.$field.clone_from(&scene.$field);
                }
            };
        }
        update!(shadows);
        update!(quads);
        update!(paths);
        update!(underlines);
        update!(monochrome_sprites);
        update!(subpixel_sprites);
        update!(polychrome_sprites);
    }

    fn damage(&self, scene: &Scene, viewport: [u32; 2]) -> Vec<RECT> {
        let mut damage = Vec::new();
        compare_primitives(
            &self.shadows,
            &scene.shadows,
            |item| {
                let bounds = if item.inset != 0 {
                    item.element_bounds
                } else {
                    item.bounds
                        .dilate(gpui::ScaledPixels(3.0 * item.blur_radius.0))
                };
                clipped_rectangle(bounds, item.content_mask.bounds, viewport)
            },
            &mut damage,
        );
        compare_primitives(
            &self.quads,
            &scene.quads,
            |item| clipped_rectangle(item.bounds, item.content_mask.bounds, viewport),
            &mut damage,
        );
        compare_primitives(
            &self.paths,
            &scene.paths,
            |item| clipped_rectangle(item.bounds, item.content_mask.bounds, viewport),
            &mut damage,
        );
        compare_primitives(
            &self.underlines,
            &scene.underlines,
            |item| clipped_rectangle(item.bounds, item.content_mask.bounds, viewport),
            &mut damage,
        );
        compare_primitives(
            &self.monochrome_sprites,
            &scene.monochrome_sprites,
            |item| {
                clipped_rectangle(
                    transformed_bounds(item.bounds, item.transformation),
                    item.content_mask.bounds,
                    viewport,
                )
            },
            &mut damage,
        );
        compare_primitives(
            &self.subpixel_sprites,
            &scene.subpixel_sprites,
            |item| {
                clipped_rectangle(
                    transformed_bounds(item.bounds, item.transformation),
                    item.content_mask.bounds,
                    viewport,
                )
            },
            &mut damage,
        );
        compare_primitives(
            &self.polychrome_sprites,
            &scene.polychrome_sprites,
            |item| clipped_rectangle(item.bounds, item.content_mask.bounds, viewport),
            &mut damage,
        );
        damage
    }
}

fn compare_primitives<T: PartialEq>(
    old: &[T],
    new: &[T],
    bounds: impl Fn(&T) -> Option<RECT>,
    damage: &mut Vec<RECT>,
) {
    for index in 0..old.len().max(new.len()) {
        let previous = old.get(index);
        let current = new.get(index);
        if previous == current {
            continue;
        }
        damage.extend(previous.and_then(&bounds));
        damage.extend(current.and_then(&bounds));
    }
}

fn clipped_rectangle(
    bounds: gpui::Bounds<gpui::ScaledPixels>,
    clip: gpui::Bounds<gpui::ScaledPixels>,
    viewport: [u32; 2],
) -> Option<RECT> {
    let rectangle = RECT {
        left: (bounds.origin.x.0 - 1.0)
            .floor()
            .max(clip.origin.x.0)
            .max(0.0) as i32,
        top: (bounds.origin.y.0 - 1.0)
            .floor()
            .max(clip.origin.y.0)
            .max(0.0) as i32,
        right: (bounds.origin.x.0 + bounds.size.width.0 + 1.0)
            .ceil()
            .min(clip.origin.x.0 + clip.size.width.0)
            .min(viewport[0] as f32)
            .ceil() as i32,
        bottom: (bounds.origin.y.0 + bounds.size.height.0 + 1.0)
            .ceil()
            .min(clip.origin.y.0 + clip.size.height.0)
            .min(viewport[1] as f32)
            .ceil() as i32,
    };
    (rectangle.left < rectangle.right && rectangle.top < rectangle.bottom).then_some(rectangle)
}

fn transformed_bounds(
    bounds: gpui::Bounds<gpui::ScaledPixels>,
    transform: gpui::TransformationMatrix,
) -> gpui::Bounds<gpui::ScaledPixels> {
    let mut minimum = [f32::INFINITY; 2];
    let mut maximum = [f32::NEG_INFINITY; 2];
    for (x, y) in [
        (0.0, 0.0),
        (bounds.size.width.0, 0.0),
        (0.0, bounds.size.height.0),
        (bounds.size.width.0, bounds.size.height.0),
    ] {
        let point = transform.apply(gpui::point(
            gpui::px(bounds.origin.x.0 + x),
            gpui::px(bounds.origin.y.0 + y),
        ));
        for (axis, value) in [f32::from(point.x), f32::from(point.y)]
            .into_iter()
            .enumerate()
        {
            minimum[axis] = minimum[axis].min(value);
            maximum[axis] = maximum[axis].max(value);
        }
    }
    gpui::Bounds::new(
        gpui::point(
            gpui::ScaledPixels(minimum[0]),
            gpui::ScaledPixels(minimum[1]),
        ),
        gpui::size(
            gpui::ScaledPixels(maximum[0] - minimum[0]),
            gpui::ScaledPixels(maximum[1] - minimum[1]),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{Bounds, ContentMask, GpuRect, GpuTextureOptions, ScaledPixels, point, size};

    #[derive(Debug)]
    struct TestTexture(u64);
    impl gpui::GpuTexture for TestTexture {
        fn service_id(&self) -> u64 {
            1
        }
        fn epoch(&self) -> u64 {
            1
        }
        fn resource_id(&self) -> u64 {
            self.0
        }
        fn size(&self) -> [u32; 2] {
            [16, 16]
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn texture_scene(
        resource: u64,
        revision: u64,
        previous_revision: Option<u64>,
        damage: GpuDamage,
    ) -> Scene {
        let bounds = Bounds::new(
            point(ScaledPixels(20.0), ScaledPixels(30.0)),
            size(ScaledPixels(160.0), ScaledPixels(160.0)),
        );
        let mut scene = Scene::default();
        scene.gpu_textures.push(PaintGpuTexture {
            order: 0,
            bounds,
            destination: bounds,
            content_mask: ContentMask { bounds },
            texture: std::sync::Arc::new(TestTexture(resource)),
            options: GpuTextureOptions {
                revision,
                previous_revision,
                damage,
                ..Default::default()
            },
        });
        scene
    }

    #[test]
    fn fractional_clips_keep_the_last_covered_pixel_in_damage() {
        let bounds = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(20.0), ScaledPixels(20.0)),
        );
        let clip = Bounds::new(
            point(ScaledPixels(2.25), ScaledPixels(3.25)),
            size(ScaledPixels(5.5), ScaledPixels(6.5)),
        );
        let ordinary = clipped_rectangle(bounds, clip, [20, 20]).expect("ordinary damage");
        assert_eq!(
            (ordinary.left, ordinary.top, ordinary.right, ordinary.bottom),
            (2, 3, 8, 10)
        );
        let mut scene = texture_scene(1, 1, None, GpuDamage::Full);
        let texture = &mut scene.gpu_textures[0];
        texture.destination = bounds;
        texture.content_mask.bounds = clip;
        let damage = texture_rectangle(
            texture,
            GpuRect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            [20, 20],
        )
        .expect("texture damage");
        assert_eq!(
            (damage.left, damage.top, damage.right, damage.bottom),
            (2, 3, 8, 10)
        );
    }

    #[test]
    fn disjoint_damage_includes_the_gap_for_complete_recomposition() {
        let result = union_damage(&[
            RECT {
                left: 2,
                top: 3,
                right: 5,
                bottom: 8,
            },
            RECT {
                left: 11,
                top: 1,
                right: 13,
                bottom: 4,
            },
        ])
        .expect("damage");
        assert_eq!(
            (result.left, result.top, result.right, result.bottom),
            (2, 1, 13, 8)
        );
    }

    #[test]
    fn unchanged_scene_skips_and_ordinary_changes_invalidate() -> Result<()> {
        let mut tracker = DamageTracker::default();
        let mut scene = Scene::default();
        assert_eq!(tracker.damage(&scene, [80, 60], 0)?.len(), 1);
        tracker.commit(&scene, 0)?;
        assert!(tracker.damage(&scene, [80, 60], 0)?.is_empty());
        let bounds = Bounds::new(
            point(ScaledPixels(8.0), ScaledPixels(9.0)),
            size(ScaledPixels(4.0), ScaledPixels(5.0)),
        );
        let clip = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(80.0), ScaledPixels(60.0)),
        );
        scene.quads.push(gpui::Quad {
            bounds,
            content_mask: ContentMask { bounds: clip },
            ..Default::default()
        });
        let damage = tracker.damage(&scene, [80, 60], 0)?;
        assert_eq!(damage.len(), 1);
        assert_eq!(
            (
                damage[0].left,
                damage[0].top,
                damage[0].right,
                damage[0].bottom
            ),
            (7, 8, 13, 15)
        );
        tracker.commit(&scene, 0)?;
        scene.quads.clear();
        let removed = tracker.damage(&scene, [80, 60], 0)?;
        assert_eq!(
            (
                removed[0].left,
                removed[0].top,
                removed[0].right,
                removed[0].bottom
            ),
            (7, 8, 13, 15)
        );
        Ok(())
    }

    #[test]
    fn shadow_damage_includes_its_gaussian_tail() -> Result<()> {
        let mut tracker = DamageTracker::default();
        let mut scene = Scene::default();
        tracker.commit(&scene, 0)?;
        let bounds = Bounds::new(
            point(ScaledPixels(40.0), ScaledPixels(40.0)),
            size(ScaledPixels(20.0), ScaledPixels(20.0)),
        );
        let clip = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(100.0), ScaledPixels(100.0)),
        );
        scene.shadows.push(gpui::Shadow {
            order: 0,
            blur_radius: ScaledPixels(4.0),
            bounds,
            content_mask: ContentMask { bounds: clip },
            color: gpui::black(),
            corner_radii: Default::default(),
            element_bounds: bounds,
            element_corner_radii: Default::default(),
            inset: 0,
            pad: 0,
        });
        let damage = tracker.damage(&scene, [100, 100], 0)?;
        assert_eq!(
            (
                damage[0].left,
                damage[0].top,
                damage[0].right,
                damage[0].bottom
            ),
            (27, 27, 73, 73)
        );
        Ok(())
    }

    #[test]
    fn replacement_texture_damage_expands_filter_support_before_scaling() -> Result<()> {
        let mut tracker = DamageTracker::default();
        tracker.commit(&texture_scene(1, 10, None, GpuDamage::Full), 0)?;
        let scene = texture_scene(
            2,
            11,
            Some(10),
            GpuDamage::Rects(std::sync::Arc::from([GpuRect {
                x: 4,
                y: 5,
                width: 1,
                height: 1,
            }])),
        );
        let damage = tracker.damage(&scene, [400, 300], 0)?;
        assert_eq!(damage.len(), 1);
        assert!(damage[0].left <= 50 && damage[0].right >= 80);
        assert!(damage[0].top <= 70 && damage[0].bottom >= 100);
        assert!(damage[0].right < 400 && damage[0].bottom < 300);
        tracker.commit(&scene, 0)?;
        assert!(tracker.damage(&scene, [400, 300], 0)?.is_empty());
        Ok(())
    }

    #[test]
    fn skipped_revision_cannot_reuse_incomplete_damage() -> Result<()> {
        let mut tracker = DamageTracker::default();
        tracker.commit(&texture_scene(1, 10, None, GpuDamage::Full), 0)?;
        let scene = texture_scene(2, 12, Some(11), GpuDamage::None);
        let damage = tracker.damage(&scene, [400, 300], 0)?;
        assert_eq!(damage.len(), 1);
        assert_eq!(
            (
                damage[0].left,
                damage[0].top,
                damage[0].right,
                damage[0].bottom
            ),
            (20, 30, 180, 190)
        );
        Ok(())
    }

    #[test]
    fn native_compositing_preserves_premultiplied_alpha_transform_and_clip() -> Result<()> {
        let host = crate::gpu_extensions::OffscreenGpuHost::new()?;
        let service = host.service();
        let native = service
            .as_any()
            .downcast_ref::<DirectXGpuService>()
            .context("native GPU service")?;
        let source = native.with_context(|context| {
            let pixels = [255u8, 0, 0, 255];
            let data = D3D11_SUBRESOURCE_DATA {
                pSysMem: pixels.as_ptr().cast(),
                SysMemPitch: 4,
                SysMemSlicePitch: 0,
            };
            let mut source = None;
            unsafe {
                context.device.CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Width: 1,
                        Height: 1,
                        MipLevels: 1,
                        ArraySize: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Usage: D3D11_USAGE_DEFAULT,
                        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                        ..Default::default()
                    },
                    Some(&data),
                    Some(&mut source),
                )
            }?;
            source.context("source texture")
        })?;
        let source = native.wrap_texture(source)?;
        let bounds = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(8.0), ScaledPixels(8.0)),
        );
        let clip = Bounds::new(
            point(ScaledPixels(2.0), ScaledPixels(1.0)),
            size(ScaledPixels(4.0), ScaledPixels(3.0)),
        );
        let output = native.with_context(|context| {
            let target = RetainedTarget::new(context.device, [9, 7])?;
            let renderer = ExtensionRenderer::new(context.device)?;
            unsafe {
                context
                    .immediate_context
                    .ClearRenderTargetView(&target.view, &[0.0; 4]);
                context
                    .immediate_context
                    .OMSetRenderTargets(Some(&[Some(target.view.clone())]), None);
                context
                    .immediate_context
                    .RSSetViewports(Some(&[D3D11_VIEWPORT {
                        Width: 9.0,
                        Height: 7.0,
                        MaxDepth: 1.0,
                        ..Default::default()
                    }]));
                context.immediate_context.RSSetState(&renderer.rasterizer);
                context.immediate_context.RSSetScissorRects(Some(&[RECT {
                    left: 0,
                    top: 0,
                    right: 9,
                    bottom: 7,
                }]));
            }
            renderer.pipeline.draw(
                context.immediate_context,
                native,
                &PaintGpuTexture {
                    order: 0,
                    bounds,
                    destination: bounds,
                    content_mask: ContentMask { bounds: clip },
                    texture: source,
                    options: GpuTextureOptions {
                        opacity: 0.5,
                        transform: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        ..Default::default()
                    },
                },
                [9, 7],
            )?;
            Ok(target.texture)
        })?;
        let output = native.wrap_texture(output)?;
        let mut readback = service.readback(
            &output,
            GpuRect {
                x: 0,
                y: 0,
                width: 9,
                height: 7,
            },
        )?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let image = loop {
            if let Some(image) = readback.poll()? {
                break image;
            }
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "compositing readback timed out"
            );
            std::thread::yield_now();
        };
        for y in 0..7usize {
            for x in 0..9usize {
                let pixel = &image.pixels[(y * 9 + x) * 4..(y * 9 + x + 1) * 4];
                if (2..6).contains(&x) && (1..4).contains(&y) {
                    assert!(
                        (127..=128).contains(&pixel[0])
                            && pixel[1] == 0
                            && pixel[2] == 0
                            && (127..=128).contains(&pixel[3]),
                        "{x},{y}: {pixel:?}"
                    );
                } else {
                    assert_eq!(pixel, &[0; 4], "{x},{y}");
                }
            }
        }
        Ok(())
    }
}
