//! Opt-in interfaces for producers that render with GPUI's GPU device.
//!
//! Producers prepare complete frame candidates before publishing textures. Texture
//! revisions are immutable while referenced by a scene: double buffering or retained
//! tiles must be used when a subsequent frame changes their contents.

use std::{any::Any, fmt::Debug, sync::Arc};

use anyhow::Result;

/// An integer rectangle in texture pixels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuRect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl GpuRect {
    /// Returns whether this is a nonempty, in-bounds texture region.
    pub fn fits(self, size: [u32; 2]) -> bool {
        self.width > 0
            && self.height > 0
            && self
                .x
                .checked_add(self.width)
                .is_some_and(|end| end <= size[0])
            && self
                .y
                .checked_add(self.height)
                .is_some_and(|end| end <= size[1])
    }
}

/// Damage relative to the previous successfully presented revision of a texture.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum GpuDamage {
    /// All pixels may have changed.
    #[default]
    Full,
    /// No pixels changed.
    None,
    /// These texture-pixel rectangles changed. Include filter support and old bounds.
    Rects(Arc<[GpuRect]>),
}

/// A backend-owned image, created on the GPU service backing the target window.
pub trait GpuTexture: Any + Debug + Send + Sync {
    /// Identity of the owning GPU service.
    fn service_id(&self) -> u64;
    /// Device generation; resources from an earlier generation cannot be used.
    fn epoch(&self) -> u64;
    /// Stable identity of this allocation within its service.
    fn resource_id(&self) -> u64;
    /// Texture dimensions in pixels.
    fn size(&self) -> [u32; 2];
    /// Backend-specific resource access.
    fn as_any(&self) -> &dyn Any;
}

/// An owned readback with tightly packed RGBA8 pixels, preserving premultiplied alpha.
#[derive(Debug)]
pub struct GpuReadbackImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA bytes, without staging-row padding.
    pub pixels: Vec<u8>,
}

/// A nonblocking readback. Poll on the GPU service's owning thread.
pub trait GpuReadback: Send {
    /// Returns pixels when complete, without waiting for the GPU.
    fn poll(&mut self) -> Result<Option<GpuReadbackImage>>;
    /// Reuse completed staging storage for another copy. Returns false when the
    /// backend or dimensions/format are incompatible; never discards the old handle.
    /// The caller must retain any allocation-budget lease for this handle's lifetime.
    fn restart(&mut self, _texture: &Arc<dyn GpuTexture>, _region: GpuRect) -> Result<bool> {
        Ok(false)
    }
}

/// Completion of previously submitted work, polled without a CPU/GPU wait.
pub trait GpuCompletion: Send {
    /// Returns true once all work preceding this marker has retired.
    fn poll(&mut self) -> Result<bool>;
}

/// A unit of GPU preparation, executed in order on the service owner thread.
pub trait GpuJob {
    /// Prepare GPU resources. Errors prevent the caller from publishing a candidate.
    fn execute(&mut self, service: &dyn GpuService) -> Result<()>;
}

/// Cumulative device-service diagnostics for rendering and export measurements.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuServiceDiagnostics {
    /// Successful window presentations.
    pub presentations: u64,
    /// Presentation requests whose output was already current.
    pub skipped_presentations: u64,
    /// Presentations that reconstructed the entire window.
    pub full_presentations: u64,
    /// Total pixels submitted in presentation damage rectangles.
    pub presented_pixels: u64,
    /// Staging bytes reserved by unfinished readbacks.
    pub pending_readback_bytes: u64,
    /// RGBA payload bytes submitted through the portable upload API, excluding row padding.
    /// Native adapter uploads are measured by their caller.
    pub uploaded_bytes: u64,
    /// RGBA payload bytes submitted for readback, including subsequently cancelled copies.
    pub readback_bytes: u64,
    /// Native staging allocations created for readback.
    pub readback_allocations: u64,
    /// Copies submitted by restarting compatible completed readbacks.
    pub reused_readbacks: u64,
    /// Cumulative CPU time encoding successful window compositions, in nanoseconds.
    pub compositor_cpu_ns: u64,
    /// Cumulative CPU time inside successful Present1 calls, in nanoseconds.
    /// This is submission time, not GPU execution time or display scanout latency.
    pub present_cpu_ns: u64,
    /// Monotonic time when the latest successful Present1 submission returned.
    /// This is suitable for correlating native input dispatch with submission.
    pub last_presentation_end: Option<std::time::Instant>,
}

/// Native implementation of a GPU device service.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GpuBackend {
    /// Direct3D 11 on Windows.
    Direct3D11,
}

/// Immutable limits for the actual adapter backing a GPU service.
#[derive(Clone, Debug)]
pub struct GpuCapabilities {
    /// Native implementation used by this service.
    pub backend: GpuBackend,
    /// Operating-system adapter description.
    pub adapter_name: Arc<str>,
    /// Native adapter vendor identifier, when available.
    pub vendor_id: Option<u32>,
    /// Native adapter device identifier, when available.
    pub device_id: Option<u32>,
    /// Maximum width or height of a two-dimensional texture.
    pub max_texture_size: u32,
    /// Supported RGBA8 UNORM sample counts, including single sampling.
    pub rgba8_sample_counts: Arc<[u32]>,
    /// Whether the device supports compute shader execution.
    pub compute_shaders: bool,
    /// Maximum bytes in a single tightly packed RGBA8 texture upload.
    /// Actual allocations remain subject to available memory.
    pub max_upload_bytes: u64,
    /// Aggregate staging bytes permitted for unfinished readbacks.
    pub max_pending_readback_bytes: u64,
}

/// GPUI's GPU device service, also available to offscreen applications.
pub trait GpuService: Any + Send + Sync {
    /// Stable service identity.
    fn service_id(&self) -> u64;
    /// Current device generation.
    fn epoch(&self) -> u64;
    /// Backend-specific scoped execution adapter.
    fn as_any(&self) -> &dyn Any;
    /// Inspect adapter identity and limits without submitting GPU work.
    fn capabilities(&self) -> &GpuCapabilities;
    /// Read device-service counters without submitting GPU work.
    fn diagnostics(&self) -> GpuServiceDiagnostics {
        GpuServiceDiagnostics::default()
    }
    /// Allocate a transparent, shader-readable and renderable RGBA8 UNORM image.
    fn create_rgba8_texture(&self, _size: [u32; 2]) -> Result<Arc<dyn GpuTexture>> {
        anyhow::bail!("GPU texture creation is unsupported by this backend")
    }
    /// Upload premultiplied RGBA8 pixels into a uniquely owned image.
    ///
    /// Published scene resources cannot be modified. The mutable Arc must have no
    /// other strong or weak owners, and callers must retire prior GPU use first.
    fn upload_rgba8_region(
        &self,
        _texture: &mut Arc<dyn GpuTexture>,
        _region: GpuRect,
        _bytes: &[u8],
        _row_stride: u32,
    ) -> Result<()> {
        anyhow::bail!("GPU region uploads are unsupported by this backend")
    }
    /// Insert a nonblocking completion marker after previous submissions.
    fn completion(&self) -> Result<Box<dyn GpuCompletion>> {
        anyhow::bail!("GPU completion markers are unsupported by this backend")
    }
    /// Retain an owned resource until earlier GPU work completes. May be called
    /// from any thread, including a final texture or memory-lease destructor.
    /// Device retirement releases outstanding payloads after epoch invalidation.
    fn defer_release(&self, resource: Box<dyn Any + Send>);
    /// Poll deferred releases on the owner thread without waiting for the GPU.
    /// Returns true while work remains. Native app platforms arrange polling;
    /// standalone offscreen hosts must pump it explicitly before teardown.
    fn poll_deferred(&self) -> Result<bool>;
    /// Execute a job on the owning thread, ordered after previous submissions.
    fn run_job(&self, job: &mut dyn GpuJob) -> Result<()>;
    /// Queue a bounded staging copy. Display rendering never needs this operation.
    fn readback(
        &self,
        texture: &Arc<dyn GpuTexture>,
        region: GpuRect,
    ) -> Result<Box<dyn GpuReadback>>;
}

/// Texture compositing parameters, independent of any application's scene model.
#[derive(Clone, Debug)]
pub struct GpuTextureOptions {
    /// Content revision published with this texture.
    pub revision: u64,
    /// Previous revision to which `damage` applies. A mismatch forces full damage.
    pub previous_revision: Option<u64>,
    /// Changed texture pixels since `previous_revision`.
    pub damage: GpuDamage,
    /// Premultiplied compositing opacity in the inclusive range zero to one.
    pub opacity: f32,
    /// Affine transform [a, b, c, d, tx, ty] in logical pixels, relative to bounds origin.
    pub transform: [f32; 6],
    /// Use linear filtering rather than nearest-neighbor sampling.
    pub linear_filtering: bool,
}

impl Default for GpuTextureOptions {
    fn default() -> Self {
        Self {
            revision: 0,
            previous_revision: None,
            damage: GpuDamage::Full,
            opacity: 1.0,
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            linear_filtering: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_reject_overflow_and_empty_extents() {
        assert!(
            !GpuRect {
                x: u32::MAX,
                y: 0,
                width: 2,
                height: 1
            }
            .fits([8, 8])
        );
        assert!(!GpuRect::default().fits([8, 8]));
        assert!(
            GpuRect {
                x: 2,
                y: 3,
                width: 6,
                height: 5
            }
            .fits([8, 8])
        );
    }

    #[derive(Debug)]
    struct TestTexture;

    impl GpuTexture for TestTexture {
        fn service_id(&self) -> u64 {
            1
        }
        fn epoch(&self) -> u64 {
            1
        }
        fn resource_id(&self) -> u64 {
            1
        }
        fn size(&self) -> [u32; 2] {
            [8, 8]
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn texture_batches_preserve_overlapping_scene_order_and_replay() {
        use crate::{
            Bounds, ContentMask, PaintGpuTexture, PrimitiveBatch, Quad, ScaledPixels, Scene, point,
            size,
        };
        let bounds = Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(8.0), ScaledPixels(8.0)),
        );
        let mask = ContentMask { bounds };
        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            bounds,
            content_mask: mask,
            ..Default::default()
        });
        scene.insert_primitive(PaintGpuTexture {
            order: 0,
            bounds,
            destination: bounds,
            content_mask: mask,
            texture: Arc::new(TestTexture),
            options: GpuTextureOptions::default(),
        });
        scene.insert_primitive(Quad {
            bounds,
            content_mask: mask,
            ..Default::default()
        });
        scene.finish();
        let mut batches = scene.batches();
        assert!(matches!(batches.next(), Some(PrimitiveBatch::Quads(_))));
        assert!(matches!(
            batches.next(),
            Some(PrimitiveBatch::GpuTextures(_))
        ));
        assert!(matches!(batches.next(), Some(PrimitiveBatch::Quads(_))));
        assert!(batches.next().is_none());
        let mut replay = Scene::default();
        replay.replay(0..scene.len(), &scene);
        replay.finish();
        assert_eq!(replay.gpu_textures.len(), 1);
        assert_eq!(replay.gpu_textures[0].order, scene.gpu_textures[0].order);
    }
}
