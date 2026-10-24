//! D3D11 implementation of GPUI's opt-in GPU extension contract.

use std::{
    any::Any,
    collections::VecDeque,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::ThreadId,
};

use anyhow::{Context, Result, ensure};
use futures::{StreamExt as _, channel::mpsc};
use gpui::{
    GpuBackend, GpuCapabilities, GpuCompletion, GpuJob, GpuReadback, GpuReadbackImage, GpuRect,
    GpuService, GpuServiceDiagnostics, GpuTexture,
};
use windows::Win32::Graphics::{
    Direct3D::D3D_FEATURE_LEVEL_11_0,
    Direct3D11::*,
    Dxgi::{Common::*, DXGI_ERROR_WAS_STILL_DRAWING, IDXGIDevice},
};
use windows::core::Interface as _;

static NEXT_SERVICE: AtomicU64 = AtomicU64::new(1);
static NEXT_RESOURCE: AtomicU64 = AtomicU64::new(1);
const MAX_READBACK_BYTES: u64 = 256 * 1024 * 1024;

/// Borrowed native device access, valid only within `with_context`.
pub struct DirectXGpuContext<'a> {
    /// GPUI's existing D3D11 device.
    pub device: &'a ID3D11Device,
    /// GPUI's serialized immediate context. Do not retain or use it on workers.
    pub immediate_context: &'a ID3D11DeviceContext,
}

/// A device service shared by GPUI windows and offscreen jobs on their owner thread.
pub struct DirectXGpuService {
    self_ref: Weak<Self>,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    owner: ThreadId,
    identity: u64,
    valid: Arc<AtomicBool>,
    executing: AtomicBool,
    readback_bytes: Arc<AtomicU64>,
    total_uploaded_bytes: AtomicU64,
    total_readback_bytes: AtomicU64,
    readback_allocations: AtomicU64,
    reused_readbacks: AtomicU64,
    presentations: AtomicU64,
    skipped_presentations: AtomicU64,
    full_presentations: AtomicU64,
    presented_pixels: AtomicU64,
    compositor_cpu_ns: AtomicU64,
    present_cpu_ns: AtomicU64,
    last_presentation_end: parking_lot::Mutex<Option<std::time::Instant>>,
    capabilities: GpuCapabilities,
    deferred: parking_lot::Mutex<DeferredReleases>,
    retirement_wake: parking_lot::Mutex<Option<mpsc::UnboundedSender<()>>>,
    retirement_wake_pending: AtomicBool,
}

#[derive(Default)]
struct DeferredReleases {
    pending: Vec<Box<dyn Any + Send>>,
    submitted: VecDeque<DeferredBatch>,
}

struct DeferredBatch {
    completion: Box<dyn GpuCompletion>,
    resources: Vec<Box<dyn Any + Send>>,
}

impl DirectXGpuService {
    pub(crate) fn new(device: ID3D11Device, context: ID3D11DeviceContext) -> Arc<Self> {
        Self::new_for_owner(device, context, std::thread::current().id())
    }

    pub(crate) fn new_for_owner(
        device: ID3D11Device,
        context: ID3D11DeviceContext,
        owner: ThreadId,
    ) -> Arc<Self> {
        let capabilities = device_capabilities(&device);
        Arc::new_cyclic(|self_ref| Self {
            self_ref: self_ref.clone(),
            device,
            context,
            owner,
            identity: NEXT_SERVICE.fetch_add(1, Ordering::Relaxed),
            valid: Arc::new(AtomicBool::new(true)),
            executing: AtomicBool::new(false),
            readback_bytes: Arc::new(AtomicU64::new(0)),
            total_uploaded_bytes: AtomicU64::new(0),
            total_readback_bytes: AtomicU64::new(0),
            readback_allocations: AtomicU64::new(0),
            reused_readbacks: AtomicU64::new(0),
            presentations: AtomicU64::new(0),
            skipped_presentations: AtomicU64::new(0),
            full_presentations: AtomicU64::new(0),
            presented_pixels: AtomicU64::new(0),
            compositor_cpu_ns: AtomicU64::new(0),
            present_cpu_ns: AtomicU64::new(0),
            last_presentation_end: parking_lot::Mutex::new(None),
            capabilities,
            deferred: Default::default(),
            retirement_wake: Default::default(),
            retirement_wake_pending: AtomicBool::new(false),
        })
    }

    pub(crate) fn owner_thread(&self) -> ThreadId {
        self.owner
    }

    fn check_owner(&self) -> Result<()> {
        ensure!(
            std::thread::current().id() == self.owner,
            "GPU work must execute on the GPUI device owner thread"
        );
        ensure!(
            self.valid.load(Ordering::Acquire),
            "GPU device generation has been retired"
        );
        unsafe { self.device.GetDeviceRemovedReason() }.context("GPUI GPU device was lost")
    }

    fn readback_source<'a>(
        &self,
        texture: &'a Arc<dyn GpuTexture>,
        region: GpuRect,
    ) -> Result<&'a DirectXGpuTexture> {
        self.check_owner()?;
        ensure!(
            texture.service_id() == self.identity && texture.epoch() == self.epoch(),
            "Readback texture belongs to another GPU device generation"
        );
        ensure!(
            region.fits(texture.size()),
            "Readback region is outside the texture"
        );
        let texture = texture
            .as_any()
            .downcast_ref::<DirectXGpuTexture>()
            .context("Unsupported native texture")?;
        texture.validate(self)?;
        ensure!(
            region.fits(texture.size),
            "Readback region is outside the native texture"
        );
        Ok(texture)
    }

    fn reserve_readback(&self, region: GpuRect) -> Result<ReadbackReservation> {
        let bytes = u64::from(region.width) * u64::from(region.height) * 4;
        self.readback_bytes.fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| pending.checked_add(bytes).filter(|total| *total <= MAX_READBACK_BYTES))
            .map_err(|_| anyhow::anyhow!("Readbacks exceed the 256 MiB in-flight limit; export smaller tiles or retire pending copies"))?;
        Ok(ReadbackReservation {
            counter: self.readback_bytes.clone(),
            bytes,
        })
    }

    fn copy_readback(
        &self,
        source: &DirectXGpuTexture,
        region: GpuRect,
        staging: &ID3D11Texture2D,
    ) {
        let region_box = D3D11_BOX {
            left: region.x,
            top: region.y,
            right: region.x + region.width,
            bottom: region.y + region.height,
            front: 0,
            back: 1,
        };
        unsafe {
            self.context.CopySubresourceRegion(
                staging,
                0,
                0,
                0,
                0,
                &source.texture,
                0,
                Some(&region_box),
            );
            // Offscreen jobs have no Present to submit their staging copy.
            self.context.Flush();
        }
        self.total_readback_bytes.fetch_add(
            u64::from(region.width) * u64::from(region.height) * 4,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn invalidate(&self) {
        self.valid.store(false, Ordering::Release);
        let deferred = std::mem::take(&mut *self.deferred.lock());
        self.retirement_wake.lock().take();
        // Payload destructors may enqueue another lease, including on this service.
        drop(deferred);
    }

    pub(crate) fn attach_retirement_polling(
        self: &Arc<Self>,
        foreground: &gpui::ForegroundExecutor,
        background: &gpui::BackgroundExecutor,
    ) {
        let (sender, mut receiver) = mpsc::unbounded();
        {
            let mut wake = self.retirement_wake.lock();
            if wake.is_some() {
                return;
            }
            *wake = Some(sender);
        }
        let service = Arc::downgrade(self);
        let background = background.clone();
        foreground
            .spawn(async move {
                while receiver.next().await.is_some() {
                    loop {
                        let Some(service) = service.upgrade() else {
                            return;
                        };
                        match service.poll_retired() {
                            Ok(false) => {
                                service
                                    .retirement_wake_pending
                                    .store(false, Ordering::Release);
                                if service.has_deferred()
                                    && !service.retirement_wake_pending.swap(true, Ordering::AcqRel)
                                {
                                    continue;
                                }
                                break;
                            }
                            Ok(true) => {}
                            Err(error) => {
                                log::error!("GPU resource retirement failed: {error:#}");
                                service
                                    .retirement_wake_pending
                                    .store(false, Ordering::Release);
                                break;
                            }
                        }
                        drop(service);
                        background.timer(std::time::Duration::from_millis(2)).await;
                    }
                }
            })
            .detach();
        if self.has_deferred() {
            self.wake_retirement();
        }
    }

    fn has_deferred(&self) -> bool {
        let deferred = self.deferred.lock();
        !deferred.pending.is_empty() || !deferred.submitted.is_empty()
    }

    fn wake_retirement(&self) {
        let sender = self.retirement_wake.lock().clone();
        if let Some(sender) = sender
            && !self.retirement_wake_pending.swap(true, Ordering::AcqRel)
            && sender.unbounded_send(()).is_err()
        {
            self.retirement_wake_pending.store(false, Ordering::Release);
        }
    }

    fn poll_retired(&self) -> Result<bool> {
        ensure!(
            std::thread::current().id() == self.owner,
            "Deferred GPU resources must be polled on their owner thread"
        );
        if !self.valid.load(Ordering::Acquire) {
            return Ok(false);
        }
        if let Err(error) = unsafe { self.device.GetDeviceRemovedReason() } {
            self.invalidate();
            return Err(error).context("GPU device lost while retiring resources");
        }
        let mut released = Vec::new();
        {
            let mut deferred = self.deferred.lock();
            while let Some(batch) = deferred.submitted.front_mut() {
                if !batch.completion.poll()? {
                    break;
                }
                if let Some(batch) = deferred.submitted.pop_front() {
                    released.extend(batch.resources);
                }
            }
            if !deferred.pending.is_empty() && deferred.submitted.len() < 2 {
                let completion = self.completion()?;
                let resources = std::mem::take(&mut deferred.pending);
                deferred.submitted.push_back(DeferredBatch {
                    completion,
                    resources,
                });
            }
        }
        drop(released);
        Ok(self.has_deferred())
    }

    pub(crate) fn record_presentation(
        &self,
        pixels: u64,
        full: bool,
        compositor_cpu_ns: u64,
        present_cpu_ns: u64,
        presentation_end: std::time::Instant,
    ) {
        self.presentations.fetch_add(1, Ordering::Relaxed);
        self.presented_pixels.fetch_add(pixels, Ordering::Relaxed);
        self.compositor_cpu_ns
            .fetch_add(compositor_cpu_ns, Ordering::Relaxed);
        self.present_cpu_ns
            .fetch_add(present_cpu_ns, Ordering::Relaxed);
        *self.last_presentation_end.lock() = Some(presentation_end);
        if full {
            self.full_presentations.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_unchanged_frame(&self) {
        self.skipped_presentations.fetch_add(1, Ordering::Relaxed);
    }

    /// Insert a nonblocking retirement marker after all previously submitted jobs.
    pub fn completion(&self) -> Result<Box<dyn GpuCompletion>> {
        self.check_owner()?;
        let mut query = None;
        unsafe {
            self.device.CreateQuery(
                &D3D11_QUERY_DESC {
                    Query: D3D11_QUERY_EVENT,
                    MiscFlags: 0,
                },
                Some(&mut query),
            )
        }?;
        let query = query.context("Creating GPU completion query")?;
        unsafe {
            self.context.End(&query);
            self.context.Flush();
        }
        Ok(Box::new(DirectXCompletion {
            query,
            context: self.context.clone(),
            device: self.device.clone(),
            owner: self.owner,
            valid: self.valid.clone(),
            complete: false,
        }))
    }

    /// Execute native preparation in submission order, without creating another device.
    ///
    /// Bind all state used by the job. GPUI restores its state before scene rendering.
    /// A job must not mutate a texture referenced by an already published scene.
    pub fn with_context<T>(
        &self,
        operation: impl FnOnce(DirectXGpuContext<'_>) -> Result<T>,
    ) -> Result<T> {
        self.check_owner()?;
        ensure!(
            !self.executing.swap(true, Ordering::AcqRel),
            "GPU execution scopes cannot be nested"
        );
        struct Scope<'a>(&'a DirectXGpuService);
        impl Drop for Scope<'_> {
            fn drop(&mut self) {
                unsafe {
                    self.0.context.ClearState();
                }
                self.0.executing.store(false, Ordering::Release);
                if self.0.has_deferred()
                    && let Err(error) = self.0.poll_retired()
                {
                    log::error!("GPU resource retirement failed: {error:#}");
                }
            }
        }
        let _scope = Scope(self);
        let result = operation(DirectXGpuContext {
            device: &self.device,
            immediate_context: &self.context,
        });
        self.check_owner()?;
        result
    }

    /// Adopt a shader-readable RGBA/BGRA8 texture created on this service's device.
    pub fn wrap_texture(&self, texture: ID3D11Texture2D) -> Result<Arc<dyn GpuTexture>> {
        self.check_owner()?;
        let resource_device = unsafe { texture.GetDevice() }?;
        ensure!(
            resource_device == self.device,
            "GPU texture belongs to another device"
        );
        let mut description = D3D11_TEXTURE2D_DESC::default();
        unsafe {
            texture.GetDesc(&mut description);
        }
        ensure!(
            matches!(
                description.Format,
                DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM
            ),
            "GPU texture must use premultiplied RGBA8 or BGRA8 UNORM"
        );
        ensure!(
            description.ArraySize == 1
                && description.SampleDesc.Count == 1
                && description.MipLevels == 1,
            "GPU texture must be a single-sample, single-level 2D image"
        );
        ensure!(
            description.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 != 0,
            "GPU texture requires shader-resource binding"
        );
        let mut view = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))
        }?;
        Ok(Arc::new(DirectXGpuTexture {
            texture,
            view: view.context("D3D11 did not return a texture view")?,
            identity: self.identity,
            resource: NEXT_RESOURCE.fetch_add(1, Ordering::Relaxed),
            size: [description.Width, description.Height],
            format: description.Format,
            valid: self.valid.clone(),
        }))
    }
}

fn device_capabilities(device: &ID3D11Device) -> GpuCapabilities {
    let description = device
        .cast::<IDXGIDevice>()
        .ok()
        .and_then(|device| unsafe { device.GetAdapter().ok()?.GetDesc().ok() });
    let adapter_name = description
        .as_ref()
        .map(|description| {
            let end = description
                .Description
                .iter()
                .position(|value| *value == 0)
                .unwrap_or(description.Description.len());
            String::from_utf16_lossy(&description.Description[..end])
        })
        .unwrap_or_else(|| "Unknown Direct3D adapter".into());
    let feature_level = unsafe { device.GetFeatureLevel() };
    let max_texture_size = if feature_level.0 >= D3D_FEATURE_LEVEL_11_0.0 {
        16384
    } else {
        8192
    };
    let mut samples = vec![1];
    for count in [2, 4, 8, 16, 32] {
        if unsafe { device.CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, count) }
            .is_ok_and(|quality_levels| quality_levels > 0)
        {
            samples.push(count);
        }
    }
    GpuCapabilities {
        backend: GpuBackend::Direct3D11,
        adapter_name: adapter_name.into(),
        vendor_id: description.as_ref().map(|description| description.VendorId),
        device_id: description.as_ref().map(|description| description.DeviceId),
        max_texture_size,
        rgba8_sample_counts: samples.into(),
        compute_shaders: feature_level.0 >= D3D_FEATURE_LEVEL_11_0.0,
        max_upload_bytes: u64::from(max_texture_size).pow(2) * 4,
        max_pending_readback_bytes: MAX_READBACK_BYTES,
    }
}

impl GpuService for DirectXGpuService {
    fn service_id(&self) -> u64 {
        self.identity
    }
    fn epoch(&self) -> u64 {
        if self.valid.load(Ordering::Acquire) {
            1
        } else {
            2
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }
    fn diagnostics(&self) -> GpuServiceDiagnostics {
        GpuServiceDiagnostics {
            presentations: self.presentations.load(Ordering::Relaxed),
            skipped_presentations: self.skipped_presentations.load(Ordering::Relaxed),
            full_presentations: self.full_presentations.load(Ordering::Relaxed),
            presented_pixels: self.presented_pixels.load(Ordering::Relaxed),
            pending_readback_bytes: self.readback_bytes.load(Ordering::Acquire),
            uploaded_bytes: self.total_uploaded_bytes.load(Ordering::Relaxed),
            readback_bytes: self.total_readback_bytes.load(Ordering::Relaxed),
            readback_allocations: self.readback_allocations.load(Ordering::Relaxed),
            reused_readbacks: self.reused_readbacks.load(Ordering::Relaxed),
            compositor_cpu_ns: self.compositor_cpu_ns.load(Ordering::Relaxed),
            present_cpu_ns: self.present_cpu_ns.load(Ordering::Relaxed),
            last_presentation_end: *self.last_presentation_end.lock(),
        }
    }
    fn create_rgba8_texture(&self, size: [u32; 2]) -> Result<Arc<dyn GpuTexture>> {
        ensure!(
            size.iter()
                .all(|extent| *extent > 0 && *extent <= self.capabilities.max_texture_size),
            "RGBA8 texture dimensions exceed this device's capabilities"
        );
        let texture = self.with_context(|context| {
            let mut texture = None;
            unsafe {
                context.device.CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Width: size[0],
                        Height: size[1],
                        MipLevels: 1,
                        ArraySize: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Usage: D3D11_USAGE_DEFAULT,
                        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0)
                            as u32,
                        ..Default::default()
                    },
                    None,
                    Some(&mut texture),
                )
            }?;
            let texture = texture.context("Creating RGBA8 texture")?;
            let mut view = None;
            unsafe {
                context
                    .device
                    .CreateRenderTargetView(&texture, None, Some(&mut view))
            }?;
            let view = view.context("Creating RGBA8 initialization target")?;
            unsafe {
                context
                    .immediate_context
                    .ClearRenderTargetView(&view, &[0.0; 4])
            };
            Ok(texture)
        })?;
        self.wrap_texture(texture)
    }
    fn upload_rgba8_region(
        &self,
        texture: &mut Arc<dyn GpuTexture>,
        region: GpuRect,
        bytes: &[u8],
        row_stride: u32,
    ) -> Result<()> {
        self.check_owner()?;
        ensure!(
            region.fits(texture.size()),
            "RGBA8 upload region exceeds the texture"
        );
        let row_bytes = region
            .width
            .checked_mul(4)
            .context("RGBA8 row size overflow")?;
        ensure!(
            row_stride >= row_bytes,
            "RGBA8 row stride is shorter than one row"
        );
        let required = u64::from(region.height - 1) * u64::from(row_stride) + u64::from(row_bytes);
        ensure!(
            required <= bytes.len() as u64,
            "RGBA8 upload data is shorter than its rows"
        );
        let texture =
            Arc::get_mut(texture).context("Cannot upload to a shared or published GPU texture")?;
        ensure!(
            Any::type_id(&*texture) == std::any::TypeId::of::<DirectXGpuTexture>(),
            "Region upload requires a directly owned native texture, not a forwarding wrapper"
        );
        let native = texture
            .as_any()
            .downcast_ref::<DirectXGpuTexture>()
            .context("Unsupported GPU upload backend")?;
        native.validate(self)?;
        ensure!(
            native.format == DXGI_FORMAT_R8G8B8A8_UNORM,
            "RGBA8 uploads require an RGBA8 target"
        );
        self.with_context(|context| {
            let bounds = D3D11_BOX {
                left: region.x,
                top: region.y,
                front: 0,
                right: region.x + region.width,
                bottom: region.y + region.height,
                back: 1,
            };
            unsafe {
                context.immediate_context.UpdateSubresource(
                    &native.texture,
                    0,
                    Some(&bounds),
                    bytes.as_ptr().cast(),
                    row_stride,
                    0,
                )
            };
            Ok(())
        })?;
        self.total_uploaded_bytes.fetch_add(
            u64::from(row_bytes) * u64::from(region.height),
            Ordering::Relaxed,
        );
        Ok(())
    }
    fn completion(&self) -> Result<Box<dyn GpuCompletion>> {
        DirectXGpuService::completion(self)
    }
    fn defer_release(&self, resource: Box<dyn Any + Send>) {
        let mut deferred = self.deferred.lock();
        if !self.valid.load(Ordering::Acquire) {
            drop(deferred);
            drop(resource);
            return;
        }
        deferred.pending.push(resource);
        drop(deferred);
        self.wake_retirement();
    }
    fn poll_deferred(&self) -> Result<bool> {
        self.poll_retired()
    }
    fn run_job(&self, job: &mut dyn GpuJob) -> Result<()> {
        self.check_owner()?;
        job.execute(self)
    }
    fn readback(
        &self,
        texture: &Arc<dyn GpuTexture>,
        region: GpuRect,
    ) -> Result<Box<dyn GpuReadback>> {
        let texture = self.readback_source(texture, region)?;
        let reservation = self.reserve_readback(region)?;
        let description = D3D11_TEXTURE2D_DESC {
            Width: region.width,
            Height: region.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: texture.format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..Default::default()
        };
        let mut staging = None;
        unsafe {
            self.device
                .CreateTexture2D(&description, None, Some(&mut staging))
        }?;
        let staging = staging.context("D3D11 did not return a staging texture")?;
        self.readback_allocations.fetch_add(1, Ordering::Relaxed);
        self.copy_readback(texture, region, &staging);
        Ok(Box::new(DirectXReadback {
            service: self.self_ref.clone(),
            staging: Some(staging),
            context: self.context.clone(),
            device: self.device.clone(),
            owner: self.owner,
            valid: self.valid.clone(),
            size: [region.width, region.height],
            bgra: texture.format == DXGI_FORMAT_B8G8R8A8_UNORM,
            format: texture.format,
            complete: false,
            reservation: Some(reservation),
        }))
    }
}

/// Native resource backing a portable GPU texture. Its contents must stay immutable
/// for the lifetime of every scene that references the published revision.
#[derive(Debug)]
pub struct DirectXGpuTexture {
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    identity: u64,
    resource: u64,
    size: [u32; 2],
    format: DXGI_FORMAT,
    valid: Arc<AtomicBool>,
}

impl DirectXGpuTexture {
    /// Native texture for copies and shader binding inside an owner-thread scope.
    pub fn texture(&self) -> &ID3D11Texture2D {
        &self.texture
    }
    /// Native shader-resource view.
    pub fn shader_resource_view(&self) -> &ID3D11ShaderResourceView {
        &self.view
    }
    pub(crate) fn validate(&self, service: &DirectXGpuService) -> Result<()> {
        ensure!(
            self.valid.load(Ordering::Acquire) && self.identity == service.identity,
            "GPU scene references a retired or foreign texture"
        );
        Ok(())
    }
}

impl GpuTexture for DirectXGpuTexture {
    fn service_id(&self) -> u64 {
        self.identity
    }
    fn epoch(&self) -> u64 {
        1
    }
    fn resource_id(&self) -> u64 {
        self.resource
    }
    fn size(&self) -> [u32; 2] {
        self.size
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct DirectXReadback {
    service: Weak<DirectXGpuService>,
    staging: Option<ID3D11Texture2D>,
    context: ID3D11DeviceContext,
    device: ID3D11Device,
    owner: ThreadId,
    valid: Arc<AtomicBool>,
    size: [u32; 2],
    bgra: bool,
    format: DXGI_FORMAT,
    complete: bool,
    reservation: Option<ReadbackReservation>,
}

impl Drop for DirectXReadback {
    fn drop(&mut self) {
        if !self.complete
            && let Some(staging) = self.staging.take()
            && let Some(service) = self.service.upgrade()
        {
            service.defer_release(Box::new((staging, self.reservation.take())));
        }
    }
}

struct ReadbackReservation {
    counter: Arc<AtomicU64>,
    bytes: u64,
}
impl Drop for ReadbackReservation {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

struct DirectXCompletion {
    query: ID3D11Query,
    context: ID3D11DeviceContext,
    device: ID3D11Device,
    owner: ThreadId,
    valid: Arc<AtomicBool>,
    complete: bool,
}

impl GpuCompletion for DirectXCompletion {
    fn poll(&mut self) -> Result<bool> {
        ensure!(
            std::thread::current().id() == self.owner,
            "Completion must be polled on its GPUI owner thread"
        );
        ensure!(
            self.valid.load(Ordering::Acquire),
            "Completion device generation was retired"
        );
        unsafe { self.device.GetDeviceRemovedReason() }?;
        if self.complete {
            return Ok(true);
        }
        let mut ready = 0u32;
        unsafe {
            self.context.GetData(
                &self.query,
                Some((&mut ready as *mut u32).cast()),
                4,
                D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
            )
        }?;
        self.complete = ready != 0;
        Ok(self.complete)
    }
}

impl GpuReadback for DirectXReadback {
    fn poll(&mut self) -> Result<Option<GpuReadbackImage>> {
        ensure!(!self.complete, "Readback result has already been consumed");
        ensure!(
            std::thread::current().id() == self.owner,
            "Readback must be polled on its GPUI owner thread"
        );
        ensure!(
            self.valid.load(Ordering::Acquire),
            "Readback device generation was retired"
        );
        unsafe { self.device.GetDeviceRemovedReason() }?;
        let staging = self
            .staging
            .as_ref()
            .context("Readback result has already been consumed")?;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        match unsafe {
            self.context.Map(
                staging,
                0,
                D3D11_MAP_READ,
                D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32,
                Some(&mut mapped),
            )
        } {
            Err(error) if error.code() == DXGI_ERROR_WAS_STILL_DRAWING => return Ok(None),
            result => result.context("Mapping GPU readback")?,
        }
        let row_bytes = self.size[0] as usize * 4;
        let mut pixels = vec![0; row_bytes * self.size[1] as usize];
        for (row, destination) in pixels.chunks_exact_mut(row_bytes).enumerate() {
            let source = unsafe {
                std::slice::from_raw_parts(
                    (mapped.pData as *const u8).add(row * mapped.RowPitch as usize),
                    row_bytes,
                )
            };
            destination.copy_from_slice(source);
        }
        unsafe {
            self.context.Unmap(staging, 0);
        }
        if self.bgra {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        self.complete = true;
        self.reservation.take();
        Ok(Some(GpuReadbackImage {
            width: self.size[0],
            height: self.size[1],
            pixels,
        }))
    }

    fn restart(&mut self, texture: &Arc<dyn GpuTexture>, region: GpuRect) -> Result<bool> {
        ensure!(self.complete, "Cannot restart a pending readback");
        let service = self
            .service
            .upgrade()
            .context("Readback device owner was released")?;
        let source = service.readback_source(texture, region)?;
        if self.size != [region.width, region.height] || self.format != source.format {
            return Ok(false);
        }
        let staging = self
            .staging
            .as_ref()
            .context("Readback staging texture missing")?;
        let reservation = service.reserve_readback(region)?;
        service.copy_readback(source, region, staging);
        service.reused_readbacks.fetch_add(1, Ordering::Relaxed);
        self.reservation = Some(reservation);
        self.complete = false;
        Ok(true)
    }
}

/// A standalone GPUI device owner for GPU exports/tests without a native window.
/// Windowed hosts must use `App::gpu_service` or `Window::gpu_service` instead.
pub struct OffscreenGpuHost {
    devices: crate::DirectXDevices,
}

impl OffscreenGpuHost {
    /// Create a GPUI-owned offscreen D3D11 device on the current thread.
    pub fn new() -> Result<Self> {
        Ok(Self {
            devices: crate::DirectXDevices::new()?,
        })
    }
    /// Obtain the offscreen host's device service.
    pub fn service(&self) -> Arc<dyn GpuService> {
        self.devices.gpu_service.clone()
    }
}

impl Drop for OffscreenGpuHost {
    fn drop(&mut self) {
        self.devices.gpu_service.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    struct Released(Arc<AtomicU64>);
    impl Drop for Released {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn generic_region_upload_checks_ownership_stride_and_retirement() -> Result<()> {
        let host = OffscreenGpuHost::new()?;
        let service = host.service();
        assert!(service.create_rgba8_texture([0, 4]).is_err());
        assert!(
            service
                .create_rgba8_texture([service.capabilities().max_texture_size + 1, 4])
                .is_err()
        );
        let mut texture = service.create_rgba8_texture([5, 4])?;
        let region = GpuRect {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        let bytes = [
            128, 0, 0, 128, 0, 64, 0, 64, 99, 99, 99, 99, 0, 0, 32, 32, 100, 100, 100, 100,
        ];
        let alias = texture.clone();
        assert!(
            service
                .upload_rgba8_region(&mut texture, region, &bytes, 12)
                .is_err()
        );
        drop(alias);
        let weak = Arc::downgrade(&texture);
        assert!(
            service
                .upload_rgba8_region(&mut texture, region, &bytes, 12)
                .is_err()
        );
        drop(weak);
        assert!(
            service
                .upload_rgba8_region(&mut texture, region, &bytes, 4)
                .is_err()
        );
        assert!(
            service
                .upload_rgba8_region(&mut texture, region, &bytes[..12], 12)
                .is_err()
        );
        assert_eq!(service.diagnostics().uploaded_bytes, 0);
        service.upload_rgba8_region(&mut texture, region, &bytes, 12)?;
        assert_eq!(service.diagnostics().uploaded_bytes, 16);
        let mut copy = service.readback(
            &texture,
            GpuRect {
                x: 0,
                y: 0,
                width: 5,
                height: 4,
            },
        )?;
        assert_eq!(service.diagnostics().readback_bytes, 80);
        let deadline = Instant::now() + Duration::from_secs(10);
        let image = loop {
            if let Some(image) = copy.poll()? {
                break image;
            }
            ensure!(
                Instant::now() < deadline,
                "region upload readback timed out"
            );
            std::thread::yield_now();
        };
        for y in 0..4 {
            for x in 0..5 {
                let expected: &[u8] = match (x, y) {
                    (1, 1) => &bytes[..4],
                    (2, 1) => &bytes[4..8],
                    (1, 2) => &bytes[12..16],
                    (2, 2) => &bytes[16..20],
                    _ => &[0; 4],
                };
                assert_eq!(
                    &image.pixels[(y * 5 + x) * 4..(y * 5 + x + 1) * 4],
                    expected
                );
            }
        }
        let released = Arc::new(AtomicU64::new(0));
        let worker_service = service.clone();
        let worker_released = released.clone();
        std::thread::spawn(move || {
            worker_service.defer_release(Box::new((texture, Released(worker_released))))
        })
        .join()
        .map_err(|_| anyhow::anyhow!("retirement worker panicked"))?;
        assert_eq!(released.load(Ordering::Relaxed), 0);
        while service.poll_deferred()? {
            ensure!(Instant::now() < deadline, "deferred release timed out");
            assert!(host.devices.gpu_service.deferred.lock().submitted.len() <= 2);
            std::thread::yield_now();
        }
        assert_eq!(released.load(Ordering::Relaxed), 1);
        let mut marker = service.completion()?;
        while !marker.poll()? {
            ensure!(Instant::now() < deadline, "portable completion timed out");
            std::thread::yield_now();
        }
        service.defer_release(Box::new(Released(released.clone())));
        drop(host);
        assert_eq!(released.load(Ordering::Relaxed), 2);
        service.defer_release(Box::new(Released(released.clone())));
        assert_eq!(released.load(Ordering::Relaxed), 3);
        Ok(())
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn foreground_retirement_survives_worker_and_recursive_payload_drops() -> Result<()> {
        struct Followup {
            service: std::sync::Weak<DirectXGpuService>,
            released: Arc<AtomicU64>,
        }
        impl Drop for Followup {
            fn drop(&mut self) {
                self.released.fetch_add(1, Ordering::Relaxed);
                if let Some(service) = self.service.upgrade() {
                    service.defer_release(Box::new(Released(self.released.clone())));
                }
            }
        }
        let host = OffscreenGpuHost::new()?;
        let service = host.devices.gpu_service.clone();
        let dispatcher = gpui::TestDispatcher::new(127);
        let foreground = gpui::ForegroundExecutor::new(Arc::new(dispatcher.clone()));
        let background = gpui::BackgroundExecutor::new(Arc::new(dispatcher.clone()));
        service.attach_retirement_polling(&foreground, &background);
        let released = Arc::new(AtomicU64::new(0));
        let payload = Followup {
            service: Arc::downgrade(&service),
            released: released.clone(),
        };
        let worker = service.clone();
        std::thread::spawn(move || worker.defer_release(Box::new(payload)))
            .join()
            .map_err(|_| anyhow::anyhow!("retirement worker panicked"))?;
        let deadline = Instant::now() + Duration::from_secs(10);
        while released.load(Ordering::Relaxed) != 2 {
            dispatcher.run_until_parked();
            dispatcher.advance_clock(Duration::from_millis(2));
            ensure!(Instant::now() < deadline, "foreground retirement timed out");
            std::thread::yield_now();
        }
        assert!(!service.has_deferred());
        drop(host);
        dispatcher.run_until_parked();
        Ok(())
    }

    fn source_texture(
        service: &DirectXGpuService,
        width: u32,
        height: u32,
    ) -> Result<Arc<dyn GpuTexture>> {
        let pixels = (0..width * height)
            .flat_map(|index| [index as u8, 37, 91, 255])
            .collect::<Vec<_>>();
        let texture = service.with_context(|context| {
            let mut texture = None;
            let data = D3D11_SUBRESOURCE_DATA {
                pSysMem: pixels.as_ptr().cast(),
                SysMemPitch: width * 4,
                SysMemSlicePitch: 0,
            };
            unsafe {
                context.device.CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Width: width,
                        Height: height,
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
                    Some(&mut texture),
                )
            }?;
            texture.context("test texture")
        })?;
        service.wrap_texture(texture)
    }

    #[test]
    fn offscreen_readback_preserves_subregion_rows_and_completion() -> Result<()> {
        let host = OffscreenGpuHost::new()?;
        let service = host.service();
        let capabilities = service.capabilities();
        assert_eq!(capabilities.backend, GpuBackend::Direct3D11);
        assert!(!capabilities.adapter_name.is_empty());
        assert!(capabilities.max_texture_size >= 8192);
        assert_eq!(capabilities.rgba8_sample_counts.first(), Some(&1));
        assert_eq!(capabilities.max_pending_readback_bytes, MAX_READBACK_BYTES);
        let native = service
            .as_any()
            .downcast_ref::<DirectXGpuService>()
            .context("native service")?;
        let texture = source_texture(native, 13, 7)?;
        assert_eq!(texture.service_id(), service.service_id());
        assert!(
            service
                .readback(
                    &texture,
                    GpuRect {
                        x: 12,
                        y: 0,
                        width: 2,
                        height: 1
                    }
                )
                .is_err()
        );
        let mut readback = service.readback(
            &texture,
            GpuRect {
                x: 2,
                y: 1,
                width: 7,
                height: 3,
            },
        )?;
        let mut completion = native.completion()?;
        let deadline = Instant::now() + Duration::from_secs(10);
        let image = loop {
            if let Some(image) = readback.poll()? {
                break image;
            }
            ensure!(Instant::now() < deadline, "GPU readback timed out");
            std::thread::yield_now();
        };
        assert_eq!((image.width, image.height, image.pixels.len()), (7, 3, 84));
        for row in 0..3usize {
            for column in 0..7usize {
                let offset = (row * 7 + column) * 4;
                assert_eq!(
                    &image.pixels[offset..offset + 4],
                    &[((row + 1) * 13 + column + 2) as u8, 37, 91, 255]
                );
            }
        }
        while !completion.poll()? {
            ensure!(Instant::now() < deadline, "GPU completion timed out");
            std::thread::yield_now();
        }
        assert_eq!(native.readback_bytes.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn completed_readback_reuses_staging_and_cancelled_copy_retires_reservation() -> Result<()> {
        let host = OffscreenGpuHost::new()?;
        let service = host.service();
        let native = service
            .as_any()
            .downcast_ref::<DirectXGpuService>()
            .context("native service")?;
        let texture = source_texture(native, 13, 7)?;
        let region = GpuRect {
            x: 2,
            y: 1,
            width: 7,
            height: 3,
        };
        let mut readback = service.readback(&texture, region)?;
        assert!(readback.restart(&texture, region).is_err());
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(image) = readback.poll()? {
                assert_eq!(image.pixels[0], 15);
                break;
            }
            ensure!(
                Instant::now() < deadline,
                "initial reusable readback timed out"
            );
            std::thread::yield_now();
        }
        assert!(!readback.restart(&texture, GpuRect { width: 6, ..region })?);
        assert_eq!(service.diagnostics().pending_readback_bytes, 0);
        assert!(readback.restart(
            &texture,
            GpuRect {
                x: 3,
                y: 2,
                ..region
            }
        )?);
        loop {
            if let Some(image) = readback.poll()? {
                assert_eq!(image.pixels[0], 29);
                break;
            }
            ensure!(Instant::now() < deadline, "restarted readback timed out");
            std::thread::yield_now();
        }
        assert!(readback.restart(&texture, region)?);
        drop(readback);
        let counters = service.diagnostics();
        assert_eq!(counters.readback_allocations, 1);
        assert_eq!(counters.reused_readbacks, 2);
        assert_eq!(counters.readback_bytes, 84 * 3);
        assert_eq!(counters.pending_readback_bytes, 84);
        while service.poll_deferred()? {
            ensure!(
                Instant::now() < deadline,
                "cancelled readback retirement timed out"
            );
            std::thread::yield_now();
        }
        assert_eq!(service.diagnostics().pending_readback_bytes, 0);
        Ok(())
    }

    #[test]
    fn readback_validates_native_ownership_and_bounds_through_wrappers() -> Result<()> {
        #[derive(Debug)]
        struct MisreportedTexture {
            source: Arc<dyn GpuTexture>,
            identity: u64,
        }
        impl GpuTexture for MisreportedTexture {
            fn service_id(&self) -> u64 {
                self.identity
            }
            fn epoch(&self) -> u64 {
                1
            }
            fn resource_id(&self) -> u64 {
                self.source.resource_id()
            }
            fn size(&self) -> [u32; 2] {
                [64, 64]
            }
            fn as_any(&self) -> &dyn Any {
                self.source.as_any()
            }
        }
        let host = OffscreenGpuHost::new()?;
        let service = host.service();
        let texture: Arc<dyn GpuTexture> = Arc::new(MisreportedTexture {
            source: service.create_rgba8_texture([2, 2])?,
            identity: service.service_id(),
        });
        assert!(
            service
                .readback(
                    &texture,
                    GpuRect {
                        x: 2,
                        y: 0,
                        width: 1,
                        height: 1
                    }
                )
                .is_err()
        );
        let foreign_host = OffscreenGpuHost::new()?;
        let foreign: Arc<dyn GpuTexture> = Arc::new(MisreportedTexture {
            source: foreign_host.service().create_rgba8_texture([2, 2])?,
            identity: service.service_id(),
        });
        assert!(
            service
                .readback(
                    &foreign,
                    GpuRect {
                        x: 0,
                        y: 0,
                        width: 1,
                        height: 1
                    }
                )
                .is_err()
        );
        assert_eq!(service.diagnostics().pending_readback_bytes, 0);
        assert_eq!(service.diagnostics().readback_bytes, 0);
        Ok(())
    }

    #[test]
    fn native_access_rejects_workers_and_retired_device_generations() -> Result<()> {
        let host = OffscreenGpuHost::new()?;
        let service = host.service();
        let worker_service = service.clone();
        let rejected = std::thread::spawn(move || {
            worker_service
                .as_any()
                .downcast_ref::<DirectXGpuService>()
                .is_some_and(|native| native.with_context(|_| Ok(())).is_err())
        })
        .join()
        .map_err(|_| anyhow::anyhow!("worker panicked"))?;
        assert!(rejected);
        let native = service
            .as_any()
            .downcast_ref::<DirectXGpuService>()
            .context("native service")?;
        let texture = source_texture(native, 2, 2)?;
        drop(host);
        assert_ne!(texture.epoch(), service.epoch());
        assert!(native.with_context(|_| Ok(())).is_err());
        Ok(())
    }
}
