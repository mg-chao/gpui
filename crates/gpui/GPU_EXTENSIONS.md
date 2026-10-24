# GPU extensions

`gpu-extensions` is an optional, default-off feature for applications that produce
retained GPU images. It is forwarded by `gpui_platform` to native backends. The
initial implementation supports Windows D3D11; other backends report unsupported
capabilities. The feature contains no drawing-editor or application-specific code.

## Device ownership and preparation

Obtain `Arc<dyn GpuService>` through `Window::gpu_service()` or
`App::gpu_service()`. Both refer to the platform's existing device. The application
method also initializes the service in offscreen mode, without a visible window,
swapchain, or vsync loop. A windowed application must not initialize another GPU
device for its canvas.

`GpuService::capabilities()` reports the actual adapter name and identifiers,
texture/upload limits, supported RGBA8 sample counts, compute support, and the
aggregate readback limit. Select supported sample counts before allocating render
targets. Allocation can still fail when available device memory is exhausted.

The portable service also provides `create_rgba8_texture`,
`upload_rgba8_region`, and `completion`. New RGBA8 images are transparent.
Region uploads validate ownership, generation, extents, row stride and byte
length. They accept a mutable Arc with no other strong or weak owners; upload
before publishing, and retire previous GPU use before reusing an allocation.
Forwarding resource wrappers are read-only through this upload API. Native scoped
adapters remain responsible for aliases created through their raw device access.

Tests using GPUI's simulated platform can call
`TestAppContext::set_gpu_service(host.service())`; app and window access then refer
to that service. Keep the native host alive for the whole test. This hook supplies
device ownership and texture validation without replacing the simulated window's
headless renderer.

Standalone native tests/export tools can own an explicit
`gpui_windows::gpu_extensions::OffscreenGpuHost`. Keep the host alive while its
service, completions, or readbacks are used; dropping it retires that generation.

```rust,ignore
let service = window.gpu_service()?;
let native = service.as_any()
    .downcast_ref::<gpui_windows::gpu_extensions::DirectXGpuService>()
    .ok_or_else(|| anyhow::anyhow!("unsupported GPU backend"))?;

let texture = native.with_context(|context| {
    // Create/update resources with context.device and context.immediate_context.
    // Submit the application's own shaders here and return a finished texture.
    render_candidate(context.device, context.immediate_context)
})?;
let texture = native.wrap_texture(texture)?;
```

Native execution, submission, and polling belong to the GPUI device's owning
thread. Worker access and nested native execution scopes return errors. Each
scope must bind the state it uses; its exit clears D3D11 context state, and GPUI
rebinds its rendering state before drawing. `GpuJob` provides ordered preparation
through the portable service, while the native adapter supplies scoped backend
access. A `GpuCompletion` marker can be polled without waiting for retirement.

Device identity and epoch are attached to every resource. Device recovery retires
the previous service, rejects outstanding resources/readbacks, and exposes a new
service. Applications retain source data and reconstruct their caches. Native
device replacement on the vsync thread preserves the original UI execution owner.

## Publishing a complete frame

`Window::paint_gpu_texture(bounds, texture, options)` inserts a retained texture
into the ordinary GPUI scene order and inherited content clip. It supports an
affine transform, opacity (including inherited element opacity), and nearest or
linear sampling. Textures must be single-sample, single-mip RGBA8 or BGRA8 UNORM,
shader-readable, and contain premultiplied color.

Release builds compile the extension texture shader with strict validation and
level-3 optimization into embedded bytecode. Debug builds compile the same shader
with matching strict/optimized flags at initialization.

Prepare the image and its associated interaction overlays as one candidate before
publishing either. A failed preparation must retain the previously published
image and overlay state. A texture referenced by a published scene is immutable:
replace dirty tiles, or use a candidate allocation, instead of writing into the
currently displayed texture. Do not reuse an allocation until its scene references
and outstanding GPU work have retired.

`GpuTextureOptions` contains a content revision and optional `previous_revision`.
`GpuDamage::Rects` uses texture-pixel coordinates and is valid only relative to
that previous successfully presented revision. An unknown predecessor causes
complete texture damage. Provide all application effect dependencies in this
damage; the compositor adds its own sampling support. Immutable replacement
allocations at the same placement may use predecessor damage normally.

The Windows compositor maintains a complete retained window image and a separate
candidate. It compares ordinary GPUI primitives by their typed values, including
path vertices/colors, and damages their old/new visual bounds. Shadow damage
includes the Gaussian support. Changed texture tiles add mapped damage. The
compositor reconstructs the resulting region in scene order and copies every
pixel of that region into the current flip-sequential buffer before `Present1`.
It does not assume swapchain-buffer contents represent the preceding frame.

Unchanged output skips encoding and presentation. Resize, recovery, appearance
changes, unsupported surface content, or ambiguous texture placement/order changes
force a full frame. Failed candidate rendering leaves the last complete window
image intact. Native test captures use a separate target and do not corrupt the
dirty-rectangle presentation history.

Atlas insertions/removals carry a content revision because an allocation can reuse
the same sprite identity with different pixels. Atlas changes damage the old and
new sprite bounds. Fractional clip edges retain every potentially covered pixel.

`GpuService::diagnostics()` reports presentation/skip counts, damaged pixels,
readback reservations, cumulative `uploaded_bytes`/`readback_bytes`, compositor
CPU time and time inside `Present1`. Byte counters count accepted RGBA payloads,
excluding row padding; uploads through the native adapter are measured by the
consumer. Cancelled readbacks still count because their copy was submitted.
Take counter differences around a measurement interval. These CPU timings cover
command encoding and presentation submission; they do not measure GPU completion
or physical scanout latency.
`last_presentation_end` provides the monotonic timestamp at the return from the
latest accepted `Present1`, for correlation with native input timestamps.

## Readback

`GpuService::readback(texture, GpuRect)` queues a bounded staging copy. Poll its
`GpuReadback` on the device owner thread; `None` means pending, and a completed
`GpuReadbackImage` contains tightly packed RGBA8 premultiplied pixels. Readback
strips native row padding and BGRA ordering. It never waits for GPU completion.
Encoding can move to a CPU worker after completion.

Outstanding readbacks are limited to 256 MiB per service. Tile large exports and
retire or cancel previous requests before queueing more. Dropping a readback
with a pending copy defers its reservation and staging resource until completion.
Display updates do not require readback.

A completed readback retains its staging allocation. `GpuReadback::restart`
reuses it for the same dimensions and native format; incompatible requests return
false without changing the handle. Pending handles cannot restart. Keep a bounded
ring of completed handles together with their allocation-budget leases, and drop
unused completed handles to release their memory. GPUI does not retain an
unaccounted service-wide staging cache. `readback_allocations` and
`reused_readbacks` diagnostics distinguish allocation from reuse.
`pending_readback_bytes` falls when a result is consumed; retained completed
staging is still charged by the caller's allocation lease until its handle drops.

## Deferred resource release

`GpuService::defer_release(Box<dyn Any + Send>)` accepts resources and accounting
leases from any thread. The Windows backend groups them behind at most two
in-flight completion markers. Resources queued after an earlier marker stay in a
later batch. Completed payloads are dropped outside queue locks, so their
destructors may enqueue additional releases.

The native platform wakes a foreground task on demand and polls while releases
remain, including after the last canvas disappears. Standalone offscreen hosts
call `poll_deferred()` on their owner thread; it returns true while work remains
and never waits for GPU completion. Host/device invalidation first retires the
epoch, then releases queued payloads. Subsequent late releases drop immediately.
Payloads should use weak service references when retaining their accounting owner.

## Synchronizing the fork

This extension was developed against fork commit
`79fe6a645f68f8c9230578f154bf99852d9f716f`, from
`https://github.com/mg-chao/gpui.git`. The upstream remote is
`https://github.com/zed-industries/zed.git`.

The extension is isolated in `gpui/src/gpu_extensions.rs` and the Windows
`gpu_extensions`/`gpu_compositor` modules, with small feature-gated hooks for
platform service access, scene insertion/batching, and renderer presentation.
Application shaders, effect graphs, document logic, CPU reconstruction, cache
policies, and export formats remain with the consumer. Keep the feature-disabled
path and ordinary upstream behavior covered when rebasing.

When updating the fork, first merge or rebase the selected upstream revision.
Review device creation/recovery and context ownership, scene ordering/replay,
atlas allocation lifetime, shader build integration, and swapchain
creation/presentation hooks against the new upstream implementation. Preserve the
native platform's device as the single owner. Then compile with the feature
disabled, compile with it enabled, run the
focused extension tests, and run `./script/clippy` for the affected packages.
Finally run the consumer's native window benchmark to exercise partial presents,
idle suppression, resize, and minimize/restore with the updated event loop.

Related checks are the `gpui::gpu_extensions` tests and Windows
`gpu_extensions`/`gpu_compositor` tests. Native tests exercise real offscreen GPU
compositing, premultiplied opacity, transforms/clipping, asynchronous padded-row
readback, service retirement, damage, and revision handling. Run only these
filters for extension changes; the full workspace suite is unnecessary.
