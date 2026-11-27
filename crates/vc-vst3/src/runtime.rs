//! Realtime bridge between the host's `process()` callback and the RVC pipeline.
//!
//! Mirrors the CLI's `engine.rs` worker model: the audio thread only pushes
//! input and pops output through lock-free SPSC ring buffers, while a dedicated
//! worker thread owns the `RvcPipeline`, runs inference, and smooths/resamples
//! the result back to the host sample rate. Inference and allocation never run
//! on the audio thread.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use nice_plug::prelude::{util, ProcessMode};
use rtrb::{Consumer, Producer, RingBuffer};
use vc_core::dsp::chunk_samples_for_rate;
use vc_core::model_rvc::{
    ChunkConverter, ChunkOutputConfig, ChunkStats, F0Config, LiveParams, LoadProgress,
    NoiseGateShaping, OutputDynamicsConfig, RvcPipeline, RvcPipelineConfig,
};
use vc_core::sola::SmoothingKind;
use vc_core::validation::CONVERSION_TIMING_LIMITS;

use crate::config::PluginConfig;
use crate::params::VcRvcParams;
use crate::voice_bank::{VoiceBank, LABELS};

#[derive(Clone, Debug)]
pub(crate) struct PluginStatus {
    pub summary: String,
    pub detail: Option<String>,
}

impl PluginStatus {
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            detail: None,
        }
    }
}

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;

#[derive(Clone, Copy, Default)]
struct QueuedSample {
    value: f32,
    generation: u32,
}

struct RuntimeState {
    diagnostics: Arc<crate::diagnostics::Diagnostics>,
    generation: AtomicU32,
    ready_generation: AtomicU32,
    input_hop: AtomicUsize,
    offline: AtomicBool,
    has_converter: AtomicBool,
    applied_settings: Mutex<Option<PluginConfig>>,
    // Only offline process() waits/takes this lock. Realtime callbacks never do.
    progress: Mutex<()>,
    progress_changed: Condvar,
}

impl RuntimeState {
    fn notify_offline(&self) {
        if self.offline.load(Ordering::Acquire) {
            let _guard = self.progress.lock().unwrap_or_else(|e| e.into_inner());
            self.progress_changed.notify_all();
        }
    }
}

// Also runs when inference panics: an offline host must not wait forever for a
// worker which has exited without publishing another output chunk.
struct WorkerExit {
    running: Arc<AtomicBool>,
    state: Arc<RuntimeState>,
    params: Arc<VcRvcParams>,
}

impl Drop for WorkerExit {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        self.state.has_converter.store(false, Ordering::Release);
        self.params.loaded_voice_mask.store(0, Ordering::Release);
        self.params.active_voice.store(0, Ordering::Release);
        self.state.notify_offline();
    }
}

trait WorkerConversion: Send {
    fn profile_hop(&self) -> Option<usize> { None }
    fn quality_diagnostics(&mut self, _enabled: bool) {}
    fn quality_report(&mut self) -> String { String::new() }
    fn available_mask(&self)->u32 { 0b11110 }
    fn active_slot(&self)->Option<usize>{None}
    fn install_voice(&mut self, _slot:usize, _voice:Box<dyn WorkerConversion>)->anyhow::Result<()> { anyhow::bail!("Not a voice bank") }
    fn select_voice(&mut self, _requested: usize) -> anyhow::Result<bool> { Ok(false) }
    fn apply_live(&mut self, live: &LiveParams);
    fn reset(&mut self) -> anyhow::Result<()>;
    fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()>;
    fn process(
        &mut self,
        input: &[f32],
        rate: u32,
        output: &mut Vec<f32>,
    ) -> anyhow::Result<ChunkStats>;
    fn content_delay(&self) -> usize;
}

impl WorkerConversion for VoiceBank<Box<dyn WorkerConversion>> {
    fn profile_hop(&self) -> Option<usize> { self.current().profile_hop() }
    fn quality_diagnostics(&mut self, enabled: bool) { self.current_mut().quality_diagnostics(enabled); }
    fn quality_report(&mut self) -> String { self.current_mut().quality_report() }
    fn available_mask(&self)->u32 { VoiceBank::available_mask(self) }
    fn active_slot(&self)->Option<usize>{Some(self.active())}
    fn install_voice(&mut self, slot:usize, voice:Box<dyn WorkerConversion>)->anyhow::Result<()> { self.insert(slot,voice) }
    fn select_voice(&mut self, requested: usize) -> anyhow::Result<bool> { self.select(requested) }
    fn apply_live(&mut self, live: &LiveParams) { self.current_mut().apply_live(live); }
    fn reset(&mut self) -> anyhow::Result<()> { self.current_mut().reset() }
    fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()> {
        self.current_mut().prime(input, rate)
    }
    fn process(&mut self, input: &[f32], rate: u32, output: &mut Vec<f32>) -> anyhow::Result<ChunkStats> {
        self.current_mut().process(input, rate, output)
    }
    fn content_delay(&self) -> usize { self.current().content_delay() }
}

/// A bank slot owns its loaded timing and live parameters. Host automation
/// remains available only for the legacy single-model path.
struct ProfiledConversion {
    inner: ChunkConverter<RvcPipeline>,
    live: LiveParams,
    hop: usize,
}
impl WorkerConversion for ProfiledConversion {
    fn profile_hop(&self) -> Option<usize> { Some(self.hop) }
    fn quality_diagnostics(&mut self, enabled: bool) { self.inner.model_mut().set_quality_diagnostics(enabled); }
    fn quality_report(&mut self) -> String { let join = self.inner.last_join_diagnostics(); format!("{} join={join:?}",self.inner.model_mut().quality_report()) }
    fn apply_live(&mut self, _: &LiveParams) { self.inner.model_mut().apply_live(&self.live); }
    fn reset(&mut self) -> anyhow::Result<()> { WorkerConversion::reset(&mut self.inner) }
    fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()> { WorkerConversion::prime(&mut self.inner,input,rate) }
    fn process(&mut self,input:&[f32],rate:u32,output:&mut Vec<f32>)->anyhow::Result<ChunkStats>{
        WorkerConversion::process(&mut self.inner,input,rate,output)
    }
    fn content_delay(&self)->usize { WorkerConversion::content_delay(&self.inner) }
}
impl WorkerConversion for ChunkConverter<RvcPipeline> {
    fn apply_live(&mut self, live: &LiveParams) {
        self.model_mut().apply_live(live);
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        self.model_mut().reset_streaming_state()?;
        self.reset_streaming_state();
        Ok(())
    }

    fn process(
        &mut self,
        input: &[f32],
        rate: u32,
        output: &mut Vec<f32>,
    ) -> anyhow::Result<ChunkStats> {
        self.process_chunk(input, rate, output)
    }

    fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()> {
        ChunkConverter::prime(self, input, rate).map(|_| ())
    }

    fn content_delay(&self) -> usize {
        self.output_content_delay_samples()
    }
}

/// Discard obsolete generations without scanning a potentially multi-second
/// backlog in the callback. A single producer publishes generations in order;
/// the current generation, if present, is the suffix of each ring slice.
fn read_generation(
    consumer: &mut Consumer<QueuedSample>,
    generation: u32,
    output: &mut [f32],
) -> usize {
    let available = consumer.slots();
    if available == 0 {
        return 0;
    }
    let Ok(chunk) = consumer.read_chunk(available) else {
        return 0;
    };
    let (a, b) = chunk.as_slices();
    let mut consumed = 0;
    let mut filled = 0;
    for slice in [a, b] {
        let stale =
            slice.partition_point(|sample| (generation.wrapping_sub(sample.generation) as i32) > 0);
        consumed += stale;
        // A reset can race the worker's input read. Preserve newer input until
        // its corresponding model reset, rather than discarding or converting it.
        let current = slice[stale..].partition_point(|sample| sample.generation == generation);
        let take = current.min(output.len() - filled);
        for (sample, target) in slice[stale..stale + take]
            .iter()
            .zip(&mut output[filled..filled + take])
        {
            *target = sample.value;
        }
        filled += take;
        consumed += take;
        if take < current
            || current < slice.len() - stale
            || (filled == output.len() && current > 0)
        {
            break;
        }
    }
    chunk.commit(consumed);
    filled
}

/// Allowed range for the user-tunable chunk size (ms). The ring buffers are
/// sized for `MAX_CHUNK_MS` up front so chunk changes apply live (on reload)
/// without reallocating them.
pub const MIN_CHUNK_MS: u32 = CONVERSION_TIMING_LIMITS.min_chunk_ms;
pub const MAX_CHUNK_MS: u32 = CONVERSION_TIMING_LIMITS.max_chunk_ms;

/// Lets the editor's Load / Reload submit (a non-realtime UI thread) wake the
/// current worker even while the host is idle and not calling `process()`, so a
/// reload starts immediately instead of waiting for the worker's park timeout.
///
/// The worker handle is republished on every `PluginRuntime::start`, so this
/// stays correct across `initialize()`-driven worker restarts (sample-rate or
/// block-size changes). The realtime `process()` path deliberately does NOT use
/// this — it unparks through the [`Thread`] stored directly in [`PluginRuntime`]
/// so the audio callback never takes a lock. The editor and worker registration
/// are both off the audio thread, so the `Mutex` here is fine.
#[derive(Default)]
pub(crate) struct ReloadWaker {
    thread: Mutex<Option<Thread>>,
}

impl ReloadWaker {
    fn register(&self, thread: Thread) {
        if let Ok(mut slot) = self.thread.lock() {
            *slot = Some(thread);
        }
    }

    /// Unpark the current worker, if any. Called after the editor sets `reload`.
    pub(crate) fn wake(&self) {
        if let Ok(slot) = self.thread.lock() {
            if let Some(thread) = slot.as_ref() {
                thread.unpark();
            }
        }
    }
}

/// Owns the worker thread and the audio-thread ends of the ring buffers.
pub struct PluginRuntime {
    /// Worker `Thread` handle for the realtime wake path (input queued in
    /// `process_block`, and Drop). Stored directly so the audio callback unparks
    /// without a lock; refreshed whenever the worker is (re)spawned.
    worker_thread: Thread,
    input_producer: Producer<QueuedSample>,
    output_consumer: Consumer<QueuedSample>,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    mono_in: Vec<QueuedSample>,
    mono_out: Vec<f32>,
    /// Initial plugin latency in host samples, reported at `initialize`.
    pub latency_samples: u32,
    /// Current latency, updated after model/output initialization or reload. The audio
    /// thread re-reports it to the host (see [`PluginRuntime::poll_latency_update`]).
    latency: Arc<AtomicU32>,
    last_reported_latency: u32,
    loading: Arc<AtomicBool>,
    state: Arc<RuntimeState>,
    sample_rate: u32,
    max_block_capacity: usize,
    offline_generation: u32,
    offline_startup_remaining: usize,
}

impl PluginRuntime {
    /// Start the worker and allocate the ring buffers for the given host rate.
    /// `max_block` is the host's maximum block size used to pre-size scratch.
    ///
    /// Ring capacity is fixed here for `MAX_CHUNK_MS`. Chunk/join geometry comes
    /// from the applied Load / Reload snapshot; the callback relays the worker's
    /// updated latency. A previously loaded snapshot may rebuild fixed profiles
    /// after a host rate change, without applying staged editor changes.
    // The worker is wired up from a handful of independent shared handles
    // (editor/worker handshake flags, status, reload waker) plus the audio
    // format; bundling them into a struct would only move the same fields behind
    // an extra type without making the lifecycle clearer.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        params: Arc<VcRvcParams>,
        reload: Arc<AtomicBool>,
        loading: Arc<AtomicBool>,
        dirty: Arc<AtomicBool>,
        status: Arc<Mutex<PluginStatus>>,
        reload_waker: &Arc<ReloadWaker>,
        sample_rate: u32,
        max_block: usize,
        mode: ProcessMode,
        applied_settings: Option<PluginConfig>,
    ) -> Self {
        Self::start_inner(
            params,
            reload,
            loading,
            dirty,
            status,
            reload_waker,
            sample_rate,
            max_block,
            mode,
            applied_settings,
            #[cfg(test)]
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start_inner(
        params: Arc<VcRvcParams>,
        reload: Arc<AtomicBool>,
        loading: Arc<AtomicBool>,
        dirty: Arc<AtomicBool>,
        status: Arc<Mutex<PluginStatus>>,
        reload_waker: &Arc<ReloadWaker>,
        sample_rate: u32,
        max_block: usize,
        mode: ProcessMode,
        applied_settings: Option<PluginConfig>,
        #[cfg(test)] initial_conversion: Option<Box<dyn WorkerConversion>>,
    ) -> Self {
        // Reload requests belong to one runtime lifecycle. Dropping a stale
        // request here also guarantees the replacement starts with an enabled
        // button rather than accepting a duplicate while it handles old work.
        reload.store(false, Ordering::SeqCst);
        let autoload = applied_settings.is_some();
        loading.store(autoload, Ordering::SeqCst);
        let settings0 = applied_settings.unwrap_or_else(|| params.settings.read().unwrap().clone());
        let initial_timing_settings =
            if let Err(err) = settings0.validated_chunk_samples(sample_rate) {
                nice_plug::nice_error!("vc-vst3: invalid persisted settings: {err}");
                if let Ok(mut current) = status.lock() {
                    *current = PluginStatus {
                        summary: format!("invalid settings: {err}"),
                        detail: Some(format!("{err:#}")),
                    };
                }
                PluginConfig::default()
            } else {
                settings0.clone()
            };
        let crossfade_ms = initial_timing_settings.crossfade_ms;
        let sola_search_ms = initial_timing_settings.sola_search_ms;
        let tail_discard_ms = initial_timing_settings.rvc_output_tail_discard_ms;
        let output_extra_ms = crossfade_ms
            .saturating_add(sola_search_ms)
            .saturating_add(tail_discard_ms);

        // Size the rings for the largest allowed chunk so `chunk_ms` can change
        // without reallocating. Extra capacity does not add latency (the worker
        // pops as soon as a chunk is available).
        let max_chunk_samples = chunk_samples_for_rate(sample_rate, MAX_CHUNK_MS);
        let input_capacity =
            (max_chunk_samples * INPUT_QUEUE_CHUNKS).max(max_block + max_chunk_samples);
        let output_capacity =
            (max_chunk_samples * OUTPUT_QUEUE_CHUNKS).max(max_block + max_chunk_samples);
        let (input_producer, input_consumer) = RingBuffer::<QueuedSample>::new(input_capacity);
        let (output_producer, output_consumer) = RingBuffer::<QueuedSample>::new(output_capacity);

        let running = Arc::new(AtomicBool::new(true));

        // Before explicit model loading, only host chunk/context timing is
        // known. The worker replaces this estimate with the shared converter's
        // actual input/join/output buffering delay after its first model call.
        let chunk_ms = initial_timing_settings.chunk_ms;
        let chunk_samples = chunk_samples_for_rate(sample_rate, chunk_ms);
        let extra_samples = chunk_samples_for_rate(sample_rate, output_extra_ms);
        let latency_samples = (chunk_samples + extra_samples) as u32;
        let latency = Arc::new(AtomicU32::new(latency_samples));
        let state = Arc::new(RuntimeState {
            diagnostics: params.diagnostics.clone(),
            generation: AtomicU32::new(1),
            ready_generation: AtomicU32::new(0),
            input_hop: AtomicUsize::new(chunk_samples),
            offline: AtomicBool::new(mode == ProcessMode::Offline),
            has_converter: AtomicBool::new(false),
            applied_settings: Mutex::new(autoload.then(|| settings0.clone())),
            progress: Mutex::new(()),
            progress_changed: Condvar::new(),
        });

        let worker = WorkerCtx {
            params,
            reload,
            loading: Arc::clone(&loading),
            dirty,
            status,
            sample_rate,
            crossfade_ms,
            sola_search_ms,
            tail_discard_ms,
            latency: Arc::clone(&latency),
            running: Arc::clone(&running),
            input_consumer,
            output_producer,
            state: Arc::clone(&state),
            initial_settings: settings0,
            autoload,
            #[cfg(test)]
            initial_conversion,
        }
        .spawn();

        // Publish the worker handle for both wake paths: the realtime path keeps
        // its own clone, while the editor's reload submit goes through the shared
        // ReloadWaker (re-registered here so it tracks worker restarts).
        let worker_thread = worker.thread().clone();
        reload_waker.register(worker_thread.clone());

        Self {
            worker_thread,
            input_producer,
            output_consumer,
            running,
            worker: Some(worker),
            mono_in: Vec::with_capacity(max_block),
            mono_out: vec![0.0; max_block],
            latency_samples,
            latency,
            last_reported_latency: latency_samples,
            loading,
            state,
            sample_rate,
            max_block_capacity: input_capacity.min(output_capacity) - max_chunk_samples,
            offline_generation: 0,
            offline_startup_remaining: 0,
        }
    }

    /// Mode/block changes retain successfully loaded models and staged settings.
    /// Only initialize() may grow scratch; process() accepts the advertised bound.
    pub fn reconfigure(&mut self, rate: u32, max_block: usize, mode: ProcessMode) -> bool {
        if !self.running.load(Ordering::Acquire)
            || rate != self.sample_rate
            || max_block > self.max_block_capacity
        {
            return false;
        }
        self.mono_in.resize(max_block, QueuedSample::default());
        self.mono_in.clear();
        self.mono_out.resize(max_block, 0.0);
        self.state
            .offline
            .store(mode == ProcessMode::Offline, Ordering::Release);
        self.reset();
        self.worker_thread.unpark();
        true
    }

    pub fn applied_settings(&self) -> Option<PluginConfig> {
        self.state.applied_settings.lock().unwrap().clone()
    }

    pub(crate) fn realtime(&self)->bool { !self.state.offline.load(Ordering::Acquire) }
    pub(crate) fn wake(&self){ self.worker_thread.unpark(); }
    pub(crate) fn converter_ready(&self)->bool { self.state.has_converter.load(Ordering::Acquire) && !self.loading.load(Ordering::Acquire) }
    pub fn current_latency(&self) -> u32 {
        self.latency.load(Ordering::Relaxed)
    }

    pub fn tail_samples(&self) -> Option<u32> {
        self.state
            .has_converter
            .load(Ordering::Acquire)
            .then(|| self.current_latency())
    }

    /// Called on the audio thread. In-flight inference may finish later, so
    /// flushing queues alone is insufficient: tag every sample and wait for the
    /// worker's model + smoother reset acknowledgement before emitting new audio.
    pub fn reset(&mut self) {
        self.state.generation.fetch_add(1, Ordering::AcqRel);
        let queued = self.output_consumer.slots();
        if queued > 0 {
            if let Ok(chunk) = self.output_consumer.read_chunk(queued) {
                chunk.commit_all();
            }
        }
        self.offline_generation = 0;
        self.worker_thread.unpark();
    }

    /// Returns a new latency value if the worker changed it (chunk_ms edit) since
    /// the last call, so the audio thread can re-report it to the host.
    pub fn poll_latency_update(&mut self) -> Option<u32> {
        let current = self.latency.load(Ordering::Relaxed);
        if current != self.last_reported_latency {
            self.last_reported_latency = current;
            Some(current)
        } else {
            None
        }
    }

    /// Audio-thread entry point. Downmixes input to mono, queues it, and fills
    /// the output channels from the worker's converted audio (silence on
    /// underrun). Realtime processing is allocation/lock/wait-free. Offline
    /// rendering alone waits for the same worker, paced by sample counts rather
    /// than wall time, so freewheeling cannot discard input or insert underruns.
    pub fn process_block(&mut self, channels: &mut [&mut [f32]]) {
        if channels.is_empty() {
            return;
        }
        let n = channels[0].len();
        if n == 0 {
            return;
        }
        if n > self.mono_in.capacity() || n > self.mono_out.len() {
            // A host violating max_buffer_size must not trigger callback allocation.
            for channel in channels.iter_mut() {
                channel.fill(0.0);
            }
            return;
        }
        let generation = self.state.generation.load(Ordering::Acquire);
        let offline = self.state.offline.load(Ordering::Acquire);
        if offline && !self.wait_until_ready(generation) {
            for channel in channels.iter_mut() {
                channel.fill(0.0);
            }
            return;
        }

        // Generation travels with audio, including samples queued while reset
        // is still pending; the worker rejects the old input timeline.
        self.mono_in.clear();
        self.mono_in.resize(n, QueuedSample::default());
        if channels.len() >= 2 {
            let (left, right) = (&channels[0], &channels[1]);
            for i in 0..n {
                self.mono_in[i] = QueuedSample {
                    value: 0.5 * (left[i] + right[i]),
                    generation,
                };
            }
        } else {
            for (target, value) in self.mono_in.iter_mut().zip(channels[0].iter().copied()) {
                *target = QueuedSample { value, generation };
            }
        }

        // Queue input; drop on overflow (worker is behind, audio keeps flowing).
        let (pushed, _) = self.input_producer.push_partial_slice(&self.mono_in);
        self.state.diagnostics.input_dropped.fetch_add(n.saturating_sub(pushed.len()) as u64, Ordering::Relaxed);
        if !pushed.is_empty() {
            // Wake the worker now that input is queued instead of letting it find
            // the data on a fixed poll. unpark is wait-free (token store, or one
            // OS wakeup when actually parked), so it is safe on the audio thread.
            self.worker_thread.unpark();
        }

        self.mono_out[..n].fill(0.0);
        if offline {
            if self.offline_generation != generation {
                self.offline_generation = generation;
                self.offline_startup_remaining = self.state.input_hop.load(Ordering::Acquire);
            }
            // One input hop is the host-scheduling delay already included in
            // latency_samples. Shared model/join/resampler content delay stays
            // in the converter; do not add or remove it again here.
            let startup = n.min(self.offline_startup_remaining);
            self.offline_startup_remaining -= startup;
            let state = Arc::clone(&self.state);
            let mut guard = state.progress.lock().unwrap_or_else(|e| e.into_inner());
            let mut filled = startup;
            while filled < n {
                filled += read_generation(
                    &mut self.output_consumer,
                    generation,
                    &mut self.mono_out[filled..n],
                );
                self.worker_thread.unpark();
                if filled == n
                    || !self.running.load(Ordering::Acquire)
                    || !state.has_converter.load(Ordering::Acquire)
                    || state.generation.load(Ordering::Acquire) != generation
                {
                    break;
                }
                guard = state
                    .progress_changed
                    .wait_timeout(guard, Duration::from_millis(100))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        } else if self.state.ready_generation.load(Ordering::Acquire) == generation
            && !self.loading.load(Ordering::Acquire)
        {
            let read = read_generation(
                &mut self.output_consumer,
                generation,
                &mut self.mono_out[..n],
            );
            self.state.diagnostics.output_missing.fetch_add(n.saturating_sub(read) as u64, Ordering::Relaxed);
        } else {
            self.state.diagnostics.output_missing.fetch_add(n as u64, Ordering::Relaxed);
        }
        if self.state.generation.load(Ordering::Acquire) != generation {
            self.mono_out[..n].fill(0.0);
        }

        // Fan out mono to every output channel.
        for channel in channels.iter_mut() {
            channel[..n].copy_from_slice(&self.mono_out[..n]);
        }
    }

    fn wait_until_ready(&self, generation: u32) -> bool {
        let mut guard = self
            .state
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        loop {
            if !self.running.load(Ordering::Acquire)
                || self.state.generation.load(Ordering::Acquire) != generation
            {
                return false;
            }
            if !self.loading.load(Ordering::Acquire) {
                if !self.state.has_converter.load(Ordering::Acquire) {
                    return false;
                }
                if self.state.ready_generation.load(Ordering::Acquire) == generation {
                    return true;
                }
            }
            guard = self
                .state
                .progress_changed
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

impl Drop for PluginRuntime {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        // Wake a parked worker so it observes the cleared running flag and exits.
        // The host may unload us while idle (no process() wakes arriving), so the
        // join below would otherwise block until the worker's park timeout.
        self.worker_thread.unpark();
        self.state.notify_offline();
        let Some(handle) = self.worker.take() else {
            self.loading.store(false, Ordering::SeqCst);
            return;
        };
        // The host may call deactivate/drop from a thread that blocks while a
        // slow model load finishes. We still join here: detaching would allow a
        // worker to continue executing plugin code after the DAW unloads this
        // DLL, which is a harder crash mode than a bounded unload wait.
        let _ = handle.join();
        self.loading.store(false, Ordering::SeqCst);
    }
}

/// Everything the worker thread needs, moved into it on spawn.
struct WorkerCtx {
    params: Arc<VcRvcParams>,
    reload: Arc<AtomicBool>,
    loading: Arc<AtomicBool>,
    dirty: Arc<AtomicBool>,
    status: Arc<Mutex<PluginStatus>>,
    sample_rate: u32,
    crossfade_ms: u32,
    sola_search_ms: u32,
    tail_discard_ms: u32,
    latency: Arc<AtomicU32>,
    running: Arc<AtomicBool>,
    input_consumer: Consumer<QueuedSample>,
    output_producer: Producer<QueuedSample>,
    state: Arc<RuntimeState>,
    initial_settings: PluginConfig,
    autoload: bool,
    #[cfg(test)]
    initial_conversion: Option<Box<dyn WorkerConversion>>,
}

/// Five-ms bounded seam smoothing. A decaying endpoint correction avoids
/// replaying old speech or fading the whole new chunk to silence.
fn soften_voice_transition(audio: &mut [f32], from: f32, rate: u32) {
    let n = audio.len().min((rate as usize / 200).max(1));
    if n == 0 { return; }
    let correction = from - audio[0];
    for (i, sample) in audio[..n].iter_mut().enumerate() {
        let t = if n == 1 { 1.0 } else { i as f32 / (n-1) as f32 };
        let weight = 0.5 * (1.0 + (std::f32::consts::PI*t).cos());
        *sample = (*sample + correction*weight).clamp(-1.0,1.0);
    }
}

impl WorkerCtx {
    fn spawn(self) -> JoinHandle<()> {
        thread::Builder::new()
            .name("vc-vst3-rvc".to_string())
            .spawn(move || self.run())
            .expect("failed to spawn vc-vst3 worker thread")
    }

    fn run(mut self) {
        let _exit = WorkerExit {
            running: Arc::clone(&self.running),
            state: Arc::clone(&self.state),
            params: self.params.clone(),
        };
        let mut last_voice_request = self.params.selected_voice().saturating_sub(1) as usize;
        let initial_settings = self.initial_settings.clone();
        let mut last_profile_flush = std::time::Instant::now();
        let mut last_chunk_log = std::time::Instant::now();
        let mut last_selection = self.params.selected_voice();
        self.params.diagnostics.event("WORKER_START", &format!("rate={} provider={}", self.sample_rate, initial_settings.provider().label()));
        let initial_chunk = initial_settings.validated_chunk_samples(self.sample_rate);
        // Invalid persisted settings never load a model. The fallback only
        // sizes the idle/silent worker until the user submits valid settings.
        let mut chunk_samples = initial_chunk.as_ref().copied().unwrap_or_else(|_| {
            chunk_samples_for_rate(self.sample_rate, PluginConfig::default().chunk_ms).max(1)
        });
        let mut input_acc = Vec::<f32>::with_capacity(chunk_samples * 2);
        // Reused output buffer for the converted chunk, filled by `process_chunk`.
        let mut chunk_out = Vec::<f32>::with_capacity(chunk_samples * 2);
        let mut queued_output = Vec::<QueuedSample>::with_capacity(chunk_samples * 2);
        let mut zero_input = vec![0.0; chunk_samples];
        let mut bank_loading:Option<BankLoadTask>=None;
        let mut previous_input = Vec::<f32>::with_capacity(chunk_samples);
        let mut previous_output_end: Option<f32> = None;
        let mut pending_switch: Option<(i32, Instant)> = None;
        let mut transition_from: Option<f32> = None;

        // Do not load models during host startup, plugin scan, or project
        // restore. Some DAWs instantiate and tear down plugins on UI/control
        // threads, and CUDA/ORT loading can crash or stall the entire host if it
        // happens implicitly. The editor's Load / Reload button is the explicit
        // boundary for model (re)initialization.
        #[cfg(not(test))]
        let mut converter: Option<Box<dyn WorkerConversion>> = None;
        #[cfg(test)]
        let mut converter = self.initial_conversion.take();
        if let Err(err) = initial_chunk {
            self.set_error(format!("invalid settings: {err}"), format!("{err:#}"));
        } else if self.autoload && converter.is_none() {
            if initial_settings.has_voice_bank() && initial_settings.provider().is_tensorrt() && !self.state.offline.load(Ordering::Acquire) {
                match BankLoadTask::start(self.model_loader(),initial_settings.clone(),self.params.selected_voice()) {
                    Ok(task)=>bank_loading=Some(task),
                    Err(err)=>self.set_error("Model loader could not start",err.to_string()),
                }
            }else{converter = self.load_conversion(&initial_settings, chunk_samples);}
        } else {
            if converter.is_none() {
                self.set_idle_status();
            }
        }
        if let Some(current) = converter.as_mut() {
            if let Some(hop) = current.profile_hop() {
                chunk_samples = hop;
                zero_input.resize(hop, 0.0);
                self.state.input_hop.store(hop, Ordering::Release);
                self.latency.store(u32::try_from(hop.saturating_add(current.content_delay())).unwrap_or(u32::MAX), Ordering::Relaxed);
            }
            if let Err(err) = self.prepare_offline(current.as_mut(), &zero_input) {
                self.set_error("offline preroll failed", format!("{err:#}"));
                converter = None;
            }
        }
        self.publish_conversion(&initial_settings, converter.as_deref());
        let mut generation = self.state.generation.load(Ordering::Acquire);
        self.state
            .ready_generation
            .store(generation, Ordering::Release);
        self.loading.store(bank_loading.is_some(), Ordering::Release);
        self.state.notify_offline();

        let mut managed_checked = self.autoload;
        while self.running.load(Ordering::SeqCst) {
            if !managed_checked && !cfg!(test) && self.params.audio_seen.load(Ordering::Acquire) && !self.state.offline.load(Ordering::Acquire) {
                managed_checked = true;
                if let Some(root) = crate::diagnostics::local_dir() {
                    let mut settings = self.params.settings.read().unwrap().clone();
                    if root.join("models").is_dir() {
                        self.loading.store(true, Ordering::Release);
                        self.set_status("Discovering local voice models");
                        crate::managed_models::discover(&root, &mut settings, &self.running, &self.params.diagnostics);
                        if self.running.load(Ordering::Acquire) {
                            *self.params.settings.write().unwrap() = settings.clone();
                            self.params.request_save();
                            if settings.has_models() { self.reload.store(true, Ordering::Release); }
                            else { self.loading.store(false, Ordering::Release); self.set_error("共通モデルまたはキャラクターが未設定",crate::managed_models::missing_shared_message(&settings,&root)); }
                        }
                    } else if settings.has_models() && (settings.has_voice_bank() || (settings.model.is_file() && settings.embedder.is_file() && settings.f0_model.is_file())) {
                        self.loading.store(true, Ordering::Release);
                        self.reload.store(true, Ordering::Release);
                    }
                }
            }
            if last_profile_flush.elapsed() >= Duration::from_secs(2) {
                self.params.flush_profile();
                last_profile_flush = std::time::Instant::now();
            }
            let selection = self.params.selected_voice();
            if selection != last_selection {
                self.params.diagnostics.event("VOICE_REQUEST", &format!("slot={} previous={}", selection, last_selection));
                last_selection = selection;
            }
            // The editor prevents another request while this one is loading, so
            // the worker can act immediately without a debounce timer.
            if self.reload.swap(false, Ordering::SeqCst) {
                // Clear before taking the snapshot. An edit after this point
                // re-sets dirty and remains visibly staged for the next load.
                self.dirty.store(false, Ordering::SeqCst);
                let settings = self.params.settings.read().unwrap().clone();
                let new_chunk_samples = match settings.validated_chunk_samples(self.sample_rate) {
                    Ok(samples) => samples,
                    Err(err) => {
                        nice_plug::nice_error!("vc-vst3: invalid settings: {err}");
                        self.set_error(format!("invalid settings: {err}"), format!("{err:#}"));
                        self.dirty.store(true, Ordering::SeqCst);
                        self.loading.store(false, Ordering::SeqCst);
                        self.state.notify_offline();
                        continue;
                    }
                };
                // All join geometry belongs to the validated reload snapshot.
                // Keeping startup values here would disagree with settings
                // recovered from an invalid persisted configuration.
                self.crossfade_ms = settings.crossfade_ms;
                self.sola_search_ms = settings.sola_search_ms;
                self.tail_discard_ms = settings.rvc_output_tail_discard_ms;
                // chunk_ms may have changed; recompute and re-report latency.
                chunk_samples = new_chunk_samples;
                zero_input.resize(chunk_samples, 0.0);
                self.state.input_hop.store(chunk_samples, Ordering::Release);
                self.latency
                    .store(self.latency_samples(chunk_samples), Ordering::Relaxed);
                // Drop the old pipeline (releasing its CUDA context) before
                // building the new one, so the two never coexist.
                drop(bank_loading.take());
                drop(converter.take());
                self.state.generation.fetch_add(1, Ordering::AcqRel);
                self.publish_conversion(&settings, None);
                if settings.has_voice_bank() && settings.provider().is_tensorrt() && !self.state.offline.load(Ordering::Acquire) {
                    match BankLoadTask::start(self.model_loader(),settings.clone(),self.params.selected_voice()) {
                        Ok(task)=>bank_loading=Some(task),
                        Err(err)=>self.set_error("Model loader could not start",err.to_string()),
                    }
                }else{converter = self.load_conversion(&settings, chunk_samples);}
                if let Some(current) = converter.as_mut() {
                    if let Some(hop) = current.profile_hop() {
                chunk_samples = hop;
                zero_input.resize(hop, 0.0);
                self.state.input_hop.store(hop, Ordering::Release);
                self.latency.store(u32::try_from(hop.saturating_add(current.content_delay())).unwrap_or(u32::MAX), Ordering::Relaxed);
            }
                    if let Err(err) = self.prepare_offline(current.as_mut(), &zero_input) {
                        self.set_error("offline preroll failed", format!("{err:#}"));
                        converter = None;
                    }
                }
                input_acc.clear();
                previous_input.clear(); previous_output_end = None; pending_switch = None; transition_from = None;
                self.drain_input();
                // Loading made a fresh model timeline. A host reset issued
                // during the load is covered by that same fresh initialization.
                generation = self.state.generation.load(Ordering::Acquire);
                self.publish_conversion(&settings, converter.as_deref());
                // The requested slot may have changed while loading.
                last_voice_request=usize::MAX;
                self.state
                    .ready_generation
                    .store(generation, Ordering::Release);
                self.loading.store(bank_loading.is_some(), Ordering::SeqCst);
                self.state.notify_offline();
            }

            // Publish each completed model independently; never touch the live
            // converter from the loading thread. On a cache miss GPU builds may
            // still contend with inference, measured by the existing counters.
            let mut bank_done=false;
            if let Some(task)=bank_loading.as_mut() {
                loop {
                    match task.rx.try_recv() {
                        Ok(BankLoadMessage::Ready(i,voice))=>{
                            if let Some(current)=converter.as_mut() {
                                if let Err(e)=current.install_voice(i,voice) {
                                    task.errors.push(format!("{}: {e:#}",LABELS[i]));
                                    continue;
                                }
                                self.params.loaded_voice_mask.store(current.available_mask(),Ordering::Release);
                            }else{
                                let mut slots:[Option<Box<dyn WorkerConversion>>;4]=std::array::from_fn(|_|None);
                                slots[i]=Some(voice);
                                let bank=VoiceBank::new(slots,i).expect("just loaded one slot");
                                converter=Some(Box::new(bank));
                                let current=converter.as_mut().expect("bank");
                                chunk_samples=current.profile_hop().unwrap_or(chunk_samples);
                                zero_input.resize(chunk_samples,0.0);
                                self.state.input_hop.store(chunk_samples,Ordering::Release);
                                self.latency.store(u32::try_from(chunk_samples.saturating_add(current.content_delay())).unwrap_or(u32::MAX),Ordering::Relaxed);
                                // Discard audio collected before any model was ready.
                                input_acc.clear(); self.drain_input();
                                previous_input.clear(); previous_output_end=None;
                                pending_switch=None; transition_from=None;
                                self.publish_conversion(&task.settings,converter.as_deref());
                                self.loading.store(false,Ordering::Release);
                                self.state.notify_offline();
                                self.params.diagnostics.event("VOICE_FIRST_READY",&format!("slot={} elapsed_ms={:.2}",i+1,task.started.elapsed().as_secs_f64()*1000.0));
                            }
                            if self.params.selected_voice()==(i+1) as i32 {last_voice_request=usize::MAX;}
                            self.params.diagnostics.event("VOICE_SLOT_READY",&format!("slot={} mask={:#06b}",i+1,self.params.loaded_voice_mask.load(Ordering::Acquire)));
                        }
                        Ok(BankLoadMessage::Failed(i,err))=>{
                            self.params.diagnostics.event("VOICE_LOAD_ERROR",&format!("slot={} {err}",i+1));
                            task.errors.push(err);
                        }
                        Ok(BankLoadMessage::Done)=>{bank_done=true;break;}
                        Err(std::sync::mpsc::TryRecvError::Empty)=>break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected)=>{task.errors.push("Model loader disconnected".into());bank_done=true;break;}
                    }
                }
            }
            if bank_done {
                if let Some(task)=bank_loading.take(){
                    let count=converter.as_ref().map_or(0,|c|c.available_mask().count_ones());
                    let summary=format!("VoiceBank: {count} of 4 loaded");
                    if task.errors.is_empty(){self.set_status(summary)}else{self.set_error(summary,task.errors.join("\n"))}
                    self.params.diagnostics.event("VOICE_BANK_LOAD_DONE",&format!("ready={count} elapsed_ms={:.2}",task.started.elapsed().as_secs_f64()*1000.0));
                    if count==0 {self.dirty.store(true,Ordering::Release);}
                }
                self.loading.store(false,Ordering::Release); self.state.notify_offline();
            }

            let selected = self.params.selected_voice();
            let requested_voice = selected.saturating_sub(1) as usize;
            if selected == 0 {
                self.params.active_voice.store(0, Ordering::Release);
                last_voice_request=usize::MAX;
            } else if requested_voice != last_voice_request {
                last_voice_request = requested_voice;
                if let Some(current) = converter.as_mut() {
                    match current.select_voice(requested_voice) {
                        Ok(changed) => {
                            if changed {
                                let started = Instant::now();
                                let old_hop = chunk_samples;
                                let new_hop = current.profile_hop().unwrap_or(old_hop);
                                let continuous = !self.state.offline.load(Ordering::Acquire)
                                    && current.profile_hop().is_some()
                                    && new_hop == old_hop
                                    && self.state.generation.load(Ordering::Acquire) == generation;
                                chunk_samples = new_hop;
                                zero_input.resize(new_hop, 0.0);
                                self.state.input_hop.store(new_hop, Ordering::Release);
                                if continuous {
                                    // Clear the target's stale history, NOT the shared
                                    // microphone/output FIFO. Prime only from real recent
                                    // input, never reuse a previous character's model state.
                                    let prepared = current.reset().and_then(|()| {
                                        if previous_input.len() == new_hop {
                                            current.apply_live(&self.live_params());
                                            current.prime(&previous_input, self.sample_rate)?;
                                        }
                                        Ok(())
                                    });
                                    if let Err(err) = prepared {
                                        generation = self.fail_conversion("voice switch preparation failed", err);
                                        converter = None;
                                        input_acc.clear();
                                        previous_input.clear(); previous_output_end = None;
                                        pending_switch = None; transition_from = None;
                                        continue;
                                    }
                                    transition_from = previous_output_end;
                                    pending_switch = Some((selected, started));
                                    self.params.diagnostics.event("VOICE_SWITCH_PREPARED", &format!("slot={selected} hop={new_hop} prime_ms={:.2} retained_input={}", started.elapsed().as_secs_f64()*1000.0,input_acc.len()));
                                } else {
                                    // Host reset, offline render, or different hop:
                                    // preserve the existing strict generation boundary.
                                    self.state.generation.fetch_add(1, Ordering::AcqRel);
                                }
                                self.latency.store(u32::try_from(new_hop.saturating_add(current.content_delay())).unwrap_or(u32::MAX), Ordering::Relaxed);
                                self.set_status(format!("Voice: {}", LABELS[requested_voice]));
                            }
                            self.params.active_voice.store(selected, Ordering::Release);
                            self.params.diagnostics.event("VOICE_SELECTED",&format!("slot={selected} ready=true changed={changed}"));
                        }
                        Err(err) => {
                            self.params.active_voice.store(0, Ordering::Release);
                            self.set_error("Voice switch rejected; original voice fallback", format!("{err:#}"));
                        }
                    }
                }else{
                    self.params.active_voice.store(0,Ordering::Release);
                    self.params.diagnostics.event("VOICE_UNAVAILABLE",&format!("slot={selected}: converter not loaded"));
                }
            }
            let requested_generation = self.state.generation.load(Ordering::Acquire);
            if requested_generation != generation {
                input_acc.clear();
                previous_input.clear(); previous_output_end = None; pending_switch = None; transition_from = None;
                if let Some(current) = converter.as_mut() {
                    if let Err(err) = current.reset() {
                        generation = self.fail_conversion("stream reset failed", err);
                        converter = None;
                        continue;
                    }
                    if let Err(err) = self.prepare_offline(current.as_mut(), &zero_input) {
                        generation = self.fail_conversion("offline preroll failed", err);
                        converter = None;
                        continue;
                    }
                }
                generation = requested_generation;
                read_generation(&mut self.input_consumer, generation, &mut []);
                self.state
                    .ready_generation
                    .store(generation, Ordering::Release);
                self.state.notify_offline();
            }

            // Accumulate one input chunk.
            while input_acc.len() < chunk_samples {
                let needed = chunk_samples - input_acc.len();
                if self.input_consumer.slots() == 0 {
                    break;
                }
                let old_len = input_acc.len();
                input_acc.resize(old_len + needed, 0.0);
                let read = read_generation(
                    &mut self.input_consumer,
                    generation,
                    &mut input_acc[old_len..],
                );
                input_acc.truncate(old_len + read);
                if read == 0 {
                    break;
                }
            }
            if input_acc.len() < chunk_samples {
                // Re-check the stop flag before parking so a stop requested
                // between the loop head and here exits without waiting.
                if !self.running.load(Ordering::SeqCst) {
                    break;
                }
                // Wait for an unpark — input queued in process_block, a reload
                // submitted from the editor, or Drop — or the safety timeout.
                // Replaces the fixed 2 ms poll: a reload submitted while the host
                // is idle starts immediately, and there is no idle spin when the
                // host stops calling process().
                thread::park_timeout(Duration::from_millis(100));
                continue;
            }

            let chunk = &input_acc[..chunk_samples];
            let Some(current) = converter.as_mut() else {
                // No pipeline: discard input and stay silent.
                input_acc.clear();
                continue;
            };

            // Apply automatable parameters before converting this chunk. Builds
            // the same `LiveParams` the standalone worker does, so both drive the
            // single `apply_live` entry point rather than diverging set_* calls.
            current.apply_live(&self.live_params());
            let detailed = self.params.settings.try_read().map(|s|s.detailed_diagnostics).unwrap_or(false);
            let report_quality = detailed || last_chunk_log.elapsed() >= Duration::from_secs(5);
            current.quality_diagnostics(report_quality);
            let stats = match current.process(chunk, self.sample_rate, &mut chunk_out) {
                Ok(stats) => stats,
                Err(err) => {
                    generation = self.fail_conversion("chunk conversion failed", err);
                    converter = None;
                    input_acc.clear();
                    continue;
                }
            };
            if let Some(from) = transition_from.take() {
                soften_voice_transition(&mut chunk_out, from, self.sample_rate);
            }
            previous_input.clear();
            previous_input.extend_from_slice(chunk);
            previous_output_end = chunk_out.last().copied();
            self.params.diagnostics.record_chunk(&stats);
            let detailed = self.params.settings.try_read().map(|s|s.detailed_diagnostics).unwrap_or(false);
            if detailed || last_chunk_log.elapsed() >= Duration::from_secs(5) {
                let ms = |d: Duration| d.as_secs_f64() * 1000.0;
                self.params.diagnostics.event("INFERENCE", &format!("voice={} silent={} inference_ms={:.2} embedder_ms={:.2} f0_ms={:.2} rvc_ms={:.2} smoothing_ms={:.2} model_output_samples={} content_delay_samples={:?} {}", self.params.selected_voice(), stats.silent, ms(stats.inference_time), ms(stats.embedder_time), ms(stats.pitch_time), ms(stats.rvc_time), ms(stats.smoothing_time), stats.model_output_samples, stats.content_delay_samples, self.params.diagnostics.counters()));
                let peak = chunk_out.iter().fold(0.0_f32,|a,x|a.max(x.abs()));
                let clipped = chunk_out.iter().filter(|x|x.abs()>=1.0).count();
                let nonfinite = chunk_out.iter().filter(|x|!x.is_finite()).count();
                self.params.diagnostics.event("QUALITY",&format!("voice={} hop={} output_peak={peak:.6} output_clipped={clipped} output_nonfinite={nonfinite} {}", self.params.selected_voice(),chunk_samples,current.quality_report()));
                last_chunk_log = std::time::Instant::now();
            }
            // The shared converter now knows the model's native rate, actual
            // capped overlap, and both rate-adapter delays. Report these once
            // available; the callback only relays this precomputed atomic value.
            // SOLA's bounded search can advance audio within that nominal hold.
            let content_delay = current.content_delay();
            self.latency.store(
                u32::try_from(chunk_samples.saturating_add(content_delay)).unwrap_or(u32::MAX),
                Ordering::Relaxed,
            );
            input_acc.clear();
            if self.state.generation.load(Ordering::Acquire) != generation {
                continue;
            }
            queued_output.clear();
            queued_output.extend(
                chunk_out
                    .iter()
                    .map(|&value| QueuedSample { value, generation }),
            );
            let mut pushed = 0;
            while pushed < queued_output.len() {
                let (written, _) = self
                    .output_producer
                    .push_partial_slice(&queued_output[pushed..]);
                pushed += written.len();
                self.state.notify_offline();
                if pushed == queued_output.len()
                    || !self.state.offline.load(Ordering::Acquire)
                    || !self.running.load(Ordering::Acquire)
                    || self.state.generation.load(Ordering::Acquire) != generation
                {
                    break;
                }
                // Offline output is never dropped. Its callback consumes then
                // unparks us; realtime retains the bounded drop-on-overflow rule.
                thread::park_timeout(Duration::from_millis(100));
            }
            if !self.state.offline.load(Ordering::Acquire) {
                self.params.diagnostics.output_dropped.fetch_add(queued_output.len().saturating_sub(pushed) as u64, Ordering::Relaxed);
            }
            if pushed > 0 {
                if let Some((slot, started)) = pending_switch.take() {
                    self.params.diagnostics.event("VOICE_SWITCH_OUTPUT", &format!("slot={slot} first_enqueue_ms={:.2} queued_samples={pushed} {}",started.elapsed().as_secs_f64()*1000.0,self.params.diagnostics.counters()));
                }
            }
        }
        self.params.flush_profile();
        self.params.diagnostics.event("WORKER_STOP", &self.params.diagnostics.counters());
    }

    fn fail_conversion(&self, summary: &str, error: anyhow::Error) -> u32 {
        nice_plug::nice_error!("vc-vst3: {summary}: {error:#}");
        self.set_error(summary, format!("{error:#}"));
        // Silence every queued sample from the failed timeline, and release
        // offline waiters. Keep this worker alive to service explicit Reload;
        // retain the last successful settings for host-driven reinitialization.
        self.state.has_converter.store(false, Ordering::Release);
        self.params.loaded_voice_mask.store(0, Ordering::Release);
        self.params.active_voice.store(0, Ordering::Release);
        let generation = self
            .state
            .generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        self.state
            .ready_generation
            .store(generation, Ordering::Release);
        self.dirty.store(true, Ordering::SeqCst);
        self.state.notify_offline();
        generation
    }

    fn load_conversion(&self, settings:&PluginConfig, chunk_samples:usize)->Option<Box<dyn WorkerConversion>> {
        if settings.has_voice_bank() {
            let loader=self.model_loader();
            let mut slots:[Option<Box<dyn WorkerConversion>>;4]=std::array::from_fn(|_|None);
            let mut errors=Vec::new();
            for i in bank_load_order(self.params.selected_voice()) {
                match loader.load_voice_slot(settings,i) {
                    Ok(v)=>slots[i]=Some(v),
                    Err(e)=>errors.push(format!("{}: {e:#}",LABELS[i])),
                }
            }
            let requested=self.params.selected_voice().saturating_sub(1) as usize;
            let Some(bank)=VoiceBank::new(slots,requested) else {
                self.set_error("No voice bank model could load",errors.join("\n"));
                self.dirty.store(true,Ordering::SeqCst);
                return None;
            };
            let summary=format!("Voice: {} ({} of 4 loaded)",LABELS[bank.active()],4-errors.len());
            if errors.is_empty(){self.set_status(summary)}else{self.set_error(summary,errors.join("\n"))}
            return Some(Box::new(bank));
        }
        match self.load_current(settings, chunk_samples) {
            Ok((pipeline, kind)) => pipeline.map(|pipeline| {
                Box::new(self.chunk_converter(pipeline, kind, chunk_samples, settings))
                    as Box<dyn WorkerConversion>
            }),
            Err(()) => {
                self.dirty.store(true, Ordering::SeqCst);
                None
            }
        }
    }

    fn prepare_offline(
        &self,
        converter: &mut dyn WorkerConversion,
        zeros: &[f32],
    ) -> anyhow::Result<()> {
        if self.state.offline.load(Ordering::Acquire) {
            // A finite host render must preserve its first input hop. Reuse the
            // shared finite path's zero-hop prime; realtime keeps its existing
            // first-real-chunk startup policy. Reset/mode change owns both states.
            converter.apply_live(&self.live_params());
            converter.prime(zeros, self.sample_rate)?;
            self.latency.store(
                u32::try_from(zeros.len().saturating_add(converter.content_delay()))
                    .unwrap_or(u32::MAX),
                Ordering::Relaxed,
            );
        }
        Ok(())
    }

    fn live_params(&self) -> LiveParams {
        LiveParams {
            pitch_shift: self.params.pitch_shift.value(),
            speaker_id: self.params.speaker_id.value() as i64,
            input_gain: util::db_to_gain(self.params.input_gain_db.value()),
            output_gain: util::db_to_gain(self.params.output_gain_db.value()),
            noise_gate_enabled: self.params.noise_gate.value(),
            noise_gate_threshold: util::db_to_gain(self.params.noise_gate_threshold_db.value()),
        }
    }

    fn publish_conversion(&self, settings: &PluginConfig, converter: Option<&dyn WorkerConversion>) {
        let loaded=converter.is_some();
        let mask=converter.map_or(0,|c|c.available_mask());
        let requested=self.params.selected_voice();
        let active=converter.map_or(0,|c|if requested==0{0}else if let Some(slot)=c.active_slot(){if slot+1==requested as usize{requested}else{0}}else{requested});
        self.params.loaded_voice_mask.store(mask, Ordering::Release);
        self.params.active_voice.store(active, Ordering::Release);
        self.params.diagnostics.event("VOICE_READY",&format!("mask={mask:#06b} requested={requested} active={active}"));
        if loaded || !settings.has_models() {
            // A failed rebuild/reload does not erase the last successful Load
            // authorization. Explicitly clearing the model set does revoke it.
            *self.state.applied_settings.lock().unwrap() = loaded.then(|| settings.clone());
        }
        self.state.has_converter.store(loaded, Ordering::Release);
    }

    /// Discard everything currently queued in the input ring.
    fn drain_input(&mut self) {
        let backlog = self.input_consumer.slots();
        if backlog > 0 {
            if let Ok(chunk) = self.input_consumer.read_chunk(backlog) {
                chunk.commit_all();
            }
        }
    }

    fn set_status(&self, text: impl Into<String>) {
        let text = text.into();
        self.params.diagnostics.event("STATUS", &text);
        *self.params.model_message.lock().unwrap_or_else(|e|e.into_inner())=text.clone();
        if let Ok(mut status) = self.status.lock() {
            *status = PluginStatus::new(text);
        }
    }

    fn set_error(&self, summary: impl Into<String>, detail: impl Into<String>) {
        let summary = summary.into();
        let detail = detail.into();
        self.params.diagnostics.event("ERROR", &format!("{summary}: {detail}"));
        *self.params.model_message.lock().unwrap_or_else(|e|e.into_inner())=format!("{summary}: {detail}");
        if let Ok(mut status) = self.status.lock() {
            *status = PluginStatus {
                summary,
                detail: Some(detail),
            };
        }
    }

    fn report_load_progress(&self, progress: LoadProgress) {
        self.set_status(match progress {
            LoadProgress::Idle => "idle".to_string(),
            LoadProgress::ValidatingConfig => "validating configuration".to_string(),
            LoadProgress::PreparingProvider => "preparing execution provider".to_string(),
            LoadProgress::DownloadingProvider => "downloading execution provider".to_string(),
            LoadProgress::BuildingEngine { role } => {
                format!("building {} TensorRT engine", role.label())
            }
            LoadProgress::LoadingModel { role } => format!("loading {} model", role.label()),
            LoadProgress::OpeningAudioDevices => "opening audio devices".to_string(),
            LoadProgress::Running => "running".to_string(),
            LoadProgress::Failed => "failed".to_string(),
        });
    }

    fn set_idle_status(&self) {
        let settings = self.params.settings.read().unwrap();
        if settings.has_models() {
            self.set_status("models configured; click Load / Reload");
        } else {
            self.set_status("no models configured");
        }
    }

    fn output_extra_ms(&self) -> u32 {
        self.crossfade_ms
            .saturating_add(self.sola_search_ms)
            .saturating_add(self.tail_discard_ms)
    }

    fn latency_samples(&self, chunk_samples: usize) -> u32 {
        let extra = chunk_samples_for_rate(self.sample_rate, self.output_extra_ms());
        (chunk_samples + extra) as u32
    }

    fn chunk_converter(
        &self,
        pipeline: RvcPipeline,
        kind: SmoothingKind,
        chunk_samples: usize,
        settings: &PluginConfig,
    ) -> ChunkConverter<RvcPipeline> {
        ChunkConverter::new(
            pipeline,
            ChunkOutputConfig {
                kind,
                output_sample_rate: self.sample_rate,
                output_chunk_samples: chunk_samples,
                crossfade_ms: settings.crossfade_ms,
                sola_search_ms: settings.sola_search_ms,
                tail_discard_ms: settings.rvc_output_tail_discard_ms,
            },
        )
    }

    /// Build a pipeline from one settings snapshot, reporting status. Missing
    /// models are a valid silent configuration; load failures return `Err`.
    fn load_current(
        &self,
        settings: &PluginConfig,
        chunk_samples: usize,
    ) -> Result<(Option<RvcPipeline>, SmoothingKind), ()> {
        let kind = settings.smoothing_kind();
        if !settings.has_models() {
            nice_plug::nice_warn!("vc-vst3: no models configured; running silent");
            self.set_status("no models configured");
            return Ok((None, kind));
        }
        self.set_status("loading…");
        let provider = settings.provider();
        match self.load_pipeline(settings, provider, chunk_samples) {
            Ok(pipeline) => {
                self.set_status(format!("running ({})", provider.label()));
                Ok((Some(pipeline), kind))
            }
            Err(err) => {
                nice_plug::nice_error!("vc-vst3: failed to load RVC pipeline: {err:#}");
                self.set_error(format!("load failed: {err}"), format!("{err:#}"));
                Err(())
            }
        }
    }

    fn load_pipeline(&self, settings:&PluginConfig, provider:vc_core::Provider, chunk_samples:usize)->anyhow::Result<RvcPipeline> {
        self.model_loader().load_pipeline(settings,provider,chunk_samples)
    }
    fn model_loader(&self)->ModelLoader {
        ModelLoader{params:self.params.clone(),status:self.status.clone(),sample_rate:self.sample_rate}
    }

}


// The loader owns no audio rings. All GPU construction is off the callback and
// the inference worker; loaded converters are transferred exactly once.
struct ModelLoader {
    params:Arc<VcRvcParams>,
    status:Arc<Mutex<PluginStatus>>,
    sample_rate:u32,
}
impl ModelLoader {
    fn set_status(&self, text: impl Into<String>) {
        let text = text.into();
        self.params.diagnostics.event("STATUS", &text);
        *self.params.model_message.lock().unwrap_or_else(|e|e.into_inner())=text.clone();
        if let Ok(mut status) = self.status.lock() {
            *status = PluginStatus::new(text);
        }
    }

    fn set_error(&self, summary: impl Into<String>, detail: impl Into<String>) {
        let summary = summary.into();
        let detail = detail.into();
        self.params.diagnostics.event("ERROR", &format!("{summary}: {detail}"));
        *self.params.model_message.lock().unwrap_or_else(|e|e.into_inner())=format!("{summary}: {detail}");
        if let Ok(mut status) = self.status.lock() {
            *status = PluginStatus {
                summary,
                detail: Some(detail),
            };
        }
    }

    fn report_load_progress(&self, progress: LoadProgress) {
        self.set_status(match progress {
            LoadProgress::Idle => "idle".to_string(),
            LoadProgress::ValidatingConfig => "validating configuration".to_string(),
            LoadProgress::PreparingProvider => "preparing execution provider".to_string(),
            LoadProgress::DownloadingProvider => "downloading execution provider".to_string(),
            LoadProgress::BuildingEngine { role } => {
                format!("building {} TensorRT engine", role.label())
            }
            LoadProgress::LoadingModel { role } => format!("loading {} model", role.label()),
            LoadProgress::OpeningAudioDevices => "opening audio devices".to_string(),
            LoadProgress::Running => "running".to_string(),
            LoadProgress::Failed => "failed".to_string(),
        });
    }


    fn chunk_converter(
        &self,
        pipeline: RvcPipeline,
        kind: SmoothingKind,
        chunk_samples: usize,
        settings: &PluginConfig,
    ) -> ChunkConverter<RvcPipeline> {
        ChunkConverter::new(
            pipeline,
            ChunkOutputConfig {
                kind,
                output_sample_rate: self.sample_rate,
                output_chunk_samples: chunk_samples,
                crossfade_ms: settings.crossfade_ms,
                sola_search_ms: settings.sola_search_ms,
                tail_discard_ms: settings.rvc_output_tail_discard_ms,
            },
        )
    }


    fn load_pipeline(
        &self,
        settings: &PluginConfig,
        provider: vc_core::Provider,
        chunk_samples: usize,
    ) -> anyhow::Result<RvcPipeline> {
        if provider.is_tensorrt() {
            // Engine creation runs on this worker, never the audio thread.
            // Keep derived engines inside this plugin's folder, without
            // mutating the host process environment or global DLL search.
            if let Some(root) = crate::diagnostics::local_dir() {
                let cache = root.join("engine-cache");
                self.params.diagnostics.event("TENSORRT_CACHE", &cache.display().to_string());
                return vc_core::model_rvc::with_engine_cache_root(&cache, || {
                    self.load_pipeline_inner(settings, provider, chunk_samples)
                });
            }
        }
        if provider.is_cuda() {
            // This is deliberately on the worker's explicit Load / Reload path,
            // not plugin initialization or the realtime callback. It prevents a
            // DAW's PATH or already-installed CUDA stack from silently winning
            // DLL resolution before ONNX Runtime creates the CUDA EP session.
            crate::dll_path::preload_bundled_cuda_dlls()?;
            return crate::dll_path::with_bundled_dll_directory(|| {
                self.load_pipeline_inner(settings, provider, chunk_samples)
            });
        }
        if provider.is_windows_ml() {
            // Windows ML's small bootstrapper DLL is bundled beside the plugin,
            // while ONNX Runtime/DirectML come from Windows App SDK Runtime.
            // Keep this on the worker load path so the realtime callback never
            // performs package bootstrap or DLL resolution work.
            return crate::dll_path::with_bundled_dll_directory(|| {
                self.load_pipeline_inner(settings, provider, chunk_samples)
            });
        }
        self.load_pipeline_inner(settings, provider, chunk_samples)
    }

    fn load_pipeline_inner(
        &self,
        settings: &PluginConfig,
        provider: vc_core::Provider,
        chunk_samples: usize,
    ) -> anyhow::Result<RvcPipeline> {
        self.params.diagnostics.event("MODEL_LOAD", &format!("provider={} model={} embedder={} f0={} chunk_samples={} sample_rate={}", provider.label(), settings.model.display(), settings.embedder.display(), settings.f0_model.display(), chunk_samples, self.sample_rate));
        let report_progress = |progress| self.report_load_progress(progress);
        let mut pipeline = RvcPipeline::load(RvcPipelineConfig {
            model: &settings.model,
            embedder: &settings.embedder,
            embedder_output: settings.embedder_output.as_deref(),
            f0_model: &settings.f0_model,
            provider,
            gpu_priority: settings.gpu_priority(),
            gpu_device_id: settings.gpu_device_id,
            sample_rate: self.sample_rate,
            chunk_samples,
            // pitch / speaker / gains are DAW parameters; the worker applies the
            // current parameter values before every chunk, so these load-time
            // values are placeholders that get overwritten on the first chunk.
            speaker_id: 0,
            pitch_shift: 0.0,
            f0: F0Config {
                f0_threshold: settings.f0_threshold,
                silence_threshold: settings.silence_threshold,
                ..F0Config::default()
            },
            input_gain: 1.0,
            // Gate on/off + threshold are DAW parameters applied per chunk
            // (overwriting these load-time placeholders); attack/release/floor
            // are static and shape the gate built here.
            noise_gate_enabled: false,
            noise_gate_threshold: 0.01,
            noise_gate_shaping: NoiseGateShaping {
                attack_ms: settings.noise_gate_attack_ms,
                release_ms: settings.noise_gate_release_ms,
                floor: settings.noise_gate_floor,
            },
            output_extra_ms: settings.crossfade_ms.saturating_add(settings.sola_search_ms).saturating_add(settings.rvc_output_tail_discard_ms),
            volume_excluded_ms: settings.crossfade_ms,
            extra_convert_ms: settings.extra_convert_ms,
            output_gain: 1.0,
            output_dynamics: OutputDynamicsConfig {
                volume_envelope: settings.volume_envelope,
                rms_mix_rate: settings.rms_mix_rate,
                auto_output_gain: settings.auto_output_gain,
                target_output_rms: settings.target_output_rms,
                max_output_gain: settings.max_output_gain,
            },
            progress: Some(&report_progress),
        })?;
        if !settings.index_path.as_os_str().is_empty() && settings.index_rate > 0.0 {
            match pipeline.set_retrieval_index(&settings.index_path, settings.index_rate){
                Ok(())=>self.params.diagnostics.event("INDEX_LOADED", &format!("{} rate={}", settings.index_path.display(),settings.index_rate)),
                Err(err)=>self.params.diagnostics.event("INDEX_ERROR",&format!("{}: {err:#}; continuing RVC without retrieval",settings.index_path.display())),
            }
        }
        Ok(pipeline)
    }
    fn load_voice_slot(&self, settings:&PluginConfig, i:usize)->anyhow::Result<Box<dyn WorkerConversion>> {
        let started=Instant::now();
        let config=settings.for_voice(i);
        anyhow::ensure!(config.has_models(),"missing model, ContentVec or RMVPE");
        self.set_status(format!("Loading {}",LABELS[i]));
        let hop=config.validated_chunk_samples(self.sample_rate)?;
        let profile=&settings.voice_profiles[i];
        self.params.diagnostics.event("VOICE_PROFILE",&format!("slot={} pitch={} index_rate={} chunk_ms={} crossfade_ms={} f0_threshold={} extra_convert_ms={} input_gain_db={} output_gain_db={} protect={} formant_st={}",i+1,profile.pitch_shift,profile.index_rate,profile.chunk_ms,profile.crossfade_ms,profile.f0_threshold,profile.extra_convert_ms,profile.input_gain_db,profile.output_gain_db,profile.protect,profile.formant_shift));
        let mut pipeline=self.load_pipeline(&config,config.provider(),hop)?;
        pipeline.set_protect(profile.protect)?;
        pipeline.set_f0_ranges(profile.input_f0_min_hz,profile.input_f0_max_hz,profile.output_f0_min_hz,profile.output_f0_max_hz)?;
        self.params.diagnostics.event("VOICE_F0_RANGE",&format!("slot={} input_min_hz={} input_max_hz={} output_min_hz={} output_max_hz={}",i+1,profile.input_f0_min_hz,profile.input_f0_max_hz,profile.output_f0_min_hz,profile.output_f0_max_hz));
        pipeline.set_formant_shift(profile.formant_shift)?;
        pipeline.set_peak_protection(true);
        let inner=self.chunk_converter(pipeline,config.smoothing_kind(),hop,&config);
        self.params.diagnostics.event("VOICE_LOAD_DONE",&format!("slot={} load_ms={:.2}",i+1,started.elapsed().as_secs_f64()*1000.0));
        Ok(Box::new(ProfiledConversion{inner,live:profile.live(),hop}))
    }
}
fn bank_load_order(selected:i32)->[usize;4] {
    let first=if (1..=4).contains(&selected) {(selected-1) as usize}else{0};
    let mut order=[first,0,0,0];let mut pos=1;
    for i in 0..4 {if i!=first {order[pos]=i;pos+=1;}}
    order
}
enum BankLoadMessage {
    Ready(usize,Box<dyn WorkerConversion>),
    Failed(usize,String),
    Done,
}
struct BankLoadTask {
    rx:std::sync::mpsc::Receiver<BankLoadMessage>,
    cancel:Arc<AtomicBool>,
    handle:Option<JoinHandle<()>>,
    settings:PluginConfig,
    errors:Vec<String>,
    started:Instant,
}
impl BankLoadTask {
    fn start(loader:ModelLoader,settings:PluginConfig,selected:i32)->std::io::Result<Self> {
        let (tx,rx)=std::sync::mpsc::channel();
        let cancel=Arc::new(AtomicBool::new(false));
        let cancelled=cancel.clone();
        let snapshot=settings.clone();
        let worker=thread::current();
        let handle=thread::Builder::new().name("vc-vst3-model-loader".into()).spawn(move||{
            for i in bank_load_order(selected) {
                if cancelled.load(Ordering::Acquire){break;}
                let msg=match loader.load_voice_slot(&snapshot,i) {
                    Ok(v)=>BankLoadMessage::Ready(i,v),
                    Err(e)=>BankLoadMessage::Failed(i,format!("{}: {e:#}",LABELS[i])),
                };
                if tx.send(msg).is_err(){return;}
                worker.unpark();
            }
            let _=tx.send(BankLoadMessage::Done); worker.unpark();
        })?;
        Ok(Self{rx,cancel,handle:Some(handle),settings,errors:Vec::new(),started:Instant::now()})
    }
}
impl Drop for BankLoadTask {
    fn drop(&mut self) {
        self.cancel.store(true,Ordering::Release);
        if let Some(h)=self.handle.take(){let _=h.join();}
    }
}

#[cfg(test)]
pub(crate) fn test_loaded_runtime(
    params: Arc<VcRvcParams>,
    mode: ProcessMode,
    max_block: usize,
) -> PluginRuntime {
    tests::loaded_runtime(params, mode, max_block, tests::TestControl::default())
}

#[cfg(test)]
impl PluginRuntime {
    pub(crate) fn test_worker_id(&self) -> thread::ThreadId {
        self.worker_thread.id()
    }
    pub(crate) fn test_output_samples(&mut self) -> usize {
        self.output_consumer.slots()
    }
    pub(crate) fn test_input_hop(&self) -> usize {
        self.state.input_hop.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::time::Instant;
    use vc_core::model_rvc::{ContentDelay, ModelOutput, PassthroughModel, VoiceModel};

    const RATE: u32 = 16_000;
    const HOP: usize = 320;

    #[derive(Clone, Copy)]
    enum FailurePoint {
        Process,
        Reset,
        OfflinePrime,
    }

    struct FailingConversion {
        point: FailurePoint,
        primes: usize,
    }

    impl WorkerConversion for FailingConversion {
        fn apply_live(&mut self, _: &LiveParams) {}
        fn reset(&mut self) -> anyhow::Result<()> {
            if matches!(self.point, FailurePoint::Reset) {
                anyhow::bail!("injected reset failure");
            }
            Ok(())
        }
        fn prime(&mut self, _: &[f32], _: u32) -> anyhow::Result<()> {
            self.primes += 1;
            if self.primes > 1 && matches!(self.point, FailurePoint::OfflinePrime) {
                anyhow::bail!("injected preroll failure");
            }
            Ok(())
        }
        fn process(&mut self, _: &[f32], _: u32, _: &mut Vec<f32>) -> anyhow::Result<ChunkStats> {
            anyhow::bail!("injected conversion failure");
        }
        fn content_delay(&self) -> usize {
            0
        }
    }

    #[test]
    fn conversion_errors_release_audio_and_keep_reload_responsive() {
        for (point, mode, message) in [
            (
                FailurePoint::Process,
                ProcessMode::Offline,
                "chunk conversion failed",
            ),
            (
                FailurePoint::Process,
                ProcessMode::Realtime,
                "chunk conversion failed",
            ),
            (
                FailurePoint::Reset,
                ProcessMode::Offline,
                "stream reset failed",
            ),
            (
                FailurePoint::OfflinePrime,
                ProcessMode::Offline,
                "offline preroll failed",
            ),
        ] {
            let params = configured_params(0, 0, "sola");
            let applied = params.settings.read().unwrap().clone();
            let reload = Arc::new(AtomicBool::new(false));
            let loading = Arc::new(AtomicBool::new(false));
            let dirty = Arc::new(AtomicBool::new(false));
            let status = Arc::new(Mutex::new(PluginStatus::new("test")));
            let waker = Arc::new(ReloadWaker::default());
            let mut runtime = PluginRuntime::start_inner(
                params.clone(),
                reload.clone(),
                loading.clone(),
                dirty.clone(),
                status.clone(),
                &waker,
                RATE,
                HOP * 2,
                mode,
                None,
                Some(Box::new(FailingConversion { point, primes: 0 })),
            );
            wait_for(|| {
                runtime.state.has_converter.load(Ordering::Acquire)
                    && runtime.state.ready_generation.load(Ordering::Acquire) == 1
            });
            if matches!(point, FailurePoint::Process) {
                let mut audio = [0.25; HOP * 2];
                runtime.process_block(&mut [&mut audio]);
                assert_eq!(audio, [0.0; HOP * 2]);
            } else {
                runtime.reset();
            }
            wait_for(|| {
                !runtime.state.has_converter.load(Ordering::Acquire)
                    && dirty.load(Ordering::Acquire)
                    && runtime.state.ready_generation.load(Ordering::Acquire)
                        == runtime.state.generation.load(Ordering::Acquire)
            });
            assert!(runtime.running.load(Ordering::Acquire));
            assert!(dirty.load(Ordering::Acquire));
            assert!(runtime.state.generation.load(Ordering::Acquire) > 1);
            assert_eq!(status.lock().unwrap().summary, message);
            assert!(status
                .lock()
                .unwrap()
                .detail
                .as_ref()
                .unwrap()
                .contains("injected"));
            assert_eq!(runtime.applied_settings().unwrap(), applied);
            let mut silent = [0.5; HOP * 2];
            runtime.process_block(&mut [&mut silent]);
            assert_eq!(silent, [0.0; HOP * 2]);

            // Removing a model is a valid explicit reload with no inference.
            // It must be handled by the same worker after the backend error,
            // clearing both the loading flag and the previous load authority.
            params.settings.write().unwrap().model.clear();
            loading.store(true, Ordering::SeqCst);
            reload.store(true, Ordering::SeqCst);
            waker.wake();
            wait_for(|| !loading.load(Ordering::Acquire));
            assert!(!reload.load(Ordering::Acquire));
            assert_eq!(status.lock().unwrap().summary, "no models configured");
            assert!(!dirty.load(Ordering::Acquire));
            assert!(runtime.applied_settings().is_none());
        }
    }

    #[test]
    #[ignore = "requires VC_RS_DIAG_MODEL/EMBEDDER/F0/PROVIDER and the selected backend runtime"]
    fn reload_real_models_after_conversion_error() {
        let settings = PluginConfig {
            model: std::env::var_os("VC_RS_DIAG_MODEL")
                .expect("voice model")
                .into(),
            embedder: std::env::var_os("VC_RS_DIAG_EMBEDDER")
                .expect("embedder")
                .into(),
            f0_model: std::env::var_os("VC_RS_DIAG_F0").expect("F0 model").into(),
            provider: std::env::var("VC_RS_DIAG_PROVIDER").expect("explicit provider"),
            ..Default::default()
        };
        let rate = 48_000;
        let hop = settings.validated_chunk_samples(rate).unwrap();
        let params = Arc::new(VcRvcParams::default());
        *params.settings.write().unwrap() = settings.clone();
        let reload = Arc::new(AtomicBool::new(false));
        let loading = Arc::new(AtomicBool::new(false));
        let dirty = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(PluginStatus::new("test")));
        let waker = Arc::new(ReloadWaker::default());
        let mut runtime = PluginRuntime::start_inner(
            params,
            reload.clone(),
            loading.clone(),
            dirty.clone(),
            status.clone(),
            &waker,
            rate,
            hop * 2,
            ProcessMode::Offline,
            None,
            Some(Box::new(FailingConversion {
                point: FailurePoint::Process,
                primes: 0,
            })),
        );
        wait_for(|| {
            runtime.state.has_converter.load(Ordering::Acquire)
                && runtime.state.ready_generation.load(Ordering::Acquire) == 1
        });
        runtime.process_block(&mut [&mut vec![0.25; hop * 2]]);
        wait_for(|| {
            !runtime.state.has_converter.load(Ordering::Acquire) && dirty.load(Ordering::Acquire)
        });
        assert_eq!(status.lock().unwrap().summary, "chunk conversion failed");
        let worker = runtime.worker_thread.id();
        loading.store(true, Ordering::SeqCst);
        reload.store(true, Ordering::SeqCst);
        waker.wake();
        let deadline = Instant::now() + Duration::from_secs(180);
        let mut previous_status = String::new();
        while loading.load(Ordering::Acquire) {
            let current = status.lock().unwrap().summary.clone();
            if current != previous_status {
                println!("reload recovery: {current}");
                previous_status = current;
            }
            assert!(Instant::now() < deadline, "model reload timed out");
            thread::sleep(Duration::from_millis(20));
        }
        assert!(
            runtime.state.has_converter.load(Ordering::Acquire),
            "{:?}",
            status.lock().unwrap()
        );
        assert_eq!(runtime.worker_thread.id(), worker);
        assert!(!dirty.load(Ordering::Acquire));
        assert_eq!(runtime.applied_settings().unwrap(), settings);

        // No devices or recordings: exercise real inference on a fixed harmonic
        // waveform through the same offline host callback after explicit Reload.
        for block in 0..3 {
            let mut audio: Vec<f32> = (0..hop * 2)
                .map(|i| {
                    let phase = 2.0 * std::f32::consts::PI * 220.0 * (block * hop * 2 + i) as f32
                        / rate as f32;
                    0.1 * phase.sin() + 0.03 * (phase * 2.0).sin()
                })
                .collect();
            runtime.process_block(&mut [&mut audio]);
            assert!(audio.iter().all(|sample| sample.is_finite()));
            assert!(
                runtime.state.has_converter.load(Ordering::Acquire),
                "{:?}",
                status.lock().unwrap()
            );
        }
        println!(
            "{} recovered on the same worker; real audio is finite; latency={}",
            settings.provider,
            runtime.current_latency()
        );
    }

    #[derive(Default)]
    pub(super) struct TestControl {
        gate: Option<(Sender<()>, Receiver<()>)>,
        panic_on_audio: bool,
        resets: Arc<AtomicUsize>,
        input_delay: usize,
    }

    // A rolling identity generator with the same candidate/context and declared
    // delay contracts as RVC. Actual ChunkConverter/SOLA/PSOLA/output resampling
    // run in these tests; only neural inference is replaced.
    struct TestVoice {
        history: Vec<f32>,
        candidate_samples: usize,
        control: TestControl,
    }

    impl VoiceModel for TestVoice {
        fn input_content_delay(&self) -> ContentDelay {
            ContentDelay::from_samples(self.control.input_delay, RATE)
        }

        fn process(
            &mut self,
            input: &[f32],
            rate: u32,
            out: &mut Vec<f32>,
            pitch: &mut Vec<f32>,
        ) -> anyhow::Result<ModelOutput> {
            if input.iter().any(|&sample| sample != 0.0) {
                if let Some((started, release)) = self.control.gate.take() {
                    started.send(()).unwrap();
                    release.recv_timeout(Duration::from_secs(2)).unwrap();
                }
                assert!(!self.control.panic_on_audio, "intentional inference panic");
                thread::sleep(Duration::from_millis(1));
            }
            self.history.extend_from_slice(input);
            let needed = self.candidate_samples + self.control.input_delay;
            if self.history.len() < needed {
                let mut padded = vec![0.0; needed - self.history.len()];
                padded.append(&mut self.history);
                self.history = padded;
            }
            let end = self.history.len() - self.control.input_delay;
            out.clear();
            out.extend_from_slice(&self.history[end - self.candidate_samples..end]);
            if self.history.len() > needed {
                self.history.drain(..self.history.len() - needed);
            }
            let mut metadata_audio = Vec::new();
            let mut metadata = PassthroughModel.process(input, rate, &mut metadata_audio, pitch)?;
            pitch.resize(64, 220.0);
            metadata.raw_output_samples = out.len();
            Ok(metadata)
        }
    }

    struct TestConversion(ChunkConverter<TestVoice>);
    impl WorkerConversion for TestConversion {
        fn apply_live(&mut self, _: &LiveParams) {}
        fn reset(&mut self) -> anyhow::Result<()> {
            self.0.model_mut().history.clear();
            self.0
                .model_mut()
                .control
                .resets
                .fetch_add(1, Ordering::Relaxed);
            self.0.reset_streaming_state();
            Ok(())
        }
        fn prime(&mut self, input: &[f32], rate: u32) -> anyhow::Result<()> {
            self.0.prime(input, rate).map(|_| ())
        }
        fn process(
            &mut self,
            input: &[f32],
            rate: u32,
            out: &mut Vec<f32>,
        ) -> anyhow::Result<ChunkStats> {
            self.0.process_chunk(input, rate, out)
        }
        fn content_delay(&self) -> usize {
            self.0.output_content_delay_samples()
        }
    }

    pub(super) fn loaded_runtime(
        params: Arc<VcRvcParams>,
        mode: ProcessMode,
        max_block: usize,
        control: TestControl,
    ) -> PluginRuntime {
        let settings = params.settings.read().unwrap().clone();
        let extra = RATE as usize
            * (settings.crossfade_ms
                + settings.sola_search_ms
                + settings.rvc_output_tail_discard_ms) as usize
            / 1000;
        let hop = settings.validated_chunk_samples(RATE).unwrap();
        let converter = TestConversion(ChunkConverter::new(
            TestVoice {
                history: Vec::new(),
                candidate_samples: hop + extra,
                control,
            },
            ChunkOutputConfig {
                kind: settings.smoothing_kind(),
                output_sample_rate: RATE,
                output_chunk_samples: hop,
                crossfade_ms: settings.crossfade_ms,
                sola_search_ms: settings.sola_search_ms,
                tail_discard_ms: settings.rvc_output_tail_discard_ms,
            },
        ));
        let runtime = PluginRuntime::start_inner(
            params,
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(PluginStatus::new("test running"))),
            &Arc::new(ReloadWaker::default()),
            RATE,
            max_block,
            mode,
            None,
            Some(Box::new(converter)),
        );
        wait_for(|| {
            runtime.state.has_converter.load(Ordering::Acquire)
                && runtime.state.ready_generation.load(Ordering::Acquire) == 1
        });
        runtime
    }

    pub(crate) fn configured_params(fade: u32, tail: u32, kind: &str) -> Arc<VcRvcParams> {
        let params = Arc::new(VcRvcParams::default());
        *params.settings.write().unwrap() = PluginConfig {
            model: "test-rvc.onnx".into(),
            embedder: "test-embedder.onnx".into(),
            f0_model: "test-f0.onnx".into(),
            chunk_ms: 20,
            crossfade_ms: fade,
            sola_search_ms: 0,
            rvc_output_tail_discard_ms: tail,
            smoother: kind.into(),
            ..PluginConfig::default()
        };
        params
    }

    fn wait_for(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !predicate() {
            assert!(Instant::now() < deadline, "worker did not make progress");
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn render(runtime: &mut PluginRuntime, input: &[f32], blocks: &[usize]) -> Vec<f32> {
        let mut output = Vec::new();
        let mut offset = 0;
        let mut index = 0;
        while offset < input.len() {
            let take = blocks[index % blocks.len()].min(input.len() - offset);
            let mut block = input[offset..offset + take].to_vec();
            runtime.process_block(&mut [&mut block]);
            output.extend_from_slice(&block);
            offset += take;
            index += 1;
        }
        output
    }

    // The bank test uses real queues/worker and a deterministic model mock.
    // No GPU, scheduling deadline or perceived audio quality claim is made.
    struct SwitchMock {
        hop: usize,
        value: f32,
        seen: Arc<Mutex<Vec<f32>>>,
        primes: Arc<AtomicUsize>,
    }
    impl WorkerConversion for SwitchMock {
        fn profile_hop(&self) -> Option<usize> { Some(self.hop) }
        fn apply_live(&mut self, _: &LiveParams) {}
        fn reset(&mut self) -> anyhow::Result<()> { Ok(()) }
        fn prime(&mut self, input: &[f32], _: u32) -> anyhow::Result<()> {
            assert_eq!(input.len(), self.hop);
            assert!(input.iter().all(|v| *v == 0.2));
            self.primes.fetch_add(1, Ordering::Release);
            Ok(())
        }
        fn process(&mut self, input: &[f32], _: u32, out: &mut Vec<f32>) -> anyhow::Result<ChunkStats> {
            assert_eq!(input.len(), self.hop);
            self.seen.lock().unwrap().extend_from_slice(input);
            out.clear(); out.resize(self.hop,self.value);
            Ok(ChunkStats::default())
        }
        fn content_delay(&self) -> usize { 0 }
    }

    #[test]
    fn same_hop_switch_retains_queued_output_and_partial_input() {
        let params = configured_params(0,0,"sola");
        params.settings.write().unwrap().chunk_ms=50;
        let hop=800;
        let seen=Arc::new(Mutex::new(Vec::new()));
        let primes=Arc::new(AtomicUsize::new(0));
        let slots: [Option<Box<dyn WorkerConversion>>;4]=std::array::from_fn(|i| {
            Some(Box::new(SwitchMock{hop,value:0.2+i as f32*0.1,seen:seen.clone(),primes:primes.clone()}) as Box<dyn WorkerConversion>)
        });
        let bank=VoiceBank::new(slots,0).unwrap();
        let mut runtime=PluginRuntime::start_inner(
            params.clone(),Arc::new(AtomicBool::new(false)),Arc::new(AtomicBool::new(false)),Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(PluginStatus::new("test"))),&Arc::new(ReloadWaker::default()),RATE,hop,ProcessMode::Realtime,None,Some(Box::new(bank)));
        wait_for(||runtime.state.has_converter.load(Ordering::Acquire) && runtime.state.ready_generation.load(Ordering::Acquire)==1);
        runtime.process_block(&mut [&mut vec![0.2;hop]]);
        wait_for(||runtime.output_consumer.slots()>=hop);
        let mut half=vec![0.4;hop/2];
        runtime.process_block(&mut [&mut half]);
        assert!(half.iter().all(|v| *v==0.2));
        params.set_selected_voice(2); runtime.wake();
        wait_for(||params.active_voice.load(Ordering::Acquire)==2);
        assert_eq!(runtime.state.generation.load(Ordering::Acquire),1);
        assert_eq!(primes.load(Ordering::Acquire),1);
        let mut second=vec![0.5;hop/2];
        runtime.process_block(&mut [&mut second]);
        assert!(second.iter().all(|v| *v==0.2));
        wait_for(||runtime.output_consumer.slots()>=hop);
        let data=seen.lock().unwrap();
        assert_eq!(&data[..hop],&vec![0.2;hop]);
        assert_eq!(&data[hop..hop+hop/2],&vec![0.4;hop/2]);
        assert_eq!(&data[hop+hop/2..hop*2],&vec![0.5;hop/2]);
        drop(data);
        let mut next=vec![0.6;hop];
        runtime.process_block(&mut [&mut next]);
        assert!((next[0]-0.2).abs()<1e-6);
        assert!((next[hop-1]-0.3).abs()<1e-6);
        assert!(next.iter().all(|v| *v>=0.19));
    }

    #[test]
    fn startup_prioritizes_requested_slot_and_never_loads_a_slot_twice() {
        for selected in 0..=4 {
            let order=bank_load_order(selected);
            assert_eq!(order[0],selected.saturating_sub(1) as usize);
            let mut sorted=order;sorted.sort_unstable();
            assert_eq!(sorted,[0,1,2,3]);
        }
    }
    #[test]
    fn transition_is_five_ms_bounded_and_does_not_insert_silence() {
        let mut a=vec![0.3;2400];
        soften_voice_transition(&mut a,0.2,48000);
        assert!((a[0]-0.2).abs()<1e-6);
        assert_eq!(a[239],0.3);
        assert!(a[240..].iter().all(|v| *v==0.3));
    }

    #[test]
    fn fifty_ms_mock_pipeline_preserves_waveform_and_length() {
        let params = configured_params(40, 10, "phase-vocoder");
        params.settings.write().unwrap().chunk_ms = 50;
        let mut runtime = loaded_runtime(params, ProcessMode::Offline, 2048, TestControl::default());
        let hop = 800;
        assert_eq!(runtime.state.input_hop.load(Ordering::Acquire), hop);
        let latency = runtime.current_latency() as usize;
        // Actual smoothing clamps 40 ms to 37.5 ms for a 50 ms hop.
        assert_eq!(latency, hop + 600 + 160);
        let input: Vec<f32> = (0..hop*6+53).map(|i|0.2+0.1*(i as f32*0.017).cos()).collect();
        let mut padded = input.clone();
        padded.resize(input.len()+latency,0.0);
        let output = render(&mut runtime,&padded,&[37,512,63,17]);
        let error = output[latency..].iter().zip(&input).map(|(a,b)|(a-b).abs()).fold(0.0_f32,f32::max);
        println!("50ms mock phase-vocoder hop={hop} latency_samples={latency} max_error={error}");
        assert!(error<1e-5,"50ms waveform error={error}");
    }

    #[test]
    fn offline_freewheeling_preserves_first_short_and_final_partial_audio() {
        for kind in ["sola", "psola", "phase-vocoder"] {
            for len in [1, 37, HOP - 1, HOP + 1, HOP * 4 + 79] {
                let params = configured_params(10, 10, kind);
                let mut runtime = loaded_runtime(
                    params,
                    ProcessMode::Offline,
                    2048,
                    TestControl {
                        input_delay: 17,
                        ..TestControl::default()
                    },
                );
                let latency = runtime.current_latency() as usize;
                assert_eq!(latency, HOP + 160 + 160 + 17);
                assert_eq!(runtime.tail_samples(), Some(latency as u32));
                let mut input: Vec<f32> = (0..len)
                    .map(|i| 0.2 + 0.1 * (i as f32 * 0.017).cos())
                    .collect();
                input[0] = 0.8;
                input[len - 1] = 0.7;
                let mut host_input = input.clone();
                host_input.resize(len + latency, 0.0);
                let output = render(&mut runtime, &host_input, &[37, 1024, 63, 17]);
                let error = output[latency..]
                    .iter()
                    .zip(&input)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                assert!(error < 1e-5, "{kind} len={len} first/tail error={error}");
            }
        }
    }

    #[test]
    fn reset_rejects_queued_and_in_flight_audio_and_clears_model_context() {
        let params = configured_params(10, 0, "sola");
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let resets = Arc::new(AtomicUsize::new(0));
        let mut runtime = loaded_runtime(
            params,
            ProcessMode::Realtime,
            HOP,
            TestControl {
                gate: Some((started_tx, release_rx)),
                resets: Arc::clone(&resets),
                input_delay: HOP + 17,
                ..TestControl::default()
            },
        );
        runtime.process_block(&mut [&mut [0.75; HOP]]);
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        runtime.reset();
        let mut during_reset = [0.25; HOP];
        runtime.process_block(&mut [&mut during_reset]);
        assert!(during_reset.iter().all(|&value| value == 0.0));
        release_tx.send(()).unwrap();
        wait_for(|| {
            runtime.state.ready_generation.load(Ordering::Acquire) == 2
                && runtime.output_consumer.slots() >= HOP
        });
        assert_eq!(resets.load(Ordering::Relaxed), 1);
        let mut resumed = [0.0; HOP];
        runtime.process_block(&mut [&mut resumed]);
        assert!(resumed.iter().all(|&value| value == 0.0)); // fresh unprimed realtime policy
        wait_for(|| runtime.output_consumer.slots() >= HOP);
        let mut after_context_reset = [0.0; HOP];
        runtime.process_block(&mut [&mut after_context_reset]);
        assert!(after_context_reset
            .iter()
            .all(|&value| value.abs() <= 0.25 + 1e-6));
        wait_for(|| runtime.output_consumer.slots() >= HOP);
        runtime.reset();
        let mut queued_stale = [0.0; HOP];
        runtime.process_block(&mut [&mut queued_stale]);
        assert!(queued_stale.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn realtime_to_offline_discards_full_old_input_backlog_before_accepting_audio() {
        let params = configured_params(0, 0, "sola");
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut runtime = loaded_runtime(
            params,
            ProcessMode::Realtime,
            2048,
            TestControl {
                gate: Some((started_tx, release_rx)),
                ..TestControl::default()
            },
        );
        runtime.process_block(&mut [&mut [0.75; HOP]]);
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let old_backlog = vec![
            QueuedSample {
                value: 0.75,
                generation: 1
            };
            runtime.input_producer.slots()
        ];
        runtime
            .input_producer
            .push_entire_slice(&old_backlog)
            .unwrap();
        assert!(runtime.reconfigure(RATE, 2048, ProcessMode::Offline));
        release_tx.send(()).unwrap();
        let input = vec![0.25; HOP + 41];
        let mut host_input = input.clone();
        host_input.resize(input.len() + HOP, 0.0);
        let output = render(&mut runtime, &host_input, &[37, 1024, 63]);
        assert_eq!(&output[HOP..], input.as_slice());
    }

    #[test]
    fn offline_wait_returns_when_inference_panics() {
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            let params = configured_params(0, 0, "sola");
            let mut runtime = loaded_runtime(
                params,
                ProcessMode::Offline,
                HOP,
                TestControl {
                    panic_on_audio: true,
                    ..TestControl::default()
                },
            );
            runtime.process_block(&mut [&mut [0.25; HOP]]);
            let mut block = [0.25; HOP];
            runtime.process_block(&mut [&mut block]);
            assert!(
                !runtime.reconfigure(RATE, HOP, ProcessMode::Offline),
                "a dead worker must be replaced on host reinitialization"
            );
            done_tx.send(block).unwrap();
        });
        let output = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("offline host hung after worker panic");
        assert_eq!(output, [0.0; HOP]);
    }

    #[test]
    fn generation_read_preserves_future_input_and_handles_ring_wrap() {
        let (mut producer, mut consumer) = RingBuffer::<QueuedSample>::new(8);
        for _ in 0..6 {
            producer
                .push(QueuedSample {
                    value: 0.0,
                    generation: u32::MAX - 2,
                })
                .unwrap();
        }
        read_generation(&mut consumer, u32::MAX - 1, &mut []);
        for generation in [u32::MAX - 1, u32::MAX, 0, 1] {
            producer
                .push(QueuedSample {
                    value: generation as f32,
                    generation,
                })
                .unwrap();
        }
        let mut output = [0.0; 4];
        assert_eq!(read_generation(&mut consumer, 0, &mut output), 1);
        assert_eq!(consumer.slots(), 1); // generation 1 awaits its own reset
        producer
            .push(QueuedSample {
                value: 0.5,
                generation: 1,
            })
            .unwrap();
        assert_eq!(read_generation(&mut consumer, 1, &mut output), 2);
        assert_eq!(output[1], 0.5);
    }
}
