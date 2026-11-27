mod api;
mod cache;
mod chunk_converter;
mod f0_postprocess;
mod feature;
mod finite;
mod formant;
mod inspect;
pub(crate) mod native_tensorrt;
mod noise;
mod onnx_meta;
mod pipeline;
mod pitch;
mod process_priority;
mod retrieval;
mod sessions;
mod shape;
mod stream;
mod tensorrt;
mod time_state;

/// GPU scheduling priority requested for inference work.
///
/// Applied at two layers (see [`set_process_gpu_priority`]): a process-wide
/// Windows GPU scheduling priority class that affects every backend, plus a
/// native TensorRT CUDA *stream* priority on the TRT path. Both are scheduling
/// hints only — they do not guarantee execution order and do not prioritize
/// host/device transfers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuPriority {
    Normal,
    #[default]
    High,
}

pub use api::{ContentDelay, ModelOutput, PassthroughModel, VoiceModel};
pub use cache::{
    clear_engine_cache, engine_cache_info, engine_cache_root, ClearedEngineCache, EngineCacheEntry,
    EngineCacheInfo, ENGINE_CACHE_DIR_ENV,
};
pub use chunk_converter::{ChunkConverter, ChunkOutputConfig, ChunkStats};
pub use finite::{convert_finite, FiniteChunk, FiniteOutput};
pub use native_tensorrt::native_gpu_memory;
// Re-exported so the standalone front-ends can name the config when building
// `RvcPipelineConfig`; the processor itself stays private to the engine.
pub use f0_postprocess::F0PostprocessConfig;
pub use inspect::{
    inspect_model, read_model_metadata, validate_onnx_model, validate_rvc_model, ModelMetadata,
};
pub use pipeline::{
    F0Config, InputDenoiserConfig, LiveParams, LoadModelRole, LoadProgress, NoiseGateShaping,
    OutputDynamicsConfig, PreparedInputDenoiser, RvcPipeline, RvcPipelineConfig,
};
pub use process_priority::{set_process_gpu_priority, set_process_power_throttling};
pub use tensorrt::with_engine_cache_root;

#[cfg(test)]
mod tests;
