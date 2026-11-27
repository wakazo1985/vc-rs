//! Plugin settings: model paths and conversion defaults.
//!
//! These persist in the plugin state (via `#[persist]` on the params), so the
//! host saves/restores them per project/preset. A TOML config file is still
//! supported as a headless seed for fresh instances (see [`PluginConfig::discover`]).
//! Field names and defaults mirror the CLI `Run` arguments so a working CLI
//! setup transfers directly.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vc_core::model_rvc::GpuPriority;
use vc_core::validation::{
    validate_conversion_timing, validate_non_negative_f32, validate_rvc_chunk_ms,
    validate_unit_interval, ConversionTiming, ConversionTimingLimits, CONVERSION_TIMING_LIMITS,
};
use vc_core::Provider;

/// Search order for the config file:
/// 1. `VC_RS_VST3_CONFIG` environment variable (explicit path)
/// 2. `<os-config-dir>/vc-rs/vst3.toml` (see [`os_config_dir`])
/// 3. `vc-rs-vst3.toml` in the host's current working directory
pub const CONFIG_ENV: &str = "VC_RS_VST3_CONFIG";
pub const PLUGIN_MIN_EXTRA_CONVERT_MS: u32 = 100;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomEmbedder {
    pub path: PathBuf,
    pub output: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct VoiceSlot {
    pub model: PathBuf,
    pub index: PathBuf,
    pub embedder: PathBuf,
    pub f0_model: PathBuf,
}
/// Each slot owns all conversion settings. Protect is applied in the RVC
/// pipeline; formant adjustment is applied to the generated waveform before joining.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VoiceProfile {
    pub index_rate: f32,
    pub f0_threshold: f32,
    pub input_f0_min_hz: f32,
    pub input_f0_max_hz: f32,
    pub output_f0_min_hz: f32,
    pub output_f0_max_hz: f32,
    pub silence_threshold: f32,
    pub noise_gate_attack_ms: f32,
    pub noise_gate_release_ms: f32,
    pub noise_gate_floor: f32,
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub rvc_output_tail_discard_ms: u32,
    pub extra_convert_ms: u32,
    pub smoother: String,
    pub volume_envelope: bool,
    pub rms_mix_rate: f32,
    pub auto_output_gain: bool,
    pub target_output_rms: f32,
    pub max_output_gain: f32,
    pub pitch_shift: f32,
    pub speaker_id: i64,
    pub input_gain_db: f32,
    pub output_gain_db: f32,
    pub noise_gate_enabled: bool,
    pub noise_gate_threshold_db: f32,
    pub protect: f32,
    pub formant_shift: f32,
}
impl Default for VoiceProfile {
    fn default() -> Self {
        Self {
            index_rate: 0.5,
            f0_threshold: 0.3,
            input_f0_min_hz: 50.0,
            input_f0_max_hz: 1100.0,
            output_f0_min_hz: 50.0,
            output_f0_max_hz: 1100.0,
            silence_threshold: 0.0001,
            noise_gate_attack_ms: 5.0,
            noise_gate_release_ms: 50.0,
            noise_gate_floor: 0.0,
            chunk_ms: 50,
            crossfade_ms: 40,
            sola_search_ms: 12,
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            smoother: "phase-vocoder".to_string(),
            volume_envelope: false,
            rms_mix_rate: 0.0,
            auto_output_gain: false,
            target_output_rms: 0.03,
            max_output_gain: 512.0,
            pitch_shift: 0.0,
            speaker_id: 0,
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            noise_gate_enabled: false,
            noise_gate_threshold_db: -40.0,
            protect: 0.33,
            formant_shift: 0.0,
        }
    }
}
impl VoiceProfile {
    pub fn initial(slot: usize) -> Self {
        Self {
            pitch_shift: [14.0, 12.0, 0.0, 0.0][slot],
            index_rate: [0.4, 0.4, 0.5, 0.7][slot],
            protect: [0.25, 0.25, 0.33, 0.33][slot],
            // Provisional: favor the generated envelope over aggressive input RMS matching.
            rms_mix_rate: [0.75, 0.75, 0.8, 1.0][slot],
            ..Self::default()
        }
    }
    /// Migrate only a completely untouched stage 01/02 profile.
    pub fn upgrade_untouched(&mut self, slot: usize) -> bool {
        let updated = Self::initial(slot);
        let mut old = updated.clone();
        old.rms_mix_rate = 0.0;
        if *self == old {
            *self = updated;
            true
        } else {
            false
        }
    }
    pub fn live(&self) -> vc_core::model_rvc::LiveParams {
        vc_core::model_rvc::LiveParams {
            pitch_shift: self.pitch_shift,
            speaker_id: self.speaker_id,
            input_gain: 10.0_f32.powf(self.input_gain_db / 20.0),
            output_gain: 10.0_f32.powf(self.output_gain_db / 20.0),
            noise_gate_enabled: self.noise_gate_enabled,
            noise_gate_threshold: 10.0_f32.powf(self.noise_gate_threshold_db / 20.0),
        }
    }
    pub(crate) fn validate_live(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(self.smoother.as_str(), "sola" | "psola" | "phase-vocoder"),
            "Invalid smoother"
        );
        vc_core::model_rvc::F0PostprocessConfig::with_input_range(
            self.input_f0_min_hz,
            self.input_f0_max_hz,
        )?;
        vc_core::model_rvc::F0PostprocessConfig::with_input_range(
            self.output_f0_min_hz,
            self.output_f0_max_hz,
        )?;
        anyhow::ensure!(
            self.pitch_shift.is_finite() && (-24.0..=24.0).contains(&self.pitch_shift),
            "Invalid voice pitch"
        );
        anyhow::ensure!(
            (0..=255).contains(&self.speaker_id),
            "Invalid voice speaker"
        );
        for gain in [self.input_gain_db, self.output_gain_db] {
            anyhow::ensure!(
                gain.is_finite() && (-36.0..=36.0).contains(&gain),
                "Invalid voice gain"
            );
        }
        anyhow::ensure!(
            self.noise_gate_threshold_db.is_finite()
                && (-80.0..=0.0).contains(&self.noise_gate_threshold_db),
            "Invalid voice gate threshold"
        );
        anyhow::ensure!(
            self.protect.is_finite() && (0.0..=0.5).contains(&self.protect),
            "Invalid Protect"
        );
        anyhow::ensure!(
            self.formant_shift.is_finite() && (-12.0..=12.0).contains(&self.formant_shift),
            "Invalid formant shift"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
// Lenient: unknown/legacy keys (e.g. `pitch_shift`, now a DAW parameter) are
// ignored rather than rejected, so older config files keep parsing.
#[serde(default)]
pub struct PluginConfig {
    pub model: PathBuf,
    pub index_path: PathBuf,
    pub index_rate: f32,
    pub voice_slots: [VoiceSlot; 4],
    pub voice_profiles: [VoiceProfile; 4],
    pub hotkeys: crate::hotkeys::HotkeyConfig,
    pub startup_voice: i32,
    pub detailed_diagnostics: bool,
    pub embedder: PathBuf,
    pub f0_model: PathBuf,
    pub embedder_output: Option<String>,
    /// Optional managed identity; old paths remain Custom until explicitly changed.
    pub embedder_package: Option<vc_models::Selection>,
    /// Remember both path and advanced output override when leaving Custom.
    pub custom_embedder: Option<CustomEmbedder>,
    /// Legacy config key kept for lenient parsing only. TensorRT engine paths
    /// are no longer user-provided; native TensorRT builds cache entries from
    /// the ONNX model and fixed profile.
    pub rvc_engine: Option<PathBuf>,
    /// "cpu" | "windowsml*" | "cuda" | "tensorrt". GPU spellings resolve
    /// to whichever GPU-capable backend this package was built with (see
    /// [`PluginConfig::provider`]).
    pub provider: String,
    /// "high" | "normal". Only native TensorRT consumes this setting.
    pub gpu_priority: String,
    /// CUDA device ID used by CUDA and native TensorRT backends.
    pub gpu_device_id: u32,
    pub f0_threshold: f32,
    pub silence_threshold: f32,
    /// Noise gate attack/release/floor. Static (applied at Load/Reload); the
    /// gate's on/off and threshold are DAW parameters (see `VcRvcParams`).
    pub noise_gate_attack_ms: f32,
    pub noise_gate_release_ms: f32,
    pub noise_gate_floor: f32,
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub rvc_output_tail_discard_ms: u32,
    pub extra_convert_ms: u32,
    /// Config-only advanced choice: "sola" | "psola" | "phase-vocoder".
    /// Keep it in persisted state so older project/preset choices survive even
    /// though the editor has no selector. Applied on Load / Reload.
    pub smoother: String,
    pub volume_envelope: bool,
    pub rms_mix_rate: f32,
    pub auto_output_gain: bool,
    pub target_output_rms: f32,
    pub max_output_gain: f32,
}

impl Default for PluginConfig {
    fn default() -> Self {
        // Defaults track `crates/vc-cli/src/cli.rs` RunArgs.
        Self {
            model: PathBuf::new(),
            index_path: PathBuf::new(),
            index_rate: 0.5,
            voice_slots: std::array::from_fn(|_| VoiceSlot::default()),
            voice_profiles: std::array::from_fn(VoiceProfile::initial),
            hotkeys: Default::default(),
            startup_voice: 1,
            detailed_diagnostics: false,
            embedder: PathBuf::new(),
            f0_model: PathBuf::new(),
            embedder_output: None,
            embedder_package: None,
            custom_embedder: None,
            rvc_engine: None,
            provider: default_provider().to_string(),
            gpu_priority: "high".to_string(),
            gpu_device_id: 0,
            f0_threshold: 0.3,
            silence_threshold: 0.0001,
            noise_gate_attack_ms: 5.0,
            noise_gate_release_ms: 50.0,
            noise_gate_floor: 0.0,
            chunk_ms: 200,
            crossfade_ms: 40,
            sola_search_ms: 12,
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            smoother: "phase-vocoder".to_string(),
            volume_envelope: false,
            rms_mix_rate: 0.0,
            auto_output_gain: false,
            target_output_rms: 0.03,
            max_output_gain: 512.0,
        }
    }
}

impl PluginConfig {
    /// True when all three required model paths are set. When false the plugin
    /// runs in silent mode (the worker never loads a pipeline).
    pub fn has_models(&self) -> bool {
        if self.has_voice_bank() {
            self.voice_slots.iter().any(|slot| {
                !slot.model.as_os_str().is_empty()
                    && (!slot.embedder.as_os_str().is_empty()
                        || !self.embedder.as_os_str().is_empty())
                    && (!slot.f0_model.as_os_str().is_empty()
                        || !self.f0_model.as_os_str().is_empty())
            })
        } else {
            !self.model.as_os_str().is_empty()
                && !self.embedder.as_os_str().is_empty()
                && !self.f0_model.as_os_str().is_empty()
        }
    }

    pub fn has_voice_bank(&self) -> bool {
        self.voice_slots
            .iter()
            .any(|slot| !slot.model.as_os_str().is_empty())
    }

    pub fn for_voice(&self, slot: usize) -> Self {
        let voice = &self.voice_slots[slot];
        let mut config = self.clone();
        config.model = voice.model.clone();
        config.index_path = voice.index.clone();
        let profile = &self.voice_profiles[slot];
        config.index_rate = profile.index_rate;
        config.f0_threshold = profile.f0_threshold;
        config.silence_threshold = profile.silence_threshold;
        config.noise_gate_attack_ms = profile.noise_gate_attack_ms;
        config.noise_gate_release_ms = profile.noise_gate_release_ms;
        config.noise_gate_floor = profile.noise_gate_floor;
        config.chunk_ms = profile.chunk_ms;
        config.crossfade_ms = profile.crossfade_ms;
        config.sola_search_ms = profile.sola_search_ms;
        config.rvc_output_tail_discard_ms = profile.rvc_output_tail_discard_ms;
        config.extra_convert_ms = profile.extra_convert_ms;
        config.smoother = profile.smoother.clone();
        config.volume_envelope = profile.volume_envelope;
        config.rms_mix_rate = profile.rms_mix_rate;
        config.auto_output_gain = profile.auto_output_gain;
        config.target_output_rms = profile.target_output_rms;
        config.max_output_gain = profile.max_output_gain;
        if !voice.embedder.as_os_str().is_empty() {
            config.embedder = voice.embedder.clone();
            config.embedder_output = None;
            config.embedder_package = None;
        }
        if !voice.f0_model.as_os_str().is_empty() {
            config.f0_model = voice.f0_model.clone();
        }
        config.voice_slots = std::array::from_fn(|_| VoiceSlot::default());
        config
    }

    pub fn provider(&self) -> Provider {
        // Shared parser (canonical names + aliases); unknown spellings fall back
        // to CPU as before. Native GPU spellings ("cuda"/"tensorrt") still route
        // through `gpu_provider` so they resolve to whichever GPU backend this
        // package was compiled with; the windowsml catalog EPs pass through.
        match Provider::from_name(&self.provider) {
            Some(provider) if provider.is_cuda() || provider.is_tensorrt() => {
                gpu_provider(provider.label())
            }
            Some(provider) if provider.available_in_build() => provider,
            // A persisted Windows ML/CPU choice must never invoke an absent
            // backend in the native TensorRT-only replacement package.
            Some(_) | None => vc_core::default_provider(),
        }
    }

    pub fn smoothing_kind(&self) -> vc_core::sola::SmoothingKind {
        // Serde supplies the new default only for missing keys. Preserve saved
        // selections and the legacy SOLA fallback for unrecognized values.
        match self.smoother.trim().to_ascii_lowercase().as_str() {
            "psola" => vc_core::sola::SmoothingKind::Psola,
            "phase-vocoder" => vc_core::sola::SmoothingKind::PhaseVocoder,
            _ => vc_core::sola::SmoothingKind::Sola,
        }
    }

    pub fn gpu_priority(&self) -> GpuPriority {
        match self.gpu_priority.trim().to_ascii_lowercase().as_str() {
            "normal" => GpuPriority::Normal,
            _ => GpuPriority::High,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (0..=4).contains(&self.startup_voice),
            "Invalid startup voice"
        );
        self.hotkeys.validate().map_err(anyhow::Error::msg)?;
        if self.has_voice_bank() {
            for i in 0..4 {
                self.voice_profiles[i]
                    .validate_live()
                    .map_err(|e| anyhow::anyhow!("Voice {}: {e}", i + 1))?;
                self.for_voice(i)
                    .validate()
                    .map_err(|e| anyhow::anyhow!("Voice {}: {e}", i + 1))?;
            }
            return Ok(());
        }
        validate_unit_interval("INDEX mix rate", self.index_rate)?;
        let limits = ConversionTimingLimits {
            min_extra_convert_ms: PLUGIN_MIN_EXTRA_CONVERT_MS,
            ..CONVERSION_TIMING_LIMITS
        };
        validate_conversion_timing(
            ConversionTiming {
                chunk_ms: self.chunk_ms,
                crossfade_ms: self.crossfade_ms,
                sola_search_ms: self.sola_search_ms,
                tail_discard_ms: self.rvc_output_tail_discard_ms,
                extra_convert_ms: self.extra_convert_ms,
            },
            limits,
        )?;
        validate_rvc_chunk_ms(self.chunk_ms)?;
        validate_non_negative_f32("F0 threshold", self.f0_threshold)?;
        validate_non_negative_f32("silence threshold", self.silence_threshold)?;
        validate_non_negative_f32("noise gate attack (ms)", self.noise_gate_attack_ms)?;
        validate_non_negative_f32("noise gate release (ms)", self.noise_gate_release_ms)?;
        validate_unit_interval("noise gate floor", self.noise_gate_floor)?;
        validate_unit_interval("RMS mix rate", self.rms_mix_rate)?;
        validate_non_negative_f32("target output RMS", self.target_output_rms)?;
        validate_non_negative_f32("max output gain", self.max_output_gain)?;
        Ok(())
    }

    pub fn validated_chunk_samples(&self, sample_rate: u32) -> anyhow::Result<usize> {
        self.validate()?;
        if self.has_voice_bank() {
            // Validate every profile on the current host sample grid.
            for i in 0..4 {
                self.for_voice(i).validated_chunk_samples(sample_rate)?;
            }
            let slot = (self.startup_voice.saturating_sub(1) as usize).min(3);
            return self.for_voice(slot).validated_chunk_samples(sample_rate);
        }
        Ok(
            vc_core::validation::RvcChunkTiming::from_ms(self.chunk_ms, sample_rate)?
                .input_chunk_samples,
        )
    }

    /// Locate and parse the config file. Returns the default config when no file
    /// is found so the plugin still loads (in silent mode).
    pub fn discover() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => match toml::from_str::<PluginConfig>(&text) {
                Ok(config) => match config.validate() {
                    Ok(()) => {
                        nice_plug::nice_log!("vc-vst3: loaded config from {}", path.display());
                        config
                    }
                    Err(err) => {
                        nice_plug::nice_error!("vc-vst3: invalid config {}: {err}", path.display());
                        // Keep parseable settings visible so the worker can
                        // surface the error and the editor can correct them.
                        // Replacing 25 ms with defaults would hide the cause.
                        config
                    }
                },
                Err(err) => {
                    nice_plug::nice_error!("vc-vst3: failed to parse {}: {err}", path.display());
                    Self::default()
                }
            },
            Err(err) => {
                nice_plug::nice_error!("vc-vst3: failed to read {}: {err}", path.display());
                Self::default()
            }
        }
    }

    fn config_path() -> Option<PathBuf> {
        if let Some(explicit) = std::env::var_os(CONFIG_ENV) {
            let path = PathBuf::from(explicit);
            if path.is_file() {
                return Some(path);
            }
        }
        if let Some(path) = crate::diagnostics::profile_path() {
            if path.is_file() {
                return Some(path);
            }
        }
        if let Some(dir) = os_config_dir() {
            let path = dir.join("vc-rs").join("vst3.toml");
            if path.is_file() {
                return Some(path);
            }
        }
        let cwd = std::env::current_dir().ok()?.join("vc-rs-vst3.toml");
        cwd.is_file().then_some(cwd)
    }
}

// The default provider tracks the backend this package was built with, so a
// fresh instance is usable without first opening the GUI to pick one. Shared
// with the CLI/GUI via `vc_core::default_provider`; the variants are mutually
// exclusive by cargo feature (one per distributed package), and a CPU-only
// build with none of them falls back to "cpu".
fn default_provider() -> &'static str {
    vc_core::default_provider().label()
}

/// Resolve a requested GPU provider ("cuda" or "tensorrt") to the GPU backend
/// this package was compiled with. The variants are mutually exclusive by cargo
/// feature, so each build sees exactly one of these.
#[cfg(feature = "tensorrt")]
fn gpu_provider(requested: &str) -> Provider {
    if requested != "tensorrt" {
        nice_plug::nice_warn!(
            "vc-vst3: '{requested}' provider is not enabled in this package; using TensorRT"
        );
    }
    Provider::TensorRt
}

#[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
fn gpu_provider(requested: &str) -> Provider {
    if requested != "cuda" {
        nice_plug::nice_warn!(
            "vc-vst3: '{requested}' provider is not enabled in this package; using CUDA"
        );
    }
    Provider::Cuda
}

#[cfg(all(
    feature = "windowsml",
    not(any(feature = "cuda", feature = "tensorrt"))
))]
fn gpu_provider(requested: &str) -> Provider {
    nice_plug::nice_warn!(
        "vc-vst3: '{requested}' provider is not enabled in this package; using Windows ML"
    );
    Provider::WindowsMl
}

#[cfg(not(any(feature = "cuda", feature = "tensorrt", feature = "windowsml")))]
fn gpu_provider(requested: &str) -> Provider {
    nice_plug::nice_warn!(
        "vc-vst3: '{requested}' provider is not enabled in this CPU-only package; using CPU"
    );
    Provider::Cpu
}

/// The per-user config directory for the current OS:
/// `%APPDATA%` on Windows, `$XDG_CONFIG_HOME` (or `$HOME/.config`) elsewhere.
#[cfg(windows)]
fn os_config_dir() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from)
}

#[cfg(not(windows))]
fn os_config_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn stage03_defaults_migrate_only_untouched_profiles() {
        for i in 0..4 {
            let new = super::VoiceProfile::initial(i);
            assert_eq!(new.chunk_ms, 50);
            assert_eq!(new.rms_mix_rate, [0.75, 0.75, 0.8, 1.0][i]);
            let mut old = new.clone();
            old.rms_mix_rate = 0.0;
            let mut edited = old.clone();
            edited.pitch_shift += 1.0;
            let before = edited.clone();
            assert!(old.upgrade_untouched(i));
            assert_eq!(old, new);
            assert!(!edited.upgrade_untouched(i));
            assert_eq!(edited, before);
        }
    }
    #[test]
    fn smoother_defaults_only_when_missing_and_preserves_saved_choices() {
        use vc_core::sola::SmoothingKind;
        for (source, expected) in [
            ("", SmoothingKind::PhaseVocoder),
            ("smoother = 'sola'", SmoothingKind::Sola),
            ("smoother = 'psola'", SmoothingKind::Psola),
            ("smoother = 'phase-vocoder'", SmoothingKind::PhaseVocoder),
            ("smoother = 'unknown'", SmoothingKind::Sola),
        ] {
            let settings: super::PluginConfig = toml::from_str(source).unwrap();
            assert_eq!(settings.smoothing_kind(), expected);
            let saved = toml::to_string(&settings).unwrap();
            let restored: super::PluginConfig = toml::from_str(&saved).unwrap();
            assert_eq!(restored.smoother, settings.smoother);
            assert_eq!(restored.smoothing_kind(), expected);
        }
    }

    #[cfg(all(windows, feature = "windowsml"))]
    #[test]
    fn openvino_device_selection_survives_plugin_state_round_trip() {
        for name in [
            "windowsml-openvino",
            "windowsml-openvino-cpu",
            "windowsml-openvino-gpu",
            "windowsml-openvino-npu",
        ] {
            let config: super::PluginConfig =
                toml::from_str(&format!("provider = '{name}'")).unwrap();
            let restored: super::PluginConfig =
                toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
            assert_eq!(restored.provider().label(), name);
        }
    }
    use super::*;

    #[cfg(all(
        feature = "tensorrt",
        not(any(feature = "cuda", feature = "windowsml"))
    ))]
    #[test]
    fn old_windowsml_profile_uses_native_tensorrt() {
        for name in [
            "windowsml",
            "windowsml-directml",
            "windowsml-cpu",
            "cpu",
            "cuda",
            "tensorrt",
        ] {
            let config: PluginConfig = toml::from_str(&format!("provider = '{name}'")).unwrap();
            assert_eq!(config.provider(), Provider::TensorRt);
        }
    }

    #[test]
    fn gpu_priority_defaults_high_and_parses_normal() {
        assert_eq!(PluginConfig::default().gpu_priority(), GpuPriority::High);
        assert_eq!(PluginConfig::default().gpu_device_id, 0);
        let config: PluginConfig = toml::from_str("gpu_priority = \"normal\"").unwrap();
        assert_eq!(config.gpu_priority(), GpuPriority::Normal);
        let config: PluginConfig = toml::from_str("gpu_device_id = 2").unwrap();
        assert_eq!(config.gpu_device_id, 2);
    }

    #[test]
    fn independent_profiles_are_migrated_and_persisted() {
        let mut c: PluginConfig = toml::from_str("index_rate = 0.9").unwrap();
        for i in 0..4 {
            c.voice_slots[i].model = PathBuf::from(format!("voice-{i}.onnx"));
        }
        assert_eq!(
            std::array::from_fn::<_, 4, _>(|i| c.voice_profiles[i].pitch_shift),
            [14.0, 12.0, 0.0, 0.0]
        );
        let before = c.voice_profiles.clone();
        c.voice_profiles[1].pitch_shift = 9.0;
        c.voice_profiles[1].chunk_ms = 100;
        c.voice_profiles[1].crossfade_ms = 20;
        c.voice_profiles[1].index_rate = 0.2;
        c.voice_profiles[1].output_gain_db = -6.0;
        c.voice_profiles[1].input_f0_min_hz = 70.0;
        c.voice_profiles[1].input_f0_max_hz = 450.0;
        c.voice_profiles[1].output_f0_min_hz = 100.0;
        c.voice_profiles[1].output_f0_max_hz = 800.0;
        for i in [0, 2, 3] {
            assert_eq!(c.voice_profiles[i], before[i]);
        }
        let d: PluginConfig = toml::from_str(&toml::to_string(&c).unwrap()).unwrap();
        assert_eq!(c, d);
        assert_eq!(
            d.for_voice(1).validated_chunk_samples(48_000).unwrap(),
            4800
        );
        assert_eq!(
            d.for_voice(0).validated_chunk_samples(48_000).unwrap(),
            2400
        );
        for i in 0..4 {
            assert_eq!(VoiceProfile::initial(i).chunk_ms, 50);
        }
        assert_eq!(d.for_voice(1).index_rate, 0.2);
        assert_eq!(d.voice_profiles[1].live().pitch_shift, 9.0);
        assert!((d.voice_profiles[1].live().output_gain - 0.5011872).abs() < 0.00001);
        assert_eq!(d.for_voice(0).index_rate, 0.4);
        assert_eq!(d.voice_profiles[1].input_f0_min_hz, 70.0);
        assert_eq!(d.voice_profiles[0].input_f0_min_hz, 50.0);
        assert_eq!(d.voice_profiles[1].output_f0_max_hz, 800.0);
        assert!(c.validate().is_ok());
        c.voice_profiles[1].output_f0_max_hz = 90.0;
        assert!(c.validate().is_err());
        c.voice_profiles[1].output_f0_max_hz = 800.0;
        c.voice_profiles[1].chunk_ms = 19;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validates_timing_ranges() {
        assert!(PluginConfig::default().validate().is_ok());
        let config: PluginConfig = toml::from_str("chunk_ms = 19").unwrap();
        assert!(config.validate().is_err());
        let config: PluginConfig = toml::from_str(&format!(
            "extra_convert_ms = {}",
            PLUGIN_MIN_EXTRA_CONVERT_MS - 1
        ))
        .unwrap();
        assert!(config.validate().is_err());
        let config: PluginConfig = toml::from_str("extra_convert_ms = 3001").unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn persisted_chunk_settings_obey_rvc_frame_and_host_sample_grids() {
        let config: PluginConfig = toml::from_str("chunk_ms = 25").unwrap();
        assert_eq!(config.chunk_ms, 25);
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("multiple of 10"));
        let config: PluginConfig = toml::from_str("chunk_ms = 30").unwrap();
        assert_eq!(config.validated_chunk_samples(44_100).unwrap(), 1323);
        assert!(config.validated_chunk_samples(22_050).is_err());
        let config: PluginConfig = toml::from_str("chunk_ms = 20").unwrap();
        assert_eq!(config.validated_chunk_samples(22_050).unwrap(), 441);
        assert_eq!(
            PluginConfig::default()
                .validated_chunk_samples(48_000)
                .unwrap(),
            9_600
        );
    }
}
