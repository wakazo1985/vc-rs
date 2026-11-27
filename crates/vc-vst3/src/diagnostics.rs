//! Local diagnostic log and settings backup. Never performs I/O in process().
use crate::config::PluginConfig;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static PROFILE_LOCK: Mutex<()> = Mutex::new(());

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
// Resolve the bundle from the actual loaded module, never from the host's
// executable, working directory, or AppData. A flat VST3 uses its own directory.
fn storage_root(module_dir: &Path) -> PathBuf {
    if let Some(bundle) = module_dir.ancestors().find(|p| {
        p.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("vst3"))
    }) {
        bundle.to_path_buf()
    } else {
        module_dir.to_path_buf()
    }
}
pub(crate) fn local_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        crate::dll_path::plugin_dir().map(|dir| storage_root(&dir))
    }
    #[cfg(not(windows))]
    {
        None
    }
}
pub(crate) fn profile_path() -> Option<PathBuf> {
    local_dir().map(|p| p.join("settings.toml"))
}

fn save_to(path: &Path, config: &PluginConfig) -> anyhow::Result<()> {
    let _guard = PROFILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Profile has no parent"))?;
    fs::create_dir_all(parent)?;
    let text = toml::to_string_pretty(config)?;
    let temp = parent.join(format!(
        "settings-{}-{}.tmp",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
pub(crate) fn save_profile(config: &PluginConfig) -> anyhow::Result<()> {
    let path = profile_path().ok_or_else(|| anyhow::anyhow!("VST3 folder unavailable"))?;
    save_to(&path, config)
}
pub(crate) fn load_profile() -> anyhow::Result<PluginConfig> {
    let path = profile_path().ok_or_else(|| anyhow::anyhow!("VST3 folder unavailable"))?;
    let config: PluginConfig = toml::from_str(&fs::read_to_string(path)?)?;
    config.validate()?;
    Ok(config)
}

pub(crate) struct Diagnostics {
    path: Option<PathBuf>,
    file_guard: Mutex<()>,
    last_error: Mutex<Option<String>>,
    profile_error: Mutex<Option<String>>,
    pub runtime_error: Mutex<Option<String>>,
    pub host_latency_samples: AtomicU64,
    pub host_sample_rate: AtomicU64,
    pub last_switch_us: AtomicU64,
    pub switch_count: AtomicU64,
    pub input_dropped: AtomicU64,
    pub output_missing: AtomicU64,
    pub output_dropped: AtomicU64,
    pub chunks: AtomicU64,
    pub last_processing_us: AtomicU64,
    pub last_input_rms_bits: AtomicU64,
    pub last_output_rms_bits: AtomicU64,
}
impl Default for Diagnostics {
    fn default() -> Self {
        let path = local_dir().map(|dir| {
            dir.join("logs").join(format!(
                "voicebank-{}-{}-{}.log",
                timestamp_ms(),
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ))
        });
        Self {
            path,
            file_guard: Mutex::new(()),
            last_error: Mutex::new(None),
            profile_error: Mutex::new(None),
            runtime_error: Mutex::new(None),
            host_latency_samples: AtomicU64::new(0),
            host_sample_rate: AtomicU64::new(0),
            last_switch_us: AtomicU64::new(0),
            switch_count: AtomicU64::new(0),
            input_dropped: AtomicU64::new(0),
            output_missing: AtomicU64::new(0),
            output_dropped: AtomicU64::new(0),
            chunks: AtomicU64::new(0),
            last_processing_us: AtomicU64::new(0),
            last_input_rms_bits: AtomicU64::new(0),
            last_output_rms_bits: AtomicU64::new(0),
        }
    }
}
impl Diagnostics {
    pub(crate) fn path(&self) -> String {
        self.path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Unavailable".into())
    }
    pub(crate) fn error(&self) -> Option<String> {
        self.last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    pub(crate) fn profile_error(&self) -> Option<String> {
        self.profile_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    pub(crate) fn set_profile_error(&self, error: Option<String>) {
        *self.profile_error.lock().unwrap_or_else(|e| e.into_inner()) = error;
    }
    /// Worker/UI/control thread only. Rotate at 8 MiB, retaining one previous file.
    pub(crate) fn event(&self, category: &str, detail: &str) {
        if matches!(
            category,
            "ERROR" | "VOICE_LOAD_ERROR" | "INDEX_ERROR" | "HTTP_ERROR" | "GPU_MEMORY_UNAVAILABLE"
        ) {
            *self.runtime_error.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(format!("{category}: {detail}"));
        }
        if cfg!(test) {
            return;
        }
        let _guard = self.file_guard.lock().unwrap_or_else(|e| e.into_inner());
        let result = (|| -> anyhow::Result<()> {
            let path = self
                .path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Log directory unavailable"))?;
            fs::create_dir_all(
                path.parent()
                    .ok_or_else(|| anyhow::anyhow!("Invalid log path"))?,
            )?;
            if fs::metadata(path).map(|m| m.len()).unwrap_or(0) >= 8 * 1024 * 1024 {
                let previous = path.with_extension("previous.log");
                fs::rename(path, previous)?;
            }
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            writeln!(
                file,
                "{}\t{}\t{}",
                timestamp_ms(),
                category,
                detail.replace('\n', "\\n").replace('\r', "\\r")
            )?;
            Ok(())
        })();
        *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) =
            result.err().map(|e| format!("{e:#}"));
    }
    pub(crate) fn counters(&self) -> String {
        format!("chunks={} input_dropped={} output_missing={} output_dropped={} processing_ms={:.2} input_rms={:.6} output_rms={:.6}",
            self.chunks.load(Ordering::Relaxed), self.input_dropped.load(Ordering::Relaxed),
            self.output_missing.load(Ordering::Relaxed), self.output_dropped.load(Ordering::Relaxed),
            self.last_processing_us.load(Ordering::Relaxed) as f64 / 1000.0,
            f32::from_bits(self.last_input_rms_bits.load(Ordering::Relaxed) as u32),
            f32::from_bits(self.last_output_rms_bits.load(Ordering::Relaxed) as u32))
    }
    pub(crate) fn record_chunk(&self, stats: &vc_core::model_rvc::ChunkStats) {
        self.chunks.fetch_add(1, Ordering::Relaxed);
        self.last_processing_us.store(
            stats.processing_time.as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.last_input_rms_bits
            .store(stats.input_rms.to_bits() as u64, Ordering::Relaxed);
        self.last_output_rms_bits
            .store(stats.output_rms.to_bits() as u64, Ordering::Relaxed);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundle_storage_root_and_flat_vst3_folder() {
        let bundle = PathBuf::from("plugins/vc-vst3.vst3");
        assert_eq!(
            storage_root(&bundle.join("Contents").join("x86_64-win")),
            bundle
        );
        let flat = PathBuf::from("plugins");
        assert_eq!(storage_root(&flat), flat);
    }
    #[test]
    fn local_profile_roundtrip_and_replacement() {
        let dir = std::env::temp_dir().join(format!(
            "voicebank-profile-test-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let path = dir.join("settings.toml");
        let mut config = PluginConfig::default();
        config.voice_slots[1].model = PathBuf::from("C:/voices/girl.onnx");
        config.hotkeys.bindings[0].key = 0x41;
        config.hotkeys.bindings[0].modifiers = 1;
        config.startup_voice = 2;
        save_to(&path, &config).unwrap();
        let decoded: PluginConfig = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(decoded, config);
        config.startup_voice = 0;
        save_to(&path, &config).unwrap();
        let decoded: PluginConfig = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(decoded, config);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        let _ = fs::remove_dir_all(dir);
    }
}
