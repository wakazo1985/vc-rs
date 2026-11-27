//! Independent shortcut bindings. Windows polling runs only during active realtime
//! plugin life, never inside the audio callback. No hook, external process or key log.
use serde::{Deserialize, Serialize};

pub(crate) const ACTIONS: [&str; 6] = [
    "Original",
    "Toddler",
    "Girl",
    "Deep male",
    "Robot",
    "Hold echo",
];
pub(crate) const MODIFIERS: [(u8, &str); 4] = [(1, "Ctrl"), (2, "Alt"), (4, "Shift"), (8, "Win")];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Binding {
    pub key: u16,
    pub modifiers: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HotkeyConfig {
    pub enabled: bool,
    pub bindings: [Binding; 6],
}
impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bindings: std::array::from_fn(|i| Binding {
                key: 0x76 + i as u16,
                modifiers: 0,
            }),
        }
    }
}
pub(crate) fn supported_keys() -> impl Iterator<Item = u16> {
    std::iter::once(0)
        .chain(0x70..=0x87)
        .chain(0x41..=0x5a)
        .chain(0x30..=0x39)
        .chain(0x60..=0x69)
        .chain([
            0x20, 0x09, 0x0d, 0x1b, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2d, 0x2e,
        ])
}
pub(crate) fn key_name(key: u16) -> String {
    match key {
        0 => "Disabled".into(),
        0x70..=0x87 => format!("F{}", key - 0x70 + 1),
        0x41..=0x5a | 0x30..=0x39 => char::from_u32(key as u32).unwrap_or('?').to_string(),
        0x60..=0x69 => format!("Num {}", key - 0x60),
        0x20 => "Space".into(),
        0x09 => "Tab".into(),
        0x0d => "Enter".into(),
        0x1b => "Esc".into(),
        0x21 => "Page Up".into(),
        0x22 => "Page Down".into(),
        0x23 => "End".into(),
        0x24 => "Home".into(),
        0x25 => "Left".into(),
        0x26 => "Up".into(),
        0x27 => "Right".into(),
        0x28 => "Down".into(),
        0x2d => "Insert".into(),
        0x2e => "Delete".into(),
        _ => format!("VK {key:02X}"),
    }
}
impl HotkeyConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        for (i, binding) in self.bindings.iter().enumerate() {
            if binding.modifiers > 15 || !supported_keys().any(|key| key == binding.key) {
                return Err(format!("Invalid shortcut: {}", ACTIONS[i]));
            }
            if binding.key != 0 && self.bindings[..i].contains(binding) {
                return Err(format!("Duplicate shortcut: {}", ACTIONS[i]));
            }
        }
        Ok(())
    }
}
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Events {
    pub voice: Option<i32>,
    pub echo: bool,
}
#[derive(Default)]
struct KeyTracker {
    config: Option<HotkeyConfig>,
    previous: [bool; 6],
}
impl KeyTracker {
    fn update(
        &mut self,
        config: HotkeyConfig,
        modifiers: u8,
        mut down: impl FnMut(u16) -> bool,
    ) -> Events {
        let current = if config.enabled && config.validate().is_ok() {
            config.bindings.map(|binding| {
                binding.key != 0 && binding.modifiers == modifiers && down(binding.key)
            })
        } else {
            [false; 6]
        };
        if self.config != Some(config) {
            // Config updates and startup never manufacture a voice-selection press.
            self.config = Some(config);
            self.previous = current;
            return Events::default();
        }
        let mut result = Events {
            voice: None,
            echo: current[5],
        };
        for (i, pressed) in current.iter().take(5).enumerate() {
            if *pressed && !self.previous[i] {
                result.voice = Some(i as i32);
            }
        }
        self.previous = current;
        result
    }
}

#[cfg(windows)]
pub(crate) use worker::HotkeyWorker;
#[cfg(windows)]
mod worker {
    use super::*;
    use crate::{params::VcRvcParams, runtime::ReloadWaker};
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        thread::{self, JoinHandle},
        time::Duration,
    };
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    fn down(key: u16) -> bool {
        // Only the high bit is reliable; the low bit is shared by applications.
        unsafe { GetAsyncKeyState(i32::from(key)) < 0 }
    }
    fn modifiers() -> u8 {
        u8::from(down(0x11))
            | (u8::from(down(0x12)) << 1)
            | (u8::from(down(0x10)) << 2)
            | (u8::from(down(0x5b) || down(0x5c)) << 3)
    }
    pub(crate) struct HotkeyWorker {
        stop: Arc<AtomicBool>,
        params: Arc<VcRvcParams>,
        thread: Option<JoinHandle<()>>,
    }
    impl HotkeyWorker {
        pub(crate) fn start(
            params: Arc<VcRvcParams>,
            waker: Arc<ReloadWaker>,
        ) -> std::io::Result<Self> {
            let stop = Arc::new(AtomicBool::new(false));
            let flag = stop.clone();
            let worker_params = params.clone();
            let handle = thread::Builder::new()
                .name("voicebank-shortcuts".into())
                .spawn(move || {
                    let mut tracker = KeyTracker::default();
                    while !flag.load(Ordering::Acquire) {
                        // A transient settings lock is not a reason to lose a held key.
                        if let Ok(settings) = worker_params.settings.try_read() {
                            let config = settings.hotkeys;
                            drop(settings);
                            let events = tracker.update(config, modifiers(), down);
                            if let Some(voice) = events.voice {
                                worker_params
                                    .diagnostics
                                    .event("SHORTCUT", &format!("Voice slot {voice}"));
                                worker_params.set_selected_voice(voice);
                                waker.wake();
                            }
                            let previous = worker_params
                                .echo_hotkey
                                .swap(events.echo, Ordering::AcqRel);
                            if previous != events.echo {
                                worker_params.diagnostics.event(
                                    "SHORTCUT",
                                    if events.echo {
                                        "Echo pressed"
                                    } else {
                                        "Echo released"
                                    },
                                );
                            }
                        }
                        thread::park_timeout(Duration::from_millis(10));
                    }
                    worker_params.echo_hotkey.store(false, Ordering::Release);
                })?;
            Ok(Self {
                stop,
                params,
                thread: Some(handle),
            })
        }
    }
    impl Drop for HotkeyWorker {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(handle) = self.thread.take() {
                handle.thread().unpark();
                let _ = handle.join();
            }
            self.params.echo_hotkey.store(false, Ordering::Release);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_and_voice_edges() {
        let c = HotkeyConfig::default();
        assert!(c.validate().is_ok());
        assert_eq!(
            c.bindings.map(|b| b.key),
            [0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b]
        );
        let mut s = KeyTracker::default();
        assert_eq!(s.update(c, 0, |_| false), Events::default());
        for i in 0..5 {
            let key = c.bindings[i].key;
            assert_eq!(s.update(c, 0, |k| k == key).voice, Some(i as i32));
            assert_eq!(s.update(c, 0, |k| k == key).voice, None);
            s.update(c, 0, |_| false);
        }
    }
    #[test]
    fn echo_tracks_hold_without_changing_voice() {
        let c = HotkeyConfig::default();
        let mut s = KeyTracker::default();
        s.update(c, 0, |_| false);
        let e = s.update(c, 0, |k| k == 0x7b);
        assert!(e.echo && e.voice.is_none());
        assert!(s.update(c, 0, |k| k == 0x7b).echo);
        assert!(!s.update(c, 0, |_| false).echo);
        assert!(!s.update(c, 1, |k| k == 0x7b).echo);
    }
    #[test]
    fn remap_disable_duplicate_and_startup() {
        let mut c = HotkeyConfig::default();
        let mut s = KeyTracker::default();
        assert_eq!(s.update(c, 0, |k| k == 0x76).voice, None);
        c.bindings[0] = Binding {
            key: 0x41,
            modifiers: 1,
        };
        s.update(c, 0, |_| false);
        assert_eq!(s.update(c, 0, |k| k == 0x41).voice, None);
        assert_eq!(s.update(c, 1, |k| k == 0x41).voice, Some(0));
        c.bindings[1] = c.bindings[0];
        assert!(c.validate().is_err());
        assert_eq!(s.update(c, 1, |_| true), Events::default());
        c.enabled = false;
        assert_eq!(s.update(c, 0, |_| true), Events::default());
    }
}
