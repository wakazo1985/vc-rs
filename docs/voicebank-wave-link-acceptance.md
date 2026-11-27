# Wave Link 3 local acceptance (Windows x64)

This is NOT evidence of a successful Wave Link run. Execute on the actual PC with the local models and Windows App SDK Runtime >=2.1. No extra model downloads, DAW, Python, Stream Deck or virtual cable are required by this checklist.

- Keep the complete VST3 directory and matching module filename. Install into a Wave Link-recognized VST3 location. Ensure the regular Windows user can write the bundle root (settings.toml and logs/); do not run Wave Link elevated as a workaround. The preflight script checks the current user's access; it does not modify ACLs or install anything.
- Scan for effects, add VC-RS RVC (Windows ML) to the microphone input, open its editor. Confirm no crash. Set four local ONNX model paths and appropriate local ContentVec/RMVPE paths; Load / Reload and verify each slot reports success.
- F7: original, both mono and stereo as supported by the channel. F8-F11: press once, verify each actual model, including with editor closed. F12: hold and release while speaking on each voice, verify previous voice remains and echo stops.
- Change shortcut, save, close/reopen editor, restart Wave Link, verify saved setting and reload models as required. Verify settings.toml and logs/ exist inside actual VST3 directory. Check no duplicate hotkey events or previous voice leakage.
- Speak continuously for 60 seconds per model, switch repeatedly, check audible artifacts and realtime input/output dropped counts. Observe CPU, GPU engine and VRAM using Task Manager; test alongside normal streaming/game workload. Compare actual audio to stage 10 WAVs.
- If a failure occurs, retain the bundle's logs, Wave Link version, Windows build, provider, model name, failure operation and any audible recording the user elects to supply. Never claim a passed check without actual observation.

Host conformance on GitHub Actions uses the official Steinberg validator without real voice models. A pass cannot prove Wave Link or GPU compatibility.
