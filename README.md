# UjiCache

UjiCache is a Forge Neo extension for experimenting with residual reuse and residual prediction during Anima / Cosmos-Predict2 T2I inference.

It was split out from the broader `Nz-Anima-PredLab` experiment set. This repository now exposes only the UjiCache prototype and a compact debug log mode.

## Current Features

- Forge Neo AlwaysVisible panel: `UjiCache`
- Anima / Cosmos-Predict2 model detection
- UjiCache residual cache experiment for Anima block skipping
- Prediction formulas: `TeaCache (residual only)`, `Linear extrapolation`, and `Taylor2 curve`
- Coefficient profiles: daraskme legacy, Identity, and 24 re-calibrated Forge Neo presets; selecting a profile loosely sets the Start/End progress window
- Read-only `p_Anima(x)` display of the active polynomial coefficients
- Auto Uji mode for CSV-driven UjiCache parameter sweeps
- Debug log mode with timing logs, diagnose logs, optional UjiCache residual dump, and `Capture calibration pairs` (per-step rel_l1/out_rel JSONL for coefficient re-fitting)
- Runtime patch restore on disable, unsupported model, or unload

Logs are printed to the StabilityMatrix / Forge Neo console with the `[UjiCache]` prefix.

## Notes

UjiCache uses TeaCache-style skip decisions internally, but the standalone TeaCache experiment UI and all other PredLab experiments have been removed. The first model call is always a full calculation, and UjiCache falls back instead of using the cache when `previous_residual` is missing.

Forge Neo may pass unused kwargs such as `control` into `Anima.forward`; UjiCache ignores unused kwargs and consumes only the values it needs, especially `transformer_options`.

## Compatibility

- Target: StabilityMatrix Forge Neo / SD WebUI Forge Neo
- Primary workflow: txt2img with Anima / Cosmos-Predict2 T2I models
- Verified development environment: Windows, NVIDIA GPU, PyTorch CUDA build
- Not guaranteed: A1111 mainline, Forge classic, ComfyUI, multi-GPU, heavily modified pipelines

## Documentation

- [UjiCache specification](docs/UjiCache-spec-v1.2.md)
- [UjiCache EMA prediction notes](docs/UjiCache%20EMA%20Prediction_spec.md)
- [Auto Uji mode specification](docs/Auto-Uji-mode_spec_v1.0.md)
- [Coefficient profile presets](docs/PRESET-COEFFICIENTS.md)

## License and Credit

This project is licensed under the [Apache License 2.0](LICENSE).

Credit to `Rootport` or `Rootport-AI` is mandatory. Redistributions and derivative works must preserve the attribution in [NOTICE](NOTICE), as required by Apache License 2.0 Section 4(d).

## Development Reminder

Forge Neo may cache Gradio UI component settings in `ui-config.json`. If a slider range or default appears stale after reinstalling the extension, clear or update Forge Neo's UI config and restart the WebUI.

