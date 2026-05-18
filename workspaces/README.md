<!--
  Explains why two workspace subtrees exist and how they relate to legacy repos.
-->

# workspaces

Contains **editable full copies** of the GRU training repository and the Android runtime repository.

- `gru_model/` — copy of the PC-side GRU / train / export project; ONNX exports live under `exported_onnx_models/` inside it.
- `android_runtime/` — copy of the Oboe + ONNX Runtime Android C++ project; CMake must reference `third_party/` with **relative** paths only.

Never point build scripts at the original legacy folders outside `Android_GRU_IIR_Audio_Filter`; refresh copies with `automation/copy_legacy_sources.py` when intentionally rebasing.
