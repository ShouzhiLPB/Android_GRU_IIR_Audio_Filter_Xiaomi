<!--
  Describes the automation package role and execution order for the pipeline.
-->

# automation

Cross-cutting scripts and tests that drive the whole delivery (copy sources, fetch deps, environment gates, build, deploy, verification, packaging).

## Intended run order (maps to `docs/ARCHITECTURE_AND_ITERATIONS.md`)

1. `copy_legacy_sources.py` — I-1  
2. `fetch_third_party.py` — I-2  
3. `env_check_all.py` — I-3  
4. `build_android.py` — I-4 / I-5 (configure + build)  
5. `deploy_android.py` — I-10  
6. `run_offline_verify.py` — I-11  
7. `package_release.py` — I-14  

Each script, once implemented, must include an **English file header** and **1–2 line English** docstrings on public functions (project rule).

## Getting started

See the root **`README.md`**: `fetch_third_party.py` → `env_check_all.py` → `build_android.py`, then open `workspaces/android_app` in Android Studio for the APK.
