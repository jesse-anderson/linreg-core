# Publishing Instructions Reminder

Complete guide for publishing linreg-core to crates.io, npm, and PyPI.

---

## Prerequisites

- Authenticated accounts for:
  - [crates.io](https://crates.io) (for `cargo publish`)
  - [npm](https://www.npmjs.com) (for `npm publish`)
  - [PyPI](https://pypi.org) (for `twine upload`)

---

## 0. Push to GitHub and Verify CI

Push your changes and wait for CI to complete across all platforms:

```bash
git push
```

Then go to **Actions** tab and verify:
- All Rust tests pass (stable/beta, ubuntu/windows/macos)
- WASM build succeeds
- Python builds & tests pass (all OS × Python version combinations)
- Coverage threshold met (>=85%)

**Do not proceed until all CI checks pass.** This tests across many different systems and configurations.

Once CI passes, download the `all-wheels` artifact from the Artifacts section.

---

## 1. Rust / crates.io

### Build and Test
```bash
cargo build --release
cargo test
```

### WASM Tests (Critical - Prevents Drift)
```bash
wasm-pack test --node
```

### Dry Run
```bash
cargo publish --dry-run
```

### Publish
```bash
cargo publish
```

---

## 2. npm (WASM)

### Build WASM
```bash
wasm-pack build --release --target web
```

### Navigate to pkg directory
```bash
cd pkg
```

### Dry Run (optional)
```bash
npm publish --dry-run
```

### Publish
```bash
npm publish --access public
```

---

## 3. PyPI (Python Wheels)

### Install Built Wheel and Test (Critical)
```bash
test_venv\Scripts\activate.bat
pip install target/wheels/linreg_core*.whl
pytest tests/python/ -v
```

### Option A: Download from GitHub CI (Recommended)

1. Go to **Actions** tab in GitHub
2. Click on a recent **CI** workflow run
3. Scroll to **Artifacts** section
4. Download `all-wheels`
5. Unzip to a folder (e.g., `wheels/`)

### Option B: Build Locally
```bash
test_venv\Scripts\activate.bat
pip install maturin
maturin build --release --strip
```

### Publish All Wheels
```bash
test_venv\Scripts\activate.bat
twine upload wheels/*.whl
```

Or with full path:
```bash
test_venv\Scripts\activate.bat
twine upload C:\Users\Jesse\Documents\GitHub\linreg-core\target\wheels*.whl
```

---

## Version Sync Checklist

Before publishing, verify versions match:

| File | Version Location |
|------|------------------|
| `Cargo.toml` | `version = "x.y.z"` |
| `pkg/package.json` | `"version": "x.y.z"` |
| `WASM_Example\js\package.json` | `"version": "x.y.z"` |

---

## Quick Reference

| Task | Command |
|------|---------|
| Rust test | `cargo test` |
| WASM test | `wasm-pack test --node` |
| Rust publish | `cargo publish` |
| WASM build | `wasm-pack build --release --target web` |
| npm publish | `cd pkg && npm publish --access public` |
| activate venv | `test_venv\Scripts\activate.bat` |
| Python build | `maturin build --release --features python` |
|Python installl | `pip install --force-reinstall target/wheels/linreg_core-0.8.0-cp312-cp312-win_amd64.whl` |
| Python test | `pytest tests/python/ -v` |
| PyPI publish | `twine upload target/wheels/*.whl` |
| DLL build (x64) | `cargo build --release --target x86_64-pc-windows-msvc --features ffi` |
| DLL build (x86) | `cargo build --release --target i686-pc-windows-msvc --features ffi` |
| DLL verify deps | `dumpbin /dependents VBA_Example/linreg_core_x64.dll` |
| DLL verify exports | `dumpbin /exports VBA_Example/linreg_core_x86.dll \| findstr LR_` |

---

## 4. Windows DLL (VBA/Excel)

Produces `linreg_core_x64.dll` and `linreg_core_x86.dll` for use from VBA via `LinregCore.bas`.

### Prerequisites

```bash
rustup target add x86_64-pc-windows-msvc   # 64-bit (usually already present)
rustup target add i686-pc-windows-msvc      # 32-bit (required for 32-bit Excel)
```

### Build both DLLs

```bash
cargo build --release --target x86_64-pc-windows-msvc --features ffi
cargo build --release --target i686-pc-windows-msvc   --features ffi
```

`--features ffi` enables the FFI module and `cdylib` output. `.cargo/config.toml` automatically
applies `-C target-feature=+crt-static` for both MSVC targets so the DLLs have no VCRUNTIME
or CRT dependencies.

### Copy outputs to VBA_Example/

```bash
cp target/x86_64-pc-windows-msvc/release/linreg_core.dll VBA_Example/linreg_core_x64.dll
cp target/i686-pc-windows-msvc/release/linreg_core.dll   VBA_Example/linreg_core_x86.dll
```

### Verify dependencies (run from a Visual Studio Developer Command Prompt)

```
dumpbin /dependents VBA_Example/linreg_core_x64.dll
dumpbin /dependents VBA_Example/linreg_core_x86.dll
```

**Expected — only these system DLLs, no redistributable required:**

| DLL | Source |
|-----|--------|
| `KERNEL32.dll` | Windows core |
| `ntdll.dll` | Windows core |
| `bcryptprimitives.dll` | Windows (Rust HashMap seeding) |
| `api-ms-win-core-synch-l1-2-0.dll` | Windows 8+ (Rust OnceLock/Mutex) |

If `VCRUNTIME140.dll` or any `api-ms-win-crt-*.dll` appear, the static CRT flag in
`.cargo/config.toml` is not being picked up — verify the file exists and the targets match.

### Verify 32-bit export names are undecorated

```
dumpbin /exports VBA_Example/linreg_core_x86.dll | findstr LR_
```

Names must be plain (`LR_OLS`, `LR_Free`, …), **not** stdcall-decorated (`_LR_OLS@16`).
The `.def` file at the repo root (`linreg_core.def`) + `build.rs` handle this automatically.

### Key files for the DLL build

| File | Purpose |
|------|---------|
| `src/ffi/` | All Rust FFI source (handle store, OLS, regularized, diagnostics, prediction intervals, cross-validation, utils) |
| `linreg_core.def` | Module-definition file — exports all `LR_*` symbols with plain names (suppresses i686 stdcall decoration) |
| `build.rs` | Passes `/DEF:linreg_core.def` to the linker only when target is `i686-pc-windows-msvc` and `--features ffi` is active |
| `.cargo/config.toml` | Sets `+crt-static` for both MSVC targets so no VC++ redistributable is required |
| `VBA_Example/LinregCore.bas` | VBA module end-users import — conditional 32/64-bit Declares + Range wrappers |

### FFI coverage

| Category | Functions |
|----------|-----------|
| OLS | `LR_OLS` + 8 scalar getters + 6 vector getters |
| WLS | `LR_WLS` (reuses all OLS getters) |
| Regularized | `LR_Ridge`, `LR_Lasso`, `LR_ElasticNet` + 4 specific getters |
| Regularization path | `LR_LambdaPath` → `Vector` handle |
| Diagnostic tests | `LR_BreuschPagan`, `LR_White`, `LR_JarqueBera`, `LR_ShapiroWilk`, `LR_AndersonDarling`, `LR_HarveyCollier`, `LR_Rainbow`, `LR_Reset`, `LR_DurbinWatson`, `LR_BreuschGodfrey` |
| Influence diagnostics | `LR_CooksDistance`, `LR_DFFITS`, `LR_VIF`, `LR_DFBETAS` → `Vector`/`Matrix` handles |
| Generic vector/matrix | `LR_GetVectorLength`, `LR_GetVector`, `LR_GetMatrixRows`, `LR_GetMatrixCols`, `LR_GetMatrix` |
| Prediction intervals | `LR_PredictionIntervals` + 4 getters |
| K-Fold CV | `LR_KFoldOLS`, `LR_KFoldRidge`, `LR_KFoldLasso`, `LR_KFoldElasticNet` + 6 CV getters |
| Utilities | `LR_Version`, `LR_Init`, `LR_Free`, `LR_GetLastError` |

**Not yet exposed via FFI (future work):** LOESS (returns fitted values + SE over a 1-D x vector; needs a different input shape than the current row-major matrix API).

### Version sync

`VBA_Example/` does not have its own package file. Just ensure the DLLs are rebuilt whenever
`Cargo.toml` version changes and the `src/ffi/` API changes.

### Adding the new symbols to an existing workbook

Re-import `LinregCore.bas` over the existing module, or paste the new Declare statements and
wrapper functions (Sections 8-12 of `LinregCore.bas`) into the existing module manually.

---

## Notes

- **WASM artifacts** are auto-copied to `WASM_Example/js/` by the build process
- **CI bundles all wheels** into `all-wheels` artifact (15 wheels: 3 OS × 5 Python versions)
- **PyPI automatically serves the correct wheel** based on user's platform and Python version
- **Version bumping**: Update `Cargo.toml` first, then run `wasm-pack build` to sync npm version
