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
|Python installl | `pip install --force-reinstall target/wheels/linreg_core-0.7.0-cp312-cp312-win_amd64.whl` |
| Python test | `pytest tests/python/ -v` |
| PyPI publish | `twine upload wheels/*.whl` |

---

## Notes

- **WASM artifacts** are auto-copied to `WASM_Example/js/` by the build process
- **CI bundles all wheels** into `all-wheels` artifact (15 wheels: 3 OS × 5 Python versions)
- **PyPI automatically serves the correct wheel** based on user's platform and Python version
- **Version bumping**: Update `Cargo.toml` first, then run `wasm-pack build` to sync npm version
