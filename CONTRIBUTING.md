# Contributing to Superfermion

Welcome! We're building a high-performance quantum computing framework with
a Python API and a Rust simulation core.

## 1. Setup

### Prerequisites

- **Rust**: [rustup](https://rustup.rs/) (Stable 1.75+)
- **Python**: 3.10+ (use `uv` or `venv` for environment management)

### Development Environment

```bash
git clone https://github.com/superfermion/superfermion.git
cd superfermion

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -e ".[dev]"

# Build the Rust extension
pip install maturin
cd crates/sf-bindings && maturin develop --release && cd ../..

# Copy built extension into the package
cp target/release/lib_sf_core.so superfermion/_sf_core.so  # Linux
# cp target/release/lib_sf_core.dylib superfermion/_sf_core.so  # macOS

# Verify
python -c "import superfermion as sf; print(sf.run(sf.Circuit(2).h(0).cx(0,1), device='cpu', shots=100).counts)"
```

## 2. Development Workflow

1. **Implement**: Follow conventions in `docs/conventions.md`.
2. **Test**:
   - Rust: `cargo test --workspace`
   - Python: `pytest tests/`
3. **Lint**:
   - Rust: `cargo fmt && cargo clippy`
   - Python: `ruff check superfermion/`

### Running the suite against the engine you think you are testing

The tests call into a compiled extension, so "the suite is green" only means
"this engine is correct". A wheel left in user-site from an earlier
`pip install superfermion` will shadow an editable checkout and produce results
for a build you are not editing. Three installs are supported, and each one is
run the same way — never through `PYTHONPATH`, and never from a directory that
contains a second `superfermion/` folder:

| Install | Command | Notes |
|---|---|---|
| Editable | `pip install -e ".[dev]"` | Engine built into `superfermion/` in the checkout |
| Wheel | `pip install dist/*.whl` in a clean venv | The path a user takes |
| Sdist | `pip install dist/*.tar.gz` | Builds from the tarball, needs Rust |

Before any of them: **uninstall a stale wheel first** (`pip uninstall
superfermion`, and check `python -c "import superfermion; print(superfermion.__file__)"`).
Every run prints the engine path and its sha256 in the pytest header, and

```bash
python scripts/grad_gate.py --expect-version 0.1.25 \
    --expect-engine-root "$PWD/superfermion"
```

asserts it: the gate fails if the imported engine is not the one under the
root you named. CI runs it on the editable installs, on the abi3 wheel, on an
sdist-built install and on free-threaded 3.14t.

## 3. Pull Request Guidelines

- **Tests**: All PRs must include passing unit tests.
- **Documentation**: Update docstrings and the `docs/` directory.
- **Formatting**:
  - Rust: `cargo fmt`
  - Python: `ruff format`

## 4. Community

- Report bugs via GitHub Issues.
