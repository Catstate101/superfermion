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

## 3. Pull Request Guidelines

- **Tests**: All PRs must include passing unit tests.
- **Documentation**: Update docstrings and the `docs/` directory.
- **Formatting**:
  - Rust: `cargo fmt`
  - Python: `ruff format`

## 4. Community

- Report bugs via GitHub Issues.
