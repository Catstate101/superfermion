# Contributing to Superfermion

Welcome! We're building a high-performance quantum computing framework with
a Python API and a Rust simulation core.

## 1. Setup

### Prerequisites

- **Rust**: [rustup](https://rustup.rs/) (Stable 1.85+)
- **Python**: 3.10+ (the wheels are `abi3-py310`; use `uv` or `venv`)

### Development Environment

Run maturin from the repository root: the root `pyproject.toml` is what names
the extension `superfermion._sf_core` and points it at `crates/sf-bindings`.

```bash
git clone https://github.com/Catstate101/superfermion.git
cd superfermion

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -e ".[dev]"
pip install maturin

# Build the Rust extension -- CI builds and tests exactly this wheel
maturin build --release --out dist
pip install --force-reinstall dist/*.whl
# or, straight into the active virtualenv:
# maturin develop --release

# Verify: print the extension that is actually loaded, then run a circuit
python -c "import superfermion._sf_core as m; print(m.__file__)"
python -c "import superfermion as sf; print(sf.run(sf.Circuit(2).h(0).cx(0,1), device='cpu', shots=100).counts)"
```

Do not copy the built artefact into `superfermion/` by hand. A stale library
already in that directory wins the import silently: on Windows the suffixes are
tried as `['.cp313-win_amd64.pyd', '.pyd']`, so a leftover
`_sf_core.cp313-win_amd64.pyd` shadows the fresh `_sf_core.pyd` next to it, and
the suite passes against a build nobody edited.

## 2. Development Workflow

1. **Implement**: Follow the conventions in [`AGENTS.md`](AGENTS.md).
2. **Test**:
   - Rust: `cargo test --workspace`
   - Python: `python -m pytest tests/ -q --timeout=120`
3. **Lint**:
   - Rust: `cargo fmt --check --all` and `cargo clippy --workspace --all-targets -- -D warnings`
   - Python: `ruff check superfermion/ scripts/ --select=F --ignore=F401,F821`

### Running the suite against the engine you think you are testing

The tests call into a compiled extension, so "the suite is green" only means
"this engine is correct". A wheel left in user-site from an earlier
`pip install superfermion` will shadow an editable checkout and produce results
for a build you are not editing. Three installs are supported, and each one is
run the same way — never through `PYTHONPATH`, and never from a directory that
contains a second `superfermion/` folder:

| Install | Command | Notes |
|---|---|---|
| Editable | `pip install -e ".[dev]"` | Engine resolved from the checkout; confirm the path it prints |
| Wheel | `pip install dist/*.whl` in a clean venv | The path a user takes |
| Sdist | `pip install dist/*.tar.gz` | Builds from the tarball, needs Rust |

Before any of them: **uninstall a stale wheel first** (`pip uninstall
superfermion`, and check `python -c "import superfermion; print(superfermion.__file__)"`).
Every run prints the engine path and its sha256 in the pytest header, and
`scripts/grad_gate.py` asserts it:

```bash
python scripts/grad_gate.py --expect-version 0.1.25 \
    --expect-engine-root "$PWD/superfermion"
```

The gate fails if the imported engine is not the one under the root you named.
CI runs it on the editable installs, on the abi3 wheel, on an sdist-built
install and on free-threaded 3.14t.

## 3. Pull Request Guidelines

Open one at <https://github.com/Catstate101/superfermion/compare>; the same
page is on the docs site at <https://superfermion.com/docs/contributing>.

- **Tests**: All PRs must include passing unit tests, and a regression test must
  fail on the unfixed code — assert the exact value you observed, not a
  tolerance the broken build also passes.
- **Changelog**: Add an entry under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md),
  naming the symptom a user would have seen.
- **Documentation**: Update docstrings, `docs/`, and the page on the site that
  would otherwise be wrong.
- **Scope**: One topic per PR.
- **Formatting**:
  - Rust: `cargo fmt`
  - Python: `ruff format`

## 4. Community

- Report bugs via [GitHub Issues](https://github.com/Catstate101/superfermion/issues).
- Browse the [open pull requests](https://github.com/Catstate101/superfermion/pulls).
