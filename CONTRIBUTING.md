# Contributing to Superfermion

Welcome! We're building the first differentiable, hardware-neutral quantum-classical framework.

## 1. Setup

### Prerequisites

- **Rust**: [rustup](https://rustup.rs/) (Stable 1.80+)
- **Python**: 3.12+ (use `uv` for easy environment management)
- **Node.js**: (For frontend development)

### Development Environment

```bash
# Clone the repository
git clone https://github.com/superfermion/superfermion.git
cd superfermion

# Create virtual environment (with uv)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install Python dev dependencies
uv pip install -e ".[dev]"

# Check Rust installation
cargo check
```

## 2. Development Workflow

1.  **Read the Specs**: Before touching any code, review the 3 core specification files.
2.  **Update the Plan**: If your change affects the architecture, update `superfermion_master_plan.md`.
3.  **Implement**: Follow the guidelines in `CONVENTIONS.md`.
4.  **Test**:
    - Rust: `cargo test`
    - Python: `pytest tests/`
5.  **Track**: Update `superfermion_implementation_tracker.md` before your final commit.

## 3. Pull Request Guidelines

-   **Tests**: All PRs must include passing unit tests.
-   **Documentation**: Update docstrings and the `docs/` directory.
-   **Formatting**:
    - Rust: `cargo fmt`
    - Python: `ruff format`
-   **Type Safety**: Ensure `mypy` passes for all changed Python code.

## 4. Community

-   Join our Discord (link in README).
-   Report bugs via GitHub Issues.
-   For experimental features, see the `experiments/` directory.
