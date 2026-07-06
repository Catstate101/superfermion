# Superfermion Production Docker Image
# Multi-stage build: stage 1 builds Rust extension, stage 2 is the slim runtime

# ---- Stage 1: Build ----
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin for building the Rust extension
RUN pip install maturin>=1.0

WORKDIR /app

# Copy only what's needed for the Rust build first (layer caching)
COPY crates/ crates/
COPY Cargo.toml Cargo.lock ./

# Build the Rust extension as a wheel
RUN cd crates/sf-bindings && maturin build --release --manylinux off && cd ../..

# Copy Python package source
COPY superfermion/ superfermion/
COPY pyproject.toml README.md ./

# Install the Python package (editable install for development, or regular for production)
RUN pip install .

# Copy the built .so/.pyd into the package (maturin build puts it in target/wheels)
RUN cd crates/sf-bindings && maturin develop --release && cd ../..

# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/superfermion /app/superfermion

# Copy the compiled Rust extension
COPY --from=builder /app/superfermion/_sf_core*.so /app/superfermion/ 2>/dev/null || true

# Set environment
ENV PYTHONUNBUFFERED=1
ENV SF_ENV=production

# Smoke test
RUN python -c "import superfermion as sf; print(f'Superfermion {sf.__version__} ready')"

# Default: FastAPI server
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "superfermion.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
