# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The Superfermion team takes security seriously. If you discover a security
vulnerability, please **do not** open a public issue.

**Report via email:** security@superfermion.io

We aim to acknowledge receipt within 48 hours and provide an initial assessment
within 5 business days.

### What to Include

- Description of the vulnerability
- Steps to reproduce (proof-of-concept code if possible)
- Affected versions
- Any potential mitigations you've identified

### Process

1. You submit a report to security@superfermion.io
2. We acknowledge receipt within 48 hours
3. We investigate and assess impact within 5 business days
4. We develop and test a fix
5. We release a patch and publish an advisory
6. We credit the reporter (unless anonymity is requested)

## Scope

Security-relevant areas of the Superfermion framework include:

| Component | Risk Profile |
|-----------|-------------|
| `superfermion/security/` | Credential storage, token management, mTLS |
| `superfermion/qpu/` | API token handling, QPU provider authentication |
| `superfermion/config/` | Configuration file parsing, environment variable handling |
| `superfermion/serialization/` | Circuit/model deserialization (pickle, JSON, QASM) |
| `superfermion/cli.py` | Command-line argument parsing |
| `crates/sf-bindings/` | Rust-Python FFI boundary |
| `_sf_core` native extension | Native code execution |

## Dependency Audit Policy

- All direct dependencies are pinned with minimum versions in `pyproject.toml`
- Rust dependencies (`Cargo.toml` / `Cargo.lock`) are audited on each release
- CI runs `cargo audit` on Rust dependencies
- CI runs `pip-audit` on Python dependencies (planned)
- Transitive dependency updates are reviewed before each release

## Responsible Disclosure

We follow the [CVD (Coordinated Vulnerability Disclosure)](https://www.cisa.gov/coordinated-vulnerability-disclosure-process)
process. We request that you:

- Give us reasonable time to investigate and fix the issue before public disclosure
- Do not access or modify user data without permission
- Do not exploit the vulnerability beyond what is necessary to demonstrate it

## Security Features

Superfermion includes the following security capabilities:

- **Credential encryption** — QPU API tokens encrypted at rest via `security/credential_store.py`
- **Input sanitization** — Configurable input validation via `security/input_sanitizer.py`
- **Audit logging** — Structured security event logging via `telemetry/audit.py`
- **mTLS support** — Certificate generation for service-to-service communication

## Acknowledgments

We thank all security researchers who have responsibly disclosed vulnerabilities
to the Superfermion project.
