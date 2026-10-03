# Security policy

ElectroTrace is research software intended primarily for local or controlled research environments.

## Deployment

The default server bind is 127.0.0.1. External binds require ELECTROTRACE_API_KEY; use a strong secret and place the application behind a TLS-capable reverse proxy for shared deployments.

Do not expose the development Flask server directly to the public internet.

## Model files

New model artifacts use the scikit-learn-compatible .skops format plus a JSON metadata sidecar. ElectroTrace checks the recorded scikit-learn major version and audits unknown serialized types before loading.

Legacy .pkl files are migration-only. They are rejected by default and require an explicit trusted-local allow_pickle=True opt-in. Do not accept pickle files from users, CI uploads, or third-party downloads.

## Uploaded recordings

Uploaded recordings are stored outside the repository and are subject to the configured retention policy (ELECTROTRACE_UPLOAD_TTL_S). Large recordings should use persistent registration plus bounded window endpoints rather than JSON signal payloads.

## Reporting vulnerabilities

Please report security issues privately to the repository maintainers rather than opening a public issue containing exploit details.
