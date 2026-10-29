# Security Policy

DKG-TLAN is a research reference implementation. It has no network
listeners, no credential handling, and loads no serialized model files.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/sunilgentyala/DKG-TLAN/security/advisories/new)
or by email to sunil.gentyala@ieee.org. Do not open a public issue for
security reports. Expect an acknowledgement within 7 days.

## Scope notes

- Do not load untrusted checkpoints with `torch.load` without
  `weights_only=True`; this repository ships no checkpoints.
- The code is research-grade and is not hardened for production ingestion
  of untrusted telemetry.
