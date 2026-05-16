# Security Policy

## Supported Versions

embspec is at v0.1.x. Security fixes will be issued for the current minor (0.1.x). Older minors will not receive backports.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Report privately by emailing `mukunda.vjcs6@gmail.com` with the subject `[embspec security]`. Include:

- A description of the vulnerability and its impact.
- The version of embspec affected (`pip show embspec`).
- Reproduction steps or a minimal proof-of-concept.
- Any suggested mitigation, if you have one.

You can expect:

- An acknowledgment within 5 business days.
- A status update within 14 days.
- A coordinated disclosure window of at most 90 days from the acknowledgment, after which the issue may be publicly disclosed.

## Specific Risk Surfaces in embspec

embspec is a thin pure-Python library so the risk surface is small, but a few areas are worth special attention if you find an issue:

- **`IndexManifest.load` / `from_dict`** — manifest files are JSON loaded from disk or S3. A malformed manifest should not be able to cause arbitrary code execution; if you find one that does, please report it.
- **`DriftAdapter.load`** — loads `.npz` files via `numpy.load`. We deliberately do not use `allow_pickle=True` since pickle deserialization is a known RCE vector. If you find a path where a malicious `.npz` can break out of numeric arrays, that's a security issue.
- **`safe_log_response`** in companion library `bedrock-ops` (not in embspec itself, but commonly used together) — bugs in the redaction path are security-relevant since they could surface PII to logs.

We will not pay bug bounties at this time.
