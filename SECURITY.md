# Security policy

Recovery Lab is a local research sandbox. Its tools affect simulated state; production remediation, authenticated multi-user hosting and unrestricted execution are outside its supported scope. See [security boundaries](docs/security.md) for key handling, redaction, cache integrity and execution limits.

## Report a vulnerability

Use this repository's private vulnerability-reporting feature if it is available. Otherwise contact the repository owner privately through an established channel. Do not disclose a working exploit, API key, sensitive log excerpt or private incident data in a public issue.

Include the affected commit, Python version, advisor mode, impact and a minimal synthetic reproduction. No response-time or support-level commitment is implied.

## Credentials

Keep TypeSafe keys in the server process or an operator-managed secret store. Never commit credentials or paste them into bug reports. The local `--prompt-key` flow reads a key without echoing or saving it. If a real key is exposed, revoke it through the provider before discussing the incident publicly.
