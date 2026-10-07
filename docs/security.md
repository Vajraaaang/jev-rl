# Security and execution boundaries

Tools affect only in-memory simulated service state. No shell, arbitrary network action, production connector or user-selected model path exists. Policies receive defensive public evidence without manifest IDs, fault labels, seeds or privileged feasibility.

The console is localhost-only with Host/Origin checks, asset/policy whitelists, restrictive CSP, bounded JSON/session counts and serialized mutations. Other local processes can access it; this is not an authenticated multi-user server. Sessions are ephemeral.

Explicit live acquisition uses an environment API key. Redirects are refused to protect bearer credentials. Requests/responses/timeouts/retries/calls are bounded. A conservative dollar reservation admits requests and returned usage is counted. This is not a billing guarantee; failed/retried calls may have costs unavailable to the client.

Common text/JSON secrets, bearer values and URI credentials are redacted before transport/cache. This filter cannot recognize every sensitive format: restrict excerpts and inspect them independently before live use. API key contents are never cached. Cache keys include redacted request, model pin, rubric and provider namespace. Integrity checksums, atomic replacement, shared thread locks and Unix process locks protect cache writes. Windows has in-process locking only.

Unsigned hashes detect corruption but can be forged by an actor controlling every artifact. Source hashes identify files rather than authenticate authors or prove isolation. Sensor failures, invalid policies and inadequate verification cannot award recovery.

Real remediation requires a new authorization/target/credential/rollback/rate-limit/human-escalation design and independent service monitoring. Synthetic success does not establish these controls.
