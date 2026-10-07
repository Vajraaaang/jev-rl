# Model card

**Intended use:** local research into partial observability, diagnostic assistance and recovery decisions. Unsuitable for autonomous production remediation without a separate execution/authorization design.

**Controller:** normalized semi-gradient linear Q-learning, fixed action weights, bounded versioned public features and reproducible exploration. Actual rewards update local weights. No expert silently substitutes decisions. Intermediate candidates use validation-only reward selection.

**Hosted sensor:** pinned jev-1.13.0, three independent typed text questions, fixed hosted weights. **Offline sensor:** deterministic regex rules, labeled offline-heuristic. **Corrupted sensor:** complemented scores, explicitly an ablation rather than a plausible model.

Training/validation/test data are synthetic seeded incidents with separate finite wording templates and reserved combinations. There is no production incident corpus. Manifests disclose dynamics and labels. Model age, total budget, seed, feature schema, selection scores and hashes accompany checkpoints.

Use the saved report for measured performance. Overall rates include impossible cases. Controllers can escalate too early, underuse advice or choose excessive repairs. Linear off-policy learning is not guaranteed to converge. Shortcuts in a reward definition can produce apparent recovery, which is why independent temporal tests matter.

The advisor-enabled arm differs through available actions and optimization as well as scores. Inspect corrupted-score controls before attributing gains. Cache availability is not inference accuracy or latency. Diagnostic scores require calibration and testing on new data before operational use.

Artifacts preserve fixed weights and replayable evidence; changing model, feature, reward or rubric requires a new run. Operators remain responsible for credentials, log hygiene and any future real-service execution boundary.
