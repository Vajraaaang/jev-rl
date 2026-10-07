# Incident recovery benchmark

Offline heuristic experiment; hosted Jev effectiveness is untested.

Simulator probes alone determine sustained recovery. The advisor never grades its own success.

| Policy | Recovery | Actions | Downtime ticks | Tool cost units |
|---|---:|---:|---:|---:|
| checklist | 77.5% | 13.71 | 12.37 | 4.16 |
| random | 13.9% | 6.09 | 7.11 | 1.28 |
| greedy | 45.0% | 14.98 | 19.32 | 0.12 |
| rl_without_jev | 22.2% | 2.25 | 2.15 | 0.42 |
| rl_with_jev | 62.1% | 8.33 | 9.99 | 2.21 |
| rl_with_corrupted_advisor | 63.2% | 8.33 | 10.02 | 2.19 |

## Paired uncertainty

crossed paired bootstrap of training seeds and incident families. Synthetic population only; few training seeds make seed uncertainty unstable.

- rl_with_jev minus rl_without_jev: recovery difference 0.399, 95% interval [0.308, 0.510].
- rl_with_jev minus checklist: recovery difference -0.154, 95% interval [-0.256, -0.067].
- rl_with_jev minus rl_with_corrupted_advisor: recovery difference -0.011, 95% interval [-0.040, 0.000].

## Limits

- Small synthetic sandbox; recovery results do not establish production safety or transfer.
- Only local controllers learn. Diagnostic sensor weights and rubric remain fixed.
- Independent cause scores are multilabel scores, not calibrated probabilities.
- Three seeds by default provide limited information about training variability.
- Impossible incidents remain in overall recovery denominators; subgroup metrics report them separately.
- Costs are simulated tool units and provider acquisition estimates, not total infrastructure or output-token billing.
- Fixed validation-only checkpoint selection; no hyperparameter search or test-set checkpoint selection.
- Exploratory synthetic benchmark: previous test results have been inspected, so this is not an external blinded lockbox.
- Advisor corruption is a probability-complement stress test, not a realistic model of all diagnostic failures.

Diagnostic accuracy, per-suite, per-seed, impossible-incident, cost and provenance details are in report.json.
