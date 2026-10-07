# Recovery Lab

**An observable incident-recovery agent, a causal RL sandbox, and a reproducible experiment for diagnostic assistance.**

Recovery Lab trains a local controller to investigate and repair a small simulated service. A fixed Jev model can rank causes from collected logs. Independent health probes decide sustained recovery. The agent records evidence, hypotheses, tool decisions, action consequences, reward components, costs and termination.

The project includes a functioning incident console, trained checkpoints, six comparison arms, four evaluation suites, independent replay, integrity audits and adversarial tests. The default advisor is an **offline heuristic, not hosted Jev**. This is a research platform; synthetic success does not establish production safety or hosted Jev effectiveness.

## Quick start

Python 3.10+; no runtime dependencies, API key, database or frontend build required.

```sh
python3 -m jev_recovery serve --port 8080
```

Open **http://127.0.0.1:8080**. Inject faults, collect evidence manually or run a saved controller, and inspect the full decision/consequence timeline. The included research checkpoints enable learned controllers immediately. The console is explicitly a simulator and never calls a hosted API.

```sh
python3 -m unittest discover -s "$PWD/tests" -v
python3 -m jev_recovery benchmark --config "$PWD/configs/smoke.json" --output /tmp/recovery-smoke --cache /tmp/recovery-smoke-cache.json
python3 -m jev_recovery audit --output /tmp/recovery-smoke
python3 -m jev_recovery replay --output /tmp/recovery-smoke
```

Use a **new, empty output directory** for every experiment. Evidence is never overwritten. The full default trains 4,000 episodes per controller across seeds 11, 22 and 33, then runs six policies across four suites of 60 incidents: **4,320 paired evaluation trajectories**.

```sh
python3 -m jev_recovery benchmark --config "$PWD/configs/research.json" --output "$PWD/artifacts/my-research-run"
python3 -m jev_recovery serve --port 8080 --output "$PWD/artifacts/my-research-run"
```

JSON configs accept `episodes`, `count`, `seeds` and `advisor_mode`; file settings override CLI settings. The CLI resolves paths internally; Python APIs require absolute artifact paths. `jev-recovery` after installation and `python3 -m jev_recovery.cli` are equivalent entry points.

## The recovery agent

1. Receives a defensive public-observation copy, never the simulator or fault labels.
2. Collects configuration, pool, dependency and compact log evidence.
3. Consults the fixed diagnostic sensor only after collecting logs.
4. Selects a tool through a checklist or genuine feature-based Q-learning policy.
5. Applies freshness constraints, verification-time checks and an intervention budget.
6. Requests independent sustained verification or explicitly escalates unrecovered.

Decision summaries and hypotheses are inspectable facts, not hidden chain-of-thought text. Learned actions are never secretly replaced with checklist choices. Invalid policy outputs are blocked and recorded as escalation.

## Faults and causality

| Fault | Durable recovery | Misleading intervention |
|---|---|---|
| Incorrect credentials | Roll back known-good configuration | Restart leaves authentication broken |
| Exhausted connections | Roll back the leak, then restart exhausted sessions | Restart alone gives temporary relief and relapse |
| Failed dependency | Wait for reachability, then restart the circuit | Early restart cannot repair an external failure |

Seeded incidents vary fast/slow/permanent dependency recovery, restart duration, recurrence, log variants, archived red herrings and simultaneous faults. Train/validation/test wording is separate. One pair and the triple combination are reserved for testing. Some deadlines or permanent failures make recovery impossible; subgroup metrics identify those cases.

Actions consume simulated time; restart/rollback cause downtime. **Twelve consecutive independent healthy probes** must finish within the deadline, exceeding every generated recurrence window. Predictions cannot award recovery. Reward separates health outcomes, downtime, tool charges, intervention charges and unnecessary actions. Tool units and API acquisition dollars are distinct.

## Research and artifacts

Compare checklist, seeded random, repair-first, RL without advice, RL with advice, and the same advisor-enabled checkpoint under corrupted scores. Both trained arms receive identical incident schedules and total episode budgets. Predeclared intermediate candidates are selected on separate validation incidents; test data never selects checkpoints. Evaluation freezes weights.

The four suites isolate wording, combinations, joint transfer and operational stress. Reports include recovery/actions/downtime/interventions, unavailable diagnostics, safety blocks, tool/API costs and breakdowns by seed, recoverability, cause combination, noise and dependency regime. Paired bootstrap intervals cluster related scenario variants and training seeds. A separate labeled diagnostic study computes Brier score, exact match, precision/recall/F1 and top-cause accuracy after prediction.

`artifacts/research/report.md` is the human-readable result; `report.json` contains full metrics and uncertainty. `manifest.json` fixes incidents/configuration, `training.json` records learning and selection, `checkpoints/` freezes weights, `episodes.jsonl` preserves decisions/consequences/sensor responses, and `diagnostics.json` preserves the independent classification study. `integrity.json` inventories SHA256 hashes. Replay reproduces every trajectory without network access. Audit checks splits, pairing, budgets/schedules, selection, metrics and hashes. Unsigned checksums detect corruption, not malicious forgery.

The original tabular experiment remains under `artifacts/baseline`, with `python3 -m jev_recovery experiment` and [baseline instructions](docs/baseline.md). V1/v2 datasets and verification windows differ; their percentages are not a controlled before/after comparison.

## Hosted Jev and caching

The adapter follows the official [API reference](https://docs.typesafe.ai/api) and [models documentation](https://docs.typesafe.ai/models), checked October 6, 2026. It sends text `state` and independent typed `noul` questions to `/v1/systemone`, pins `jev-1.13.0` and validates the resolved version. Only local controller weights train.

Set `TYPESAFE_API_KEY` in your shell and explicitly choose live acquisition:

```sh
python3 -m jev_recovery benchmark --advisor live --cache "$PWD/artifacts/hosted-cache.json" --output "$PWD/artifacts/hosted-run"
python3 -m jev_recovery benchmark --advisor cached --cache "$PWD/artifacts/hosted-cache.json" --output "$PWD/artifacts/cached-run"
```

Live acquires only cache misses. Cached mode never uses the network; misses become observed diagnostic unavailability. Completed-run replay uses recorded responses rather than a mutable external cache. Offline and hosted namespaces cannot impersonate each other. The dashboard stays offline even when displaying a hosted report. No hosted call produced the included results.

The adapter bounds requests/responses/timeouts/retries/calls, applies conservative dollar admission reservations, redacts common JSON/text/URI secrets, refuses redirects, validates cache checksums, serializes shared access and writes atomically. Unix uses process locks; Windows guarantees in-process locking. Exact returned usage is reported; the spending guard is not an invoice guarantee, and failed/retried requests may have unknown billing. Redaction is best effort: restrict and inspect excerpts before live acquisition.

## Engineering and limits

Runtime is dependency-free. CI defines Python 3.10–3.13 installation/tests/smoke/audit/replay checks. Static assets are packaged and use no CDN. The localhost console validates Host/Origin, bounds bodies/sessions and serializes mutations. Docker runs offline research, not production remediation.

Linear Q-learning can be unstable; three seeds provide limited uncertainty; finite log templates do not cover arbitrary production data. A benefit over the no-advisor arm alone does not prove useful diagnostic scores caused it. Inspect corrupted-score and checklist controls. The included evaluation is exploratory synthetic research, not an untouched external lockbox.

See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md), [model card](docs/model-card.md), [HTTP API](docs/api.md), [security](docs/security.md) and [contributing](CONTRIBUTING.md).
