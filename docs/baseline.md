# Jev-assisted incident recovery in an RL sandbox

A dependency-free Python experiment: a local Q-learning controller investigates a fault-injected service, optionally consults fixed hosted Jev, and chooses repairs. The simulator and five consecutive request probes determine recovery. The default advisor is an **offline regex heuristic, not Jev**. Results from that mode establish pipeline behavior only.

Requires Python 3.10+. Run from this checkout:

```sh
python3 -m unittest discover -s tests -v
python3 -m jev_recovery.cli experiment
python3 -m jev_recovery.cli serve --port 8080
```

The experiment trains two tabular controllers across seeds 11, 22 and 33 (20,000 episodes each), freezes them, and compares them with an evidence-driven checklist. JSON checkpoints, an aggregate/per-seed report and episode action traces are written under `artifacts/baseline`. All paths supplied to the CLI are resolved to absolute paths internally. `--output`, `--episodes`, `--seeds` and `--repeats` control a run.

## Faults and actions

Incorrect database credentials require rollback to known-good configuration. Exhausted connections require rollback to stop the configuration-induced leak and restart to clear already exhausted sessions. Restart alone gives only two ticks of relief before relapse. A failed dependency recovers externally after 5–7 ticks; its open circuit then requires restart to reconnect. Combinations require appropriate diagnostic sequencing and multiple repairs.

Actions: inspect compact logs, inspect configuration, inspect connections, inspect dependency, consult Jev, restart, rollback, wait, and stop. Stop performs five further health probes. A stopped episode fails if any probe fails or the probes exceed the 24-tick budget. The deterministic simulator runs entirely in-process; the HTTP service is a localhost facade over the same simulator, not a real database deployment. There are no production connections or repair commands.

The policy receives health, elapsed-time bucket, observed diagnostic results, coarse database/upstream log symptoms and any requested Jev probabilities. It cannot see the hidden fault tuple. Jev receives only observed compact log excerpts. Its three independent probability questions support concurrent causes; sorting them ranks predefined causes. It cannot award reward or decide recovery.

Reward = +30 for sustained recovery or −25 on failed termination, minus unhealthy ticks, action costs, and penalties for unnecessary restart/rollback. Tool costs are explicit proxy units, separate from dollars. Every action advances simulated time. Health probes also contribute to downtime.

## Evaluation

Training uses clean episodes and individual faults with training log wording. Evaluation separately measures:

- unseen wording for trained fault sets;
- unseen fault combinations with known wording;
- both unseen combinations and wording.

Each controller and checklist receives identical scenarios, dependency recovery timing and a 24-tick budget. Policies remain frozen throughout evaluation. Reports include recovery rate, action count, downtime ticks, interventions, diagnostic tool cost units, cache hits, hosted calls, input tokens and estimated API dollars. Total provider acquisition statistics include training and evaluation separately from per-episode evaluation costs. Per-seed outcomes are recorded; these small deterministic fixtures do not establish statistical significance or real production transfer. The runbook is intentionally strong. RL may underperform it, especially on unseen combinations; that is a valid experimental result.

## Hosted Jev and cached training

The integration follows the official [API reference](https://docs.typesafe.ai/api) and [model/customization documentation](https://docs.typesafe.ai/models), checked October 6, 2026. Requests use `POST /v1/systemone`, text `state`, a map of typed `noul` questions and pinned `jev-1.13.0`. Hosted weights stay fixed; only the local controller is trained. API documentation currently lists $0.042 per million input tokens and free output tokens; reported dollars are estimates, not invoices.

To explicitly populate a hosted cache and run the experiment, export `TYPESAFE_API_KEY` in your shell and choose live mode:

```sh
python3 -m jev_recovery.cli experiment --advisor live --output "$PWD/artifacts/hosted"
```

Live mode calls the API on cache misses. Repeated excerpts replay cached responses during training. To rerun without network access:

```sh
python3 -m jev_recovery.cli experiment --advisor cached --output "$PWD/artifacts/replay"
```

The cache is keyed by full request, model, rubric version and provider namespace; records preserve requested/resolved model and token usage. Offline results cannot satisfy hosted-cache mode. Cache misses, malformed results, missing credentials and API errors fail explicitly; no silent offline fallback exists. Cache hits charge zero fresh API dollars. API keys are never written to the cache. Generated caches are ignored by Git. Keep logs free of secrets before sending them to the provider. No hosted call was made to create the included offline report.

## HTTP demonstration

After starting `serve`, use another terminal:

```sh
curl -s -X POST http://127.0.0.1:8080/reset -H 'Content-Type: application/json' -d '{"faults":["connections"]}'
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/logs
curl -s -X POST http://127.0.0.1:8080/action -H 'Content-Type: application/json' -d '{"action":"rollback"}'
curl -s -X POST http://127.0.0.1:8080/action -H 'Content-Type: application/json' -d '{"action":"restart"}'
curl -s -X POST http://127.0.0.1:8080/action -H 'Content-Type: application/json' -d '{"action":"stop"}'
```

`/request` executes one simulated request and advances time; `/health` is a passive health read. `/reset` creates an isolated episode. `/action` exposes the predefined sandbox actions; this HTTP demo has no configured advisor, so Jev requests are rejected. The experiment CLI supplies its configured provider.
