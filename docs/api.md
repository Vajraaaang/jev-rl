# Local HTTP API

Start `python3 -m jev_recovery serve --port 8080`; console at http://127.0.0.1:8080. Offline advisor only. JSON bodies are limited to 8 KiB; at most 64 independent in-memory sessions. Mutations are serialized; restart clears sessions but preserves experiment artifacts.

| Method | Route | Function |
|---|---|---|
| GET | /api/meta | Supported tools/policies and checkpoint availability |
| POST | /api/incidents | Create an independent fault-injected episode |
| GET | /api/incidents/{id} | Public observation, metrics and trajectory |
| POST | /api/incidents/{id}/step | Execute one supported tool |
| POST | /api/incidents/{id}/run | Run checklist or frozen learned controller |
| GET | /api/report | Fixed configured experiment report |

Creation accepts faults (list from credentials/connections/dependency), wording (train/validation/test) and integer seed. Duplicate causes, unknown fields and invalid roots fail. Responses contain ID, observation, metrics, trace, decisions, done, simulation=true and advisor_source=offline-heuristic. Creation labels never reach controllers.

Example create body: {"faults":["credentials","connections","dependency"],"wording":"test","seed":11}. Step body: {"action":"logs"}; then {"action":"jev"} can consult. Run body: {"policy":"checklist"}, {"policy":"rl_without_jev"} or {"policy":"rl_with_jev"}. Paths and shell commands are rejected. Checkpoints come only from fixed names beneath the configured output directory.

400 means invalid JSON/tool/precondition; 403 prohibited Host/Origin; 404 unknown route/session or missing report; 409 terminated session, unavailable checkpoint or session cap. Observation reads do not advance time. Recovery follows sustained verification, not just the current lamp.

Legacy reset/action/health/request/logs routes operate a separate v1 episode. This HTTP service exposes simulated causal availability; it is not a real failing database deployment.
