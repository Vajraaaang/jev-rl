# Validation record — October 6, 2026

This records observed local checks, not planned checks or a production claim.

| Check | Observed outcome |
|---|---|
| macOS source tests, Python 3.14 | 109 tests passed |
| Python compilation / JavaScript syntax | Passed |
| Full offline research benchmark | 4,320 paired trajectories, six policies, four suites, three training seeds |
| Artifact semantic/integrity audit | Passed for the complete run |
| Independent frozen-policy replay | All 4,320 trajectories reproduced without network acquisition |
| Package source identity | Every reported Python source SHA256 matches current files |
| Wheel build and isolated installation | Passed; web assets and CLI present; trained recovery verified outside checkout |
| Docker image | Built locally from official Python 3.12 slim |
| Container execution with network disabled | 109 tests passed; 144 smoke trajectories audited and replayed |
| Browser console | Actual manual log collection/advisor consultation/checklist recovery and saved learned-controller triple-fault recovery verified |
| Benchmark browser view | Displays the measured six-policy report |
| Narrow browser viewport | Layout checked, no document horizontal overflow; normal viewport restored |
| Browser console errors | None observed |
| Hosted Jev acquisition | Zero calls; effectiveness remains unmeasured |
| Hosted GitHub CI | Workflow defined; not executed remotely |

The final run is `artifacts/research`. Original evidence is preserved under `artifacts/baseline`. Two development comparisons were archived under `artifacts/history`; every archived file was SHA256-verified before replacing the expanded directories. The final benchmark must not be treated as an untouched external lockbox.

Both trained arms consumed 4,000 episodes per seed. Selected intermediate checkpoint ages are recorded separately. The checklist recovers all structurally feasible incidents. The ordinary offline-advisor arm has 62.1% overall recovery, versus 22.2% without advice and 63.2% under corrupted scores. Corruption preserving performance prevents attributing the arm difference to useful diagnosis.

Browser evidence: `artifacts/browser/recovery-dashboard.png` and `learned-controller-recovery.png`. The console currently runs at http://127.0.0.1:18081. The ordinary command uses port8080.
