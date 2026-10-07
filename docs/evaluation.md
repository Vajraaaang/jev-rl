# Evaluation protocol

The question is whether fixed log diagnostics improve a local recovery controller under a declared cost and deadline. Simulator health is ground truth; advice never grades itself.

Default training uses seeds 11/22/33, 4,000 episodes per arm and identical incident schedules. Hyperparameters are fixed. Predeclared quarter-budget checkpoints plus the final checkpoint are evaluated on distinct validation wording; best mean simulator reward wins, with earliest ties. Full training budgets remain equal. Reserved test combinations never enter training or validation.

Four paired suites isolate joint, wording, combination and operational stress transfer. Stress includes tighter deadlines, longer restarts, recurrence, late/permanent dependencies and noisy logs. Six policies include checklist, random, repair-first, two RL arms, and the advisor-enabled checkpoint under complemented scores. Corruption tests dependence on score contents separately from action availability or a different training trajectory.

Recovery requires 12 consecutive healthy end-to-end probes within the horizon. This exceeds every generated recurrence window; insufficient custom connection windows are rejected. Escalation remains unrecovered even when operationally sensible. Impossible incidents stay in overall denominators and have a structural-feasibility subgroup.

Report actions, downtime, interventions, unnecessary interventions, tool costs, safety blocks, diagnosis failures and provider acquisition costs. Probe requests inside stop are not separate policy actions but do contribute downtime. Early escalation may look cheap while failing; compare recovery alongside costs. Restart/rollback downtime is charged even when state is otherwise healthy.

An independent initial-log study applies injected labels after prediction, then measures multilabel Brier, exact match, precision/recall/F1 and top-cause membership. Independent cause scores need not sum to one and are not demonstrated calibrated probabilities.

Crossed paired bootstrap resamples controller seeds and incident families, keeping transformed scenarios and policy pairs together. Replays are not fresh observations. Intervals apply only to this finite synthetic generator; three seeds give limited optimization-variability information.

Audit checks inventory, manifests, reserved splits, pairing, equal budgets/schedules, validation selection, metrics, diagnostic scores and intervals. Replay loads frozen policies and fixed responses and reproduces all trajectories. Unsigned artifacts establish consistency, not authenticity.

This is exploratory synthetic evaluation: development outcomes were inspected. No untouched production or external lockbox is claimed. A confirmatory Jev study needs a preregistered new corpus/distribution, pinned acquisition/weights/rubric/controller budgets and unseen final evaluation. If corrupt scores preserve performance, an advisor-arm advantage alone cannot establish useful diagnosis. Production transfer and real billing require separate evidence.
