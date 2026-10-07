"""Public-evidence agent runtime and bounded troubleshooting baselines.

Decision summaries are inspectable facts and selected actions, not chain of thought.
The runtime vetoes invalid requests; it never silently turns a learned policy into
an expert runbook. A veto escalates and is reported separately from executed safety
violations. No controller is passed the environment or privileged incident labels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import random
from typing import Any, Protocol

ACTIONS = ('logs', 'config', 'connections', 'dependency', 'jev',
           'restart', 'rollback', 'wait', 'stop', 'escalate')
PUBLIC_KEYS = ('schema_version', 'tick', 'remaining_ticks', 'verification_ticks',
               'health', 'healthy_streak', 'evidence', 'last_action',
               'last_outcome', 'action_counts')
EVIDENCE_KEYS = ('logs', 'config', 'connections', 'dependency', 'diagnosis')


class Policy(Protocol):
    def select(self, observation: dict) -> str: ...


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    evidence: dict
    hypotheses: list[dict]


def public_observation(observation: dict) -> dict:
    """Allowlist root fields, then copy nested evidence before policy invocation."""
    result = {key: deepcopy(observation[key]) for key in PUBLIC_KEYS if key in observation}
    evidence = observation.get('evidence', {})
    result['evidence'] = {key: deepcopy(evidence.get(key)) for key in EVIDENCE_KEYS}
    return result


def admissible_actions(observation: dict, use_jev: bool = True,
                       max_interventions: int = 6) -> tuple[str, ...]:
    """Shared training/evaluation safety masks, without fault-label information.

    Diagnostic retries require a state-changing intervention or elapsed time;
    consultation is attempted once, including when the diagnostic sensor fails.
    A final verification must fit entirely in the remaining simulator deadline.
    """
    evidence = observation.get('evidence', {})
    counts = observation.get('action_counts', {})
    remaining = observation.get('remaining_ticks', 0)
    allowed = []
    intervention_count = counts.get('restart', 0) + counts.get('rollback', 0)
    if remaining <= 0:
        return ('escalate',)
    for action in ACTIONS:
        if action == 'logs' and counts.get(action, 0) > 0:
            continue
        if action == 'config' and (evidence.get('config') is not None or counts.get(action, 0) > counts.get('rollback', 0)):
            continue
        if action == 'connections' and (evidence.get('connections') is not None or counts.get(action, 0) > counts.get('restart', 0) + counts.get('rollback', 0)):
            continue
        if action == 'dependency':
            known = evidence.get('dependency')
            if known is not None and observation.get('tick', 0) - known.get('observed_at', 0) < 2:
                continue
        if action == 'jev' and (not use_jev or not evidence.get('logs') or counts.get('jev', 0)):
            continue
        if action in ('restart', 'rollback') and (intervention_count >= max_interventions or remaining < 2):
            continue
        if action == 'stop' and (not observation.get('health') or remaining < observation.get('verification_ticks', 6)):
            continue
        allowed.append(action)
    return tuple(allowed)


def hypothesis_summary(evidence: dict) -> list[dict]:
    facts: dict[str, tuple[float, str]] = {}
    config, pool, dependency = (evidence.get(k) for k in ('config', 'connections', 'dependency'))
    if config is not None:
        facts['credentials'] = (float(not config.get('valid', True)), 'configuration inspection')
        if not config.get('connection_release_enabled', True):
            facts['connections'] = (1.0, 'connection release configuration')
    if pool is not None and pool.get('exhausted'):
        facts['connections'] = (1.0, 'connection pool inspection')
    if dependency is not None:
        facts['dependency'] = (float(not dependency.get('reachable', True) or dependency.get('circuit_open', False)), 'dependency probe')
    diagnosis = evidence.get('diagnosis')
    if diagnosis and diagnosis.get('available'):
        for cause, score in diagnosis.get('probabilities', {}).items():
            if cause in ('credentials', 'connections', 'dependency') and cause not in facts:
                facts[cause] = (float(score), 'diagnostic advisor estimate')
    return [dict(cause=cause, score=score, basis=basis)
            for cause, (score, basis) in sorted(facts.items(), key=lambda item: (-item[1][0], item[0]))]


class RecoveryAgent:
    def __init__(self, policy: Policy, max_interventions: int = 6):
        if isinstance(max_interventions, bool) or not isinstance(max_interventions, int) or max_interventions < 0:
            raise ValueError('max_interventions must be a nonnegative integer')
        self.policy = policy
        self.max_interventions = max_interventions
        self.evidence_ledger: list[dict] = []
        self.decision_trace: list[dict] = []
        self.safety_blocks: list[dict] = []

    def decide(self, observation: dict) -> Decision:
        public = public_observation(observation)
        allowed = admissible_actions(public, getattr(self.policy, 'use_jev', True), self.max_interventions)
        try:
            requested = self.policy.select(deepcopy(public))
            error = None
        except Exception as exc:
            requested, error = None, type(exc).__name__
        if not isinstance(requested, str) or requested not in allowed:
            action = 'escalate'
            reason = f'Policy unavailable ({error}); escalate.' if error else 'Requested action blocked by tool or safety constraints; escalate.'
            self.safety_blocks.append(dict(tick=public.get('tick'), requested_action=requested if isinstance(requested, str) else None,
                                           reason=reason, allowed_actions=list(allowed)))
        else:
            action = requested
            summaries = {'logs': 'Collect compact service log evidence.', 'config': 'Inspect configuration validity and connection release settings.',
                         'connections': 'Measure current connection pool pressure.', 'dependency': 'Probe upstream reachability and circuit status.',
                         'jev': 'Request ranked cause estimates from collected logs.', 'restart': 'Restart within the intervention budget.',
                         'rollback': 'Restore known-good configuration within the intervention budget.', 'wait': 'Allow time for service or upstream state to evolve.',
                         'stop': 'Request independent sustained recovery verification.', 'escalate': 'Escalate without claiming recovery.'}
            reason = summaries[action]
        facts = deepcopy(public['evidence'])
        if not self.evidence_ledger or self.evidence_ledger[-1]['evidence'] != facts:
            self.evidence_ledger.append(dict(tick=public.get('tick'), evidence=facts))
        decision = Decision(action, reason, facts, hypothesis_summary(facts))
        self.decision_trace.append(dict(tick=public.get('tick'), **asdict(decision)))
        return decision

    def run(self, env: Any) -> dict:
        # The environment stays in the executor. select()/decide() receive only its
        # explicit public observation; privileged labels never cross this boundary.
        self.evidence_ledger.clear()
        self.decision_trace.clear()
        self.safety_blocks.clear()
        while not env.done:
            decision = self.decide(env.observation())
            env.step(decision.action)
        result = deepcopy(env.metrics())
        result.update(decision_trace=deepcopy(self.decision_trace), evidence_ledger=deepcopy(self.evidence_ledger),
                      safety_blocks=deepcopy(self.safety_blocks),
                      diagnosis_failures=sum(event['action'] == 'jev' and
                                            (event.get('observation_after', {}).get('evidence', {}).get('diagnosis') or {}).get('available') is False
                                            for event in getattr(env, 'trace', [])))
        return result


class ChecklistPolicy:
    """Strong compositional runbook, with fresh probes after state changes."""
    use_jev = False

    def select(self, observation: dict) -> str:
        evidence = observation.get('evidence', {})
        allowed = admissible_actions(observation, False)
        if observation.get('remaining_ticks', 0) <= observation.get('verification_ticks', 6):
            return 'stop' if 'stop' in allowed else 'escalate'
        config = evidence.get('config')
        if config is None:
            return 'config' if 'config' in allowed else 'escalate'
        if not config.get('valid', True) or not config.get('connection_release_enabled', True):
            return 'rollback' if 'rollback' in allowed else 'escalate'
        pool = evidence.get('connections')
        if pool is None:
            return 'connections' if 'connections' in allowed else 'escalate'
        dependency = evidence.get('dependency')
        if dependency is None:
            return 'dependency' if 'dependency' in allowed else 'wait'
        if not dependency.get('reachable', True):
            return 'dependency' if 'dependency' in allowed else 'wait'
        if pool.get('exhausted', False) or dependency.get('circuit_open', False):
            return 'restart' if 'restart' in allowed else 'escalate'
        return 'stop' if 'stop' in allowed else 'wait'


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, observation: dict) -> str:
        return self.rng.choice(admissible_actions(observation))


class GreedyRepairPolicy:
    """Repair-first baseline that intentionally skips diagnosis."""
    use_jev = False

    def select(self, observation: dict) -> str:
        allowed = admissible_actions(observation, False)
        if 'stop' in allowed:
            return 'stop'
        counts = observation.get('action_counts', {})
        preferred = 'restart' if counts.get('restart', 0) <= counts.get('rollback', 0) else 'rollback'
        return preferred if preferred in allowed else 'wait' if 'wait' in allowed else 'escalate'
