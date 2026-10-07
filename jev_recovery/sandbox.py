"""Causal partially observed incident environment, with independent health probes.

No operating-system remediation is performed. All time and intervention effects
are simulated; diagnostic providers receive only explicitly collected log text.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
import random

from jev_recovery.jev import Diagnosis
from jev_recovery.scenarios import CAUSES, Incident

ACTIONS = ('logs', 'config', 'connections', 'dependency', 'jev', 'restart',
           'rollback', 'wait', 'stop', 'escalate')
SCHEMA_VERSION = 'recovery-lab-observation-v2'
REWARD_VERSION = 'sustained-recovery-v2'
TOOL_COSTS = dict(logs=.2, config=.5, connections=.5, dependency=.5, jev=.8,
                  restart=0., rollback=0., wait=0., stop=.2, escalate=0.)

# Each split owns complete text templates: no held-out synonym enters training.
LOG_TEMPLATES = {
    'train': {
        'credentials': ('database authentication rejected for service account',
                        'database login refused: supplied credential invalid',
                        'request failed: database access denied during authentication'),
        'connections': ('database pool exhausted; borrowed sessions never released',
                        'database connection limit reached; allocation timed out',
                        'database checkout timed out: all pool entries occupied'),
        'dependency': ('upstream unavailable; circuit remains open',
                       'dependency call refused: remote endpoint not responding',
                       'upstream request failed: service unavailable'),
        'healthy': ('request completed successfully', 'health request returned 200', 'service request handled normally'),
    },
    'validation': {
        'credentials': ('SQL login authentication failed: identity mismatch',
                        'storage rejected the configured secret at login',
                        'database permission denied to runtime identity'),
        'connections': ('no available connections: lease backlog increasing',
                        'remaining connection slots exhausted by orphaned borrowers',
                        'pool saturation detected; clients pending checkout'),
        'dependency': ('backend unavailable; breaker blocked the outbound call',
                       'name resolution failed for the downstream gateway',
                       'external host down; dependency probe not acknowledged'),
        'healthy': ('synthetic transaction accepted', 'readiness transaction passed', 'request processing within target'),
    },
    'test': {
        'credentials': ('query could not run: password rejected by database',
                        'session setup failed: authentication secret is out of date',
                        'principal unauthorized: database identity rejected'),
        'connections': ('query could not run: too many clients; leased sessions not returned',
                        'workers stalled: pool capacity unavailable for new borrowers',
                        'too many connections: checkout budget expired'),
        'dependency': ('remote endpoint unreachable; breaker refuses requests',
                       'gateway connection refused: dependency health probe failed',
                       'downstream DNS lookup errored; upstream transaction aborted'),
        'healthy': ('end-to-end probe successful', 'transaction reached completion', 'application readiness confirmed'),
    },
}
NOISE = {
    'train': ('DEBUG periodic metrics flush finished', 'WARN yesterday: authentication rejected; archived event',
              'INFO pool capacity planning job completed; unrelated background task'),
    'validation': ('TRACE asynchronous housekeeping cycle', 'INFO archived password rejection from previous deployment',
                   'WARN historical endpoint unreachable; old retry sample'),
    'test': ('DEBUG telemetry batching completed', 'INFO old pool saturation alert archived before this incident',
             'WARN archived gateway connection refused; unrelated prior request'),
}


class IncidentEnv:
    def __init__(self, incident: Incident, advisor=None):
        if not isinstance(incident, Incident):
            raise TypeError('incident must be an Incident')
        self.incident = incident
        self.advisor = advisor
        self.tick = 0
        self.done = False
        self.success = False
        self.termination = None
        self.bad_credentials = 'credentials' in incident.faults
        self.leak = 'connections' in incident.faults
        self.pool_exhausted = self.leak
        self.circuit_open = 'dependency' in incident.faults
        self.relapse_at = None
        self.healthy_streak = 0
        self.evidence = dict.fromkeys(('logs', 'config', 'connections', 'dependency', 'diagnosis'))
        self.last_action = None
        self.last_outcome = None
        self.action_counts = Counter()
        self.trace = []
        self._metrics = dict(recovered=False, actions=0, downtime_ticks=0, interventions=0,
                             unnecessary_interventions=0, tool_cost_units=0., api_cost_usd=0.,
                             input_tokens=0, api_calls=0, cache_hits=0, reward=0.,
                             termination=None, safety_violations=0)

    def _dependency_reachable(self) -> bool:
        return ('dependency' not in self.incident.faults or
                (self.incident.dependency_ready_at is not None and
                 self.tick >= self.incident.dependency_ready_at))

    def healthy(self) -> bool:
        """Privileged independent end-to-end health checker; never uses diagnosis."""
        return (not self.bad_credentials and not self.pool_exhausted and
                not self.circuit_open and self._dependency_reachable())

    def observation(self) -> dict:
        return deepcopy(dict(schema_version=SCHEMA_VERSION, tick=self.tick,
                             remaining_ticks=self.incident.max_ticks - self.tick,
                             verification_ticks=self.incident.verification_ticks,
                             health=self.healthy(), healthy_streak=self.healthy_streak,
                             evidence=self.evidence, last_action=self.last_action,
                             last_outcome=self.last_outcome, action_counts=dict(self.action_counts)))

    def metrics(self) -> dict:
        return deepcopy(self._metrics | {'action_counts': dict(self.action_counts)})

    def logs(self) -> str:
        """Current compact excerpt; policy must use the explicit ``logs`` action.

        Stable noise is incident-seeded and contains explicit archived-event
        red herrings. Text reflects symptoms, not hidden fault tuples.
        """
        templates = LOG_TEMPLATES[self.incident.wording]
        variant = self.incident.log_variant
        symptoms = []
        if self.bad_credentials:
            symptoms.append(templates['credentials'][variant])
        if self.pool_exhausted:
            symptoms.append(templates['connections'][variant])
        if self.circuit_open or not self._dependency_reachable():
            symptoms.append(templates['dependency'][variant])
        if not symptoms:
            symptoms.append(templates['healthy'][variant])
        lines = [f'ERROR current: {text}' if not self.healthy() else f'INFO current: {text}' for text in symptoms]
        if self.incident.noise:
            rng = random.Random(f'logs-v2:{self.incident.seed}:{variant}')
            lines.extend(rng.sample(NOISE[self.incident.wording], 2))
            rng.shuffle(lines)
        return '\n'.join(lines)

    def _advance(self, duration: int, force_down: bool = False) -> list[bool]:
        probes = []
        for _ in range(min(duration, self.incident.max_ticks - self.tick)):
            self.tick += 1
            if self.leak and self.relapse_at is not None and self.tick >= self.relapse_at:
                self.pool_exhausted = True
            healthy = self.healthy() and not force_down
            probes.append(healthy)
            self._metrics['downtime_ticks'] += int(not healthy)
            self.healthy_streak = self.healthy_streak + 1 if healthy else 0
        return probes

    def _diagnose(self) -> tuple[dict, dict | None]:
        logs = self.evidence['logs']
        if logs is None:
            self._metrics['safety_violations'] += 1
            return {'status': 'blocked', 'message': 'Collect logs before consulting the diagnostic provider'}, None
        try:
            if self.advisor is None:
                raise RuntimeError('provider unavailable')
            result = self.advisor.diagnose(logs)
            if not isinstance(result, Diagnosis):
                raise ValueError('invalid diagnostic result type')
            if set(result.probabilities) != set(CAUSES):
                raise ValueError('invalid cause set')
            for p in result.probabilities.values():
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('invalid probability')
            if (isinstance(result.input_tokens, bool) or not isinstance(result.input_tokens, int) or
                    result.input_tokens < 0 or isinstance(result.estimated_cost_usd, bool) or
                    not isinstance(result.estimated_cost_usd, (int, float)) or
                    not math.isfinite(result.estimated_cost_usd) or result.estimated_cost_usd < 0 or
                    not isinstance(result.cache_hit, bool) or not isinstance(result.source, str) or
                    not result.source or not isinstance(result.model, str) or not result.model):
                raise ValueError('invalid diagnostic provenance')
            provider = asdict(result) | {'logs_sha256': hashlib.sha256(logs.encode()).hexdigest()}
            self.evidence['diagnosis'] = dict(available=True, **asdict(result),
                                               ranked_causes=list(result.ranked_causes), observed_at=self.tick)
            # Corruption changes diagnostic content, never acquisition billing.
            hosted = result.source in ('hosted-jev', 'corrupted-hosted-jev')
            self._metrics['cache_hits'] += int(result.cache_hit)
            self._metrics['api_calls'] += int(hosted and not result.cache_hit)
            self._metrics['input_tokens'] += result.input_tokens if hosted and not result.cache_hit else 0
            self._metrics['api_cost_usd'] += result.estimated_cost_usd if hosted and not result.cache_hit else 0.
            return {'status': 'ok', 'message': 'Fixed diagnostic provider ranked predefined causes'}, provider
        except Exception as exc:
            # Provider failures are public evidence; raw exceptions may contain secrets.
            self.evidence['diagnosis'] = dict(available=False, probabilities={}, ranked_causes=[],
                                               source='unavailable', model=None, cache_hit=False,
                                               input_tokens=0, estimated_cost_usd=0., observed_at=self.tick)
            return {'status': 'unavailable', 'message': 'Diagnostic provider unavailable',
                    'error_type': type(exc).__name__}, {'source': 'unavailable', 'error_type': type(exc).__name__}

    def step(self, action: str) -> tuple[dict, float, bool]:
        if self.done:
            raise RuntimeError('Episode already terminated')
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError('Unknown action')
        before = self.observation()
        tick_before = self.tick
        downtime_before = self._metrics['downtime_ticks']
        self.action_counts[action] += 1
        self._metrics['actions'] += 1
        provider = None
        outcome = {'status': 'ok', 'message': ''}
        duration, forced_down, intervention_cost, unnecessary_cost = 1, False, 0., 0.
        if action == 'logs':
            self.evidence['logs'] = self.logs()
            outcome['message'] = 'Collected compact application log excerpt'
        elif action == 'config':
            self.evidence['config'] = dict(valid=not self.bad_credentials,
                                            connection_release_enabled=not self.leak,
                                            version='current' if self.bad_credentials or self.leak else 'known-good',
                                            observed_at=self.tick)
            outcome['message'] = 'Validated credentials and connection release configuration'
        elif action == 'connections':
            self.evidence['connections'] = dict(exhausted=self.pool_exhausted,
                                                 active=20 if self.pool_exhausted else 3,
                                                 capacity=20, observed_at=self.tick)
            outcome['message'] = 'Inspected connection pool utilization'
        elif action == 'dependency':
            self.evidence['dependency'] = dict(reachable=self._dependency_reachable(),
                                                circuit_open=self.circuit_open, observed_at=self.tick)
            outcome['message'] = 'Probed upstream reachability and circuit state'
        elif action == 'jev':
            outcome, provider = self._diagnose()
        elif action in ('restart', 'rollback'):
            self._metrics['interventions'] += 1
            if action == 'restart':
                necessary = self.pool_exhausted or (self.circuit_open and self._dependency_reachable())
                duration, forced_down, intervention_cost = self.incident.restart_duration, True, 1.
                self.pool_exhausted = False
                self.relapse_at = self.tick + duration + self.incident.relapse_after if self.leak else None
                if self._dependency_reachable():
                    self.circuit_open = False
                self.evidence['connections'] = self.evidence['dependency'] = None
                outcome['message'] = 'Restarted application; existing connections flushed and reachable circuit reset'
            else:
                necessary = self.bad_credentials or self.leak
                duration, forced_down, intervention_cost = 2, True, 1.5
                self.bad_credentials = self.leak = False
                self.relapse_at = None
                self.evidence['config'] = self.evidence['connections'] = None
                outcome['message'] = 'Restored known-good configuration; existing pool state retained'
            if not necessary:
                self._metrics['unnecessary_interventions'] += 1
                unnecessary_cost = 2.
        elif action == 'wait':
            outcome['message'] = 'Waited one simulated tick'
        elif action == 'stop':
            duration = self.incident.verification_ticks
            outcome['message'] = 'Ran independent sustained end-to-end verification'
        elif action == 'escalate':
            outcome['message'] = 'Escalated incident for external remediation'

        health_probes = self._advance(duration, forced_down)
        if action == 'stop':
            self.success = len(health_probes) == duration and all(health_probes)
            self.done = True
            self.termination = ('recovered' if self.success else
                                'verification_budget_exhausted' if len(health_probes) < duration else 'verification_failed')
            outcome['health_probes'] = health_probes
            outcome['status'] = 'verified' if self.success else 'failed'
        elif action == 'escalate':
            self.done, self.termination = True, 'escalated'
        elif self.tick >= self.incident.max_ticks:
            self.done, self.termination = True, 'horizon_exhausted'

        tool_cost = TOOL_COSTS[action]
        self._metrics['tool_cost_units'] += tool_cost
        components = dict(downtime=-float(self._metrics['downtime_ticks'] - downtime_before),
                          action=-.1, tool=-tool_cost, intervention=-intervention_cost,
                          unnecessary=-unnecessary_cost,
                          terminal=(40. if self.success else -15.) if self.done else 0.)
        reward = sum(components.values())
        self._metrics['reward'] += reward
        self._metrics['recovered'] = self.success
        self._metrics['termination'] = self.termination
        self.last_action, self.last_outcome = action, outcome
        after = self.observation()
        self.trace.append(dict(schema_version='recovery-lab-trace-v2', action=action,
                               tick_before=tick_before, tick=self.tick,
                               observation_before=before, observation_after=deepcopy(after),
                               outcome=deepcopy(outcome), reward=reward,
                               reward_components=components, reward_version=REWARD_VERSION,
                               provider=provider, done=self.done))
        return after, reward, self.done
