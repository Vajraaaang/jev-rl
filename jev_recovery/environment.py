"""Deterministic, partially observed incident sandbox. No production actions."""
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    LOGS = 0
    CONFIG = 1
    CONNECTIONS = 2
    DEPENDENCY = 3
    JEV = 4
    RESTART = 5
    ROLLBACK = 6
    WAIT = 7
    STOP = 8


CAUSES = ('credentials', 'connections', 'dependency')
TRAIN_FAULTS = ((), ('credentials',), ('connections',), ('dependency',))
TEST_FAULTS = (('credentials', 'connections'), ('credentials', 'dependency'),
               ('connections', 'dependency'), CAUSES)
LOGS = {
    'train': {
        'credentials': 'database request failed: authentication rejected for service account',
        'connections': 'database request failed: connection pool at capacity; borrowed sessions never released',
        'dependency': 'upstream unavailable; circuit remains open',
    },
    'heldout': {
        'credentials': 'query could not run: password rejected by database',
        'connections': 'query could not run: too many clients; leased sessions not returned',
        'dependency': 'remote endpoint unreachable; breaker refuses requests',
    },
}


@dataclass(frozen=True)
class Scenario:
    faults: tuple[str, ...]
    wording: str = 'train'
    dependency_recovers_at: int = 6
    max_ticks: int = 24


class RecoveryEnv:
    """A tick is a simulated second; stop checks five further consecutive requests."""
    def __init__(self, scenario: Scenario, advisor=None):
        if not set(scenario.faults) <= set(CAUSES) or scenario.wording not in LOGS:
            raise ValueError('Unknown fault or wording')
        self.scenario, self.advisor = scenario, advisor
        self.bad_credentials = 'credentials' in scenario.faults
        self.leak = 'connections' in scenario.faults
        self.pool_exhausted = self.leak
        self.circuit_open = 'dependency' in scenario.faults
        self.tick = self.downtime = self.actions = self.interventions = 0
        self.tool_cost = self.api_cost_usd = self.input_tokens = 0
        self.cache_hits = self.api_calls = 0
        self.done = self.success = False
        self.known_config = self.known_pool = self.known_dependency = -1
        self.log_class = 0
        self.diagnosis = (-1, -1, -1)
        self.observed_logs = ''
        self.restart_at = -100
        self.trace = []

    def healthy(self):
        dependency_ready = self.tick >= self.scenario.dependency_recovers_at
        return (not self.bad_credentials and not self.pool_exhausted
                and (not self.circuit_open)
                and ('dependency' not in self.scenario.faults or dependency_ready))

    def observation(self):
        # Policy sees only diagnostic results, health, elapsed time and own actions.
        return (int(self.healthy()), self.known_config, self.known_pool,
                self.known_dependency, self.log_class, *self.diagnosis,
                min(self.tick // 3, 7))

    def _advance(self, ticks):
        for _ in range(ticks):
            self.tick += 1
            if self.leak and self.tick - self.restart_at >= 3:
                self.pool_exhausted = True
            self.downtime += not self.healthy()

    def logs(self):
        excerpts = []
        if self.bad_credentials:
            excerpts.append(LOGS[self.scenario.wording]['credentials'])
        if self.pool_exhausted:
            excerpts.append(LOGS[self.scenario.wording]['connections'])
        if self.circuit_open:
            excerpts.append(LOGS[self.scenario.wording]['dependency'])
        return '\n'.join(excerpts) or 'request completed successfully'

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode already terminated')
        action = Action(action)
        if action == Action.JEV and self.advisor is None:
            raise ValueError('No diagnostic provider configured')
        previous_down = self.downtime
        self.actions += 1
        cost = 0.2
        if action == Action.LOGS:
            self.observed_logs = self.logs()
            # Deliberately generic database symptoms remain ambiguous without diagnostics.
            self.log_class = (int('database' in self.observed_logs) +
                              2 * int(any(x in self.observed_logs for x in ('upstream', 'remote'))))
            self.tool_cost += 0.2
        elif action == Action.CONFIG:
            self.known_config = int(self.bad_credentials or self.leak)
            self.tool_cost += 0.5
            cost = 0.5
        elif action == Action.CONNECTIONS:
            self.known_pool = int(self.pool_exhausted)
            self.tool_cost += 0.5
            cost = 0.5
        elif action == Action.DEPENDENCY:
            self.known_dependency = int(self.circuit_open)
            self.tool_cost += 0.5
            cost = 0.5
        elif action == Action.JEV:
            result = self.advisor.diagnose(self.observed_logs or self.logs())
            self.diagnosis = tuple(int(result.probabilities[c] >= 0.5) for c in CAUSES)
            self.tool_cost += 0.8
            self.api_cost_usd += result.estimated_cost_usd if not result.cache_hit else 0
            self.input_tokens += result.input_tokens if not result.cache_hit else 0
            self.cache_hits += int(result.cache_hit)
            self.api_calls += int(result.source == 'hosted-jev' and not result.cache_hit)
            cost = 0.8
        elif action == Action.ROLLBACK:
            unnecessary = not (self.bad_credentials or self.leak)
            self.bad_credentials = self.leak = False
            self.known_config = -1
            self.interventions += 1
            cost = 1.5 + 2 * unnecessary
        elif action == Action.RESTART:
            unnecessary = not self.pool_exhausted and not self.circuit_open
            self.pool_exhausted = False
            self.restart_at = self.tick
            if self.tick >= self.scenario.dependency_recovers_at:
                self.circuit_open = False
            self.known_pool = self.known_dependency = -1
            self.interventions += 1
            cost = 1 + 2 * unnecessary
        elif action == Action.STOP:
            probes = []
            for _ in range(5):
                if self.tick >= self.scenario.max_ticks:
                    break
                self._advance(1)
                probes.append(self.healthy())
            self.success = len(probes) == 5 and all(probes)
            self.done = True
        if action != Action.STOP:
            self._advance(1)
        if self.tick >= self.scenario.max_ticks and not self.done:
            self.done = True
        reward = -cost - (self.downtime - previous_down)
        if self.done:
            reward += 30 if self.success else -25
        self.trace.append({'action': action.name.lower(), 'tick': self.tick,
                           'healthy': self.healthy(), 'reward': reward,
                           'observation': self.observation()})
        return self.observation(), reward, self.done
