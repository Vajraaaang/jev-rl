"""Seeded semi-gradient linear Q-learning over public diagnostic features.

Both arms use the same action-independent incident schedule and episode budget.
Evaluation uses only frozen weights; there is no runbook fallback or online update.
New evidence combinations share the feature representation. For a completely
untrained tie, the fixed tool order breaks ties deterministically. This is a
small research controller, not a claim of stable convergence for linear Q-learning.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from .agent import ACTIONS, admissible_actions

CAUSES = ('credentials', 'connections', 'dependency')
FEATURE_SCHEMA = (
    'bias', 'health', 'healthy_streak', 'elapsed', 'remaining', 'near_deadline',
    'config_unknown', 'config_bad', 'release_disabled', 'pool_unknown', 'pool_bad',
    'pool_utilization', 'dependency_unknown', 'dependency_down', 'circuit_open',
    'diagnosis_unknown', 'diagnosis_unavailable', 'jev_credentials', 'jev_connections',
    'jev_dependency', 'log_auth', 'log_pool', 'log_dependency', 'log_present',
    'config_pool', 'config_dependency', 'pool_dependency', 'health_leak',
    'health_restart', 'restart_pool', 'rollback_pool', 'dependency_time',
    'diagnosis_health', 'after_rollback_config_unknown', 'after_restart_pool_unknown',
    'last_failure', 'interventions',
) + tuple('count_' + action for action in ACTIONS) + tuple('last_' + action for action in ACTIONS)
SCHEMA_VERSION = 2


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _absolute_path(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_absolute():
        raise ValueError('checkpoint path must be absolute')
    return result


class LinearQPolicy:
    def __init__(self, use_jev: bool = False, seed: int = 0):
        if not isinstance(use_jev, bool) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('use_jev must be boolean and seed must be integer')
        self.use_jev, self.seed = use_jev, seed
        self.weights = {action: {feature: 0.0 for feature in FEATURE_SCHEMA} for action in ACTIONS}
        self.training_episodes = 0

    def features(self, observation: dict) -> dict[str, float]:
        evidence = observation.get('evidence', {})
        config, pool, dependency = (evidence.get(key) for key in ('config', 'connections', 'dependency'))
        diagnosis = evidence.get('diagnosis') if self.use_jev else None
        counts = observation.get('action_counts', {})
        elapsed = observation.get('tick', 0)
        remaining = observation.get('remaining_ticks', 0)
        budget = max(1, elapsed + remaining)
        health = float(bool(observation.get('health')))
        config_bad = float(config is not None and not config.get('valid', True))
        leak = float(config is not None and not config.get('connection_release_enabled', True))
        pool_bad = float(pool is not None and pool.get('exhausted', False))
        dependency_down = float(dependency is not None and not dependency.get('reachable', True))
        restart = min(1.0, counts.get('restart', 0) / 3)
        rollback = min(1.0, counts.get('rollback', 0) / 3)
        logs = (evidence.get('logs') or '').lower()
        # Coarse lexical signals are fixed, auditable features, not a diagnostic
        # rule selecting repairs. Their action usefulness is learned from reward.
        features = dict(
            bias=1.0, health=health, healthy_streak=min(1.0, observation.get('healthy_streak', 0) / max(1, observation.get('verification_ticks', 6))),
            elapsed=elapsed / budget, remaining=remaining / budget,
            near_deadline=float(remaining <= observation.get('verification_ticks', 6) + 3),
            config_unknown=float(config is None), config_bad=config_bad, release_disabled=leak,
            pool_unknown=float(pool is None), pool_bad=pool_bad,
            pool_utilization=0.0 if pool is None else min(1.0, pool.get('active', 0) / max(1, pool.get('capacity', 1))),
            dependency_unknown=float(dependency is None), dependency_down=dependency_down,
            circuit_open=float(dependency is not None and dependency.get('circuit_open', False)),
            diagnosis_unknown=float(self.use_jev and diagnosis is None),
            diagnosis_unavailable=float(diagnosis is not None and not diagnosis.get('available', False)),
            log_auth=float(any(word in logs for word in ('auth', 'credential', 'password', 'denied'))),
            log_pool=float(any(word in logs for word in ('pool', 'connection', 'capacity', 'slot'))),
            log_dependency=float(any(word in logs for word in ('upstream', 'dependency', 'unreachable', 'gateway'))),
            log_present=float(bool(logs)), config_pool=config_bad * pool_bad,
            config_dependency=config_bad * dependency_down, pool_dependency=pool_bad * dependency_down,
            health_leak=health * leak, health_restart=health * restart,
            restart_pool=restart * pool_bad, rollback_pool=rollback * pool_bad,
            dependency_time=dependency_down * elapsed / budget,
            diagnosis_health=health * float(diagnosis is not None and diagnosis.get('available', False)),
            after_rollback_config_unknown=rollback * float(config is None),
            after_restart_pool_unknown=restart * float(pool is None),
            last_failure=float((observation.get('last_outcome') or {}).get('status') in ('error', 'failed', 'unavailable')),
            interventions=min(1.0, (counts.get('restart', 0) + counts.get('rollback', 0)) / 6),
        )
        probabilities = diagnosis.get('probabilities', {}) if diagnosis and diagnosis.get('available') else {}
        for cause in CAUSES:
            features['jev_' + cause] = float(probabilities.get(cause, 0))
        for action in ACTIONS:
            features['count_' + action] = min(1.0, counts.get(action, 0) / 3)
            features['last_' + action] = float(observation.get('last_action') == action)
        return features

    def allowed_actions(self, observation: dict) -> tuple[str, ...]:
        return admissible_actions(observation, self.use_jev)

    def values(self, observation: dict) -> dict[str, float]:
        features = self.features(observation)
        active = tuple((key, value) for key, value in features.items() if value)
        return {action: sum(self.weights[action][key] * value for key, value in active)
                for action in self.allowed_actions(observation)}

    def select(self, observation: dict) -> str:
        values = self.values(observation)
        return max(values, key=values.get)

    def checksum(self) -> str:
        return _digest(self.weights)

    def save(self, absolute_path: str | Path) -> None:
        path = _absolute_path(absolute_path)
        payload = dict(schema_version=SCHEMA_VERSION, algorithm='linear-q-learning',
                       feature_schema=list(FEATURE_SCHEMA), actions=list(ACTIONS),
                       use_jev=self.use_jev, seed=self.seed, training_episodes=self.training_episodes,
                       weights=self.weights)
        data = dict(payload, sha256=_digest(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, absolute_path: str | Path) -> 'LinearQPolicy':
        path = _absolute_path(absolute_path)
        if path.stat().st_size > 2_000_000:
            raise ValueError('checkpoint exceeds size limit')
        try:
            data = json.loads(path.read_text())
            checksum = data.pop('sha256')
            if not isinstance(checksum, str) or _digest(data) != checksum:
                raise ValueError('checkpoint checksum mismatch')
            if data['schema_version'] != SCHEMA_VERSION or data['algorithm'] != 'linear-q-learning':
                raise ValueError('unsupported checkpoint schema or algorithm')
            if data['feature_schema'] != list(FEATURE_SCHEMA) or data['actions'] != list(ACTIONS):
                raise ValueError('feature or action schema mismatch')
            policy = cls(data['use_jev'], data['seed'])
            weights = data['weights']
            if set(weights) != set(ACTIONS):
                raise ValueError('checkpoint action weights mismatch')
            for row in weights.values():
                if not isinstance(row, dict) or set(row) != set(FEATURE_SCHEMA):
                    raise ValueError('checkpoint feature weights mismatch')
                if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in row.values()):
                    raise ValueError('checkpoint weights must be finite numbers')
            episodes = data['training_episodes']
            if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 0:
                raise ValueError('invalid training episode count')
            policy.weights, policy.training_episodes = weights, episodes
            return policy
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('malformed checkpoint') from exc


def train_policy(incidents, episodes: int, seed: int, use_jev: bool,
                 advisor=None, validation_incidents=None,
                 validation_interval: int | None = None) -> tuple[LinearQPolicy, dict]:
    """Train the full budget, optionally selecting on predeclared validation probes.

    Candidate epochs are fixed before training. Selection uses mean simulator
    reward on the validation split, never test outcomes or diagnostic grading.
    Training continues through every episode regardless of the selected snapshot.
    """
    from .agent import RecoveryAgent
    from .sandbox import IncidentEnv
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError('episodes must be a positive integer')
    incidents = tuple(incidents)
    if not incidents:
        raise ValueError('training requires incidents')
    if any(getattr(incident, 'wording', 'train') != 'train' for incident in incidents):
        raise ValueError('training accepts only the train wording split')
    validation = None if validation_incidents is None else tuple(validation_incidents)
    if validation is not None:
        if not validation or any(getattr(incident, 'wording', None) != 'validation' for incident in validation):
            raise ValueError('validation selection requires nonempty validation-split incidents')
    if validation_interval is None:
        validation_interval = max(1, episodes // 4)
    if isinstance(validation_interval, bool) or not isinstance(validation_interval, int) or validation_interval <= 0:
        raise ValueError('validation_interval must be a positive integer')
    candidate_episodes = sorted(set(range(validation_interval, episodes + 1, validation_interval)) | {episodes}) if validation else []
    validation_scores = []
    selected_weights = None
    selected_after_episode = episodes
    best_validation_reward = float('-inf')
    validation_digest = _digest({'incidents': [incident.to_dict() for incident in validation]}) if validation else None
    policy = LinearQPolicy(use_jev, seed)
    schedule_rng, action_rng = random.Random(seed), random.Random(seed + 1_000_003)
    schedule = [schedule_rng.randrange(len(incidents)) for _ in range(episodes)]
    totals = dict(recovered=0, reward=0.0, actions=0, downtime_ticks=0)
    alpha, gamma = .18, .98
    for episode, index in enumerate(schedule):
        env = IncidentEnv(incidents[index], advisor=advisor)
        epsilon = max(.06, 1.0 - episode / max(1, episodes * .85))
        while not env.done:
            observation = env.observation()
            values = policy.values(observation)
            action = action_rng.choice(tuple(values)) if action_rng.random() < epsilon else max(values, key=values.get)
            next_observation, reward, done = env.step(action)
            next_value = 0.0 if done else max(policy.values(next_observation).values())
            features = policy.features(observation)
            error = reward + gamma * next_value - values[action]
            active = [(key, value) for key, value in features.items() if value]
            # Normalize the update by feature magnitude; bounded features keep
            # large conjunctions from receiving a disproportionate learning rate.
            rate = alpha / max(1.0, sum(value * value for _, value in active))
            for key, value in active:
                policy.weights[action][key] += rate * error * value
        metrics = env.metrics()
        for key in totals:
            totals[key] += metrics[key]
        completed = episode + 1
        if completed in candidate_episodes:
            before_validation = policy.checksum()
            results = [RecoveryAgent(policy).run(IncidentEnv(incident, advisor=advisor)) for incident in validation]
            if policy.checksum() != before_validation:
                raise RuntimeError('validation evaluation mutated controller weights')
            mean_reward = sum(result['reward'] for result in results) / len(results)
            validation_scores.append(dict(after_episode=completed, mean_reward=mean_reward,
                                          recovery_rate=sum(result['recovered'] for result in results) / len(results),
                                          weights_sha256=before_validation))
            # Strict comparison gives the earliest checkpoint deterministic priority
            # when two candidates attain exactly the same validation reward.
            if mean_reward > best_validation_reward:
                best_validation_reward = mean_reward
                selected_after_episode = completed
                selected_weights = deepcopy(policy.weights)
    final_training_digest = policy.checksum()
    if selected_weights is not None:
        policy.weights = selected_weights
    policy.training_episodes = selected_after_episode
    schedule_identity = [dict(id=incidents[index].id, seed=incidents[index].seed) for index in schedule]
    summary = dict(algorithm='linear-q-learning', episodes=episodes, seed=seed,
                   use_jev=use_jev, alpha=alpha, gamma=gamma, epsilon_min=.06,
                   incident_schedule_sha256=_digest({'schedule': schedule_identity}),
                   recovery_rate=totals['recovered'] / episodes,
                   mean_reward=totals['reward'] / episodes, mean_actions=totals['actions'] / episodes,
                   mean_downtime_ticks=totals['downtime_ticks'] / episodes,
                   weights_sha256=policy.checksum(), feature_count=len(FEATURE_SCHEMA),
                   evaluation_frozen=True, selected_after_episode=selected_after_episode,
                   final_training_weights_sha256=final_training_digest,
                   selection_metric='mean_validation_reward' if validation else 'final_training_episode',
                   validation_interval=validation_interval if validation else None,
                   validation_scores=validation_scores, validation_manifest_sha256=validation_digest,
                   selection_provenance=dict(split='validation', validation_ids=[incident.id for incident in validation],
                                             candidate_episodes=candidate_episodes, greedy_frozen_evaluation=True,
                                             tie_break='earliest_candidate', full_training_budget_completed=True)
                                             if validation else None)
    return policy, summary
