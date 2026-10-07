"""Local tabular Q-learning, with observed-state keys and frozen greedy evaluation."""
import json
import random
from pathlib import Path
from .environment import Action, RecoveryEnv, Scenario, TRAIN_FAULTS


class QPolicy:
    def __init__(self, use_jev=False):
        self.use_jev = use_jev
        self.allowed = tuple(a for a in Action if use_jev or a != Action.JEV)
        self.q = {}

    def values(self, observation):
        return self.q.get(tuple(observation), [0.0] * len(Action))

    def choose(self, env):
        values = self.values(env.observation())
        return max(self.allowed, key=lambda a: (values[a], -int(a)))

    def save(self, path):
        Path(path).write_text(json.dumps({'use_jev': self.use_jev,
            'q': [[list(k), v] for k, v in sorted(self.q.items())]}))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        policy = cls(data['use_jev'])
        policy.q = {tuple(k): v for k, v in data['q']}
        return policy


def train(seed, episodes=20000, use_jev=False, advisor=None):
    rng = random.Random(seed)
    policy = QPolicy(use_jev)
    for episode in range(episodes):
        scenario = Scenario(rng.choice(TRAIN_FAULTS), dependency_recovers_at=rng.choice((5, 6, 7)))
        env = RecoveryEnv(scenario, advisor)
        epsilon = max(0.08, 1.0 - episode / (episodes * 0.8))
        while not env.done:
            state = env.observation()
            action = rng.choice(policy.allowed) if rng.random() < epsilon else policy.choose(env)
            new_state, reward, done = env.step(action)
            values = policy.q.setdefault(state, [0.0] * len(Action))
            future = 0 if done else max(policy.values(new_state)[a] for a in policy.allowed)
            values[action] += 0.2 * (reward + 0.97 * future - values[action])
    return policy


class Checklist:
    """Evidence-driven runbook; never consults hidden scenario fault labels."""
    def choose(self, env):
        if env.healthy():
            # Inspect configuration to avoid stopping during temporary restart relief.
            if env.known_config == -1:
                return Action.CONFIG
            if env.known_config == 1:
                return Action.ROLLBACK
            return Action.STOP
        if env.known_config == -1:
            return Action.CONFIG
        if env.known_config == 1:
            return Action.ROLLBACK
        if env.known_pool == -1:
            return Action.CONNECTIONS
        if env.known_pool == 1:
            return Action.RESTART
        if env.known_dependency == -1:
            return Action.DEPENDENCY
        # Reconnect after a fixed diagnostic wait, then re-check if recovery failed.
        if env.tick < 7:
            return Action.WAIT
        return Action.RESTART
