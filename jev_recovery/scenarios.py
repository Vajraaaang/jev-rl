"""Immutable, reproducible incident manifests and precommitted split boundaries."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import random

CAUSES = ('credentials', 'connections', 'dependency')
SPLITS = ('train', 'validation', 'test')
TRAIN_COMBINATIONS = ((), ('credentials',), ('connections',), ('dependency',),
                      ('credentials', 'connections'), ('credentials', 'dependency'))
TEST_COMBINATIONS = ((), ('credentials',), ('connections',), ('dependency',),
                     ('connections', 'dependency'), CAUSES)
RESERVED_COMBINATIONS = (('connections', 'dependency'), CAUSES)


@dataclass(frozen=True)
class Incident:
    """Hidden manifest. Never pass a manifest or its identifiers to a policy.

    The dependency transition is absolute simulated time; ``None`` represents
    an unrecoverable upstream. Restart flushes a pool but does not fix its leak.
    A rollback fixes configuration and leak, leaving existing saturation intact.
    """

    id: str
    faults: tuple[str, ...]
    wording: str = 'train'
    seed: int = 0
    dependency_ready_at: int | None = 8
    relapse_after: int = 4
    max_ticks: int = 48
    verification_ticks: int = 12
    restart_duration: int = 2
    log_variant: int = 0
    noise: bool = False

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 160:
            raise ValueError('Incident id must be a nonempty string of at most 160 characters')
        if not isinstance(self.faults, tuple):
            raise TypeError('faults must be a tuple')
        if any(not isinstance(c, str) or c not in CAUSES for c in self.faults):
            raise ValueError('Unknown fault')
        if len(set(self.faults)) != len(self.faults):
            raise ValueError('Duplicate fault')
        # Canonical order makes combinations stable under JSON round trips.
        object.__setattr__(self, 'faults', tuple(c for c in CAUSES if c in self.faults))
        if self.wording not in SPLITS:
            raise ValueError('Unknown wording split')
        for name in ('seed', 'relapse_after', 'max_ticks', 'verification_ticks', 'restart_duration', 'log_variant'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'{name} must be an integer')
            if value < (0 if name in ('seed', 'log_variant') else 1):
                raise ValueError(f'Invalid {name}')
        if self.log_variant > 2:
            raise ValueError('log_variant must be between 0 and 2')
        if 'connections' in self.faults and self.verification_ticks <= self.relapse_after:
            raise ValueError('verification_ticks must exceed the bounded connection relapse window')
        if self.dependency_ready_at is not None:
            if isinstance(self.dependency_ready_at, bool) or not isinstance(self.dependency_ready_at, int):
                raise TypeError('dependency_ready_at must be a nonnegative integer or null')
            if self.dependency_ready_at < 0:
                raise ValueError('dependency_ready_at must be nonnegative')
        if not isinstance(self.noise, bool):
            raise TypeError('noise must be boolean')

    @property
    def recoverable(self) -> bool:
        """Privileged structural feasibility, assuming an informed optimal actor.

        This is an evaluator descriptor, not a signal available to a controller.
        It accounts for causal operation durations and a complete final probe.
        """
        needs_rollback = bool({'credentials', 'connections'} & set(self.faults))
        earliest = 2 if needs_rollback else 0
        needs_restart = 'connections' in self.faults or 'dependency' in self.faults
        if 'dependency' in self.faults:
            if self.dependency_ready_at is None:
                return False
            earliest = max(earliest, self.dependency_ready_at)
        if needs_restart:
            earliest += self.restart_duration
        return earliest + self.verification_ticks <= self.max_ticks

    def to_dict(self) -> dict:
        data = asdict(self)
        data['faults'] = list(self.faults)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Incident':
        if not isinstance(data, dict):
            raise TypeError('Incident manifest must be an object')
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown incident fields')
        values = dict(data)
        if not isinstance(values.get('faults'), (list, tuple)):
            raise TypeError('faults must be a list or tuple')
        values['faults'] = tuple(values['faults'])
        return cls(**values)


def generate_suite(split: str, count: int, seed: int) -> list[Incident]:
    """Return deterministic balanced combinations with orthogonal stress variants.

    Combinations reserved for testing never enter training or validation.
    Wording dictionaries are also separate, so validation-based model selection
    cannot expose evaluation text. Prefixes of a suite are stable as count grows.
    """
    if split not in SPLITS:
        raise ValueError('split must be train, validation, or test')
    for name, value in (('count', count), ('seed', seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f'{name} must be an integer')
        if value < 0:
            raise ValueError(f'{name} must be nonnegative')
    combinations = TEST_COMBINATIONS if split == 'test' else TRAIN_COMBINATIONS
    rng = random.Random(f'recovery-lab-v2:{split}:{seed}')
    suite = []
    for i in range(count):
        # Each combination cycles all three dependency regimes independently.
        block = i // len(combinations)
        regime = block % 3
        ready_at = rng.randint(6, 12) if regime == 0 else rng.randint(16, 24) if regime == 1 else None
        suite.append(Incident(
            id=f'{split}-{seed}-{i:05d}', faults=combinations[i % len(combinations)],
            wording=split, seed=rng.randrange(2**31), dependency_ready_at=ready_at,
            relapse_after=rng.randint(3, 11), restart_duration=rng.randint(1, 3),
            log_variant=rng.randrange(3), noise=bool(rng.getrandbits(1)),
        ))
    return suite
