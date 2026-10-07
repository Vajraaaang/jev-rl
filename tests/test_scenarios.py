import json
import unittest

from jev_recovery.scenarios import CAUSES, Incident, generate_suite


class ScenarioTests(unittest.TestCase):
    def test_deterministic_and_serializable_suite(self):
        for split in ('train', 'validation', 'test'):
            suite = generate_suite(split, 60, 11)
            self.assertEqual(suite, generate_suite(split, 60, 11))
            self.assertEqual(60, len({x.id for x in suite}))
            self.assertEqual(suite, [Incident.from_dict(json.loads(json.dumps(x.to_dict()))) for x in suite])
            self.assertNotEqual(suite, generate_suite(split, 60, 12))

    def test_reserved_combinations_and_disjoint_wording(self):
        suites = {s: generate_suite(s, 90, 7) for s in ('train', 'validation', 'test')}
        training = {x.faults for x in suites['train'] + suites['validation']}
        heldout = {x.faults for x in suites['test']}
        self.assertNotIn(('connections', 'dependency'), training)
        self.assertNotIn(CAUSES, training)
        self.assertIn(('connections', 'dependency'), heldout)
        self.assertIn(CAUSES, heldout)
        self.assertEqual({'train'}, {x.wording for x in suites['train']})
        self.assertEqual({'validation'}, {x.wording for x in suites['validation']})
        self.assertEqual({'test'}, {x.wording for x in suites['test']})

    def test_suite_has_noise_relapse_and_permanent_variants(self):
        suite = generate_suite('test', 90, 1)
        self.assertTrue(any(x.noise for x in suite))
        self.assertTrue(any(x.dependency_ready_at is None and 'dependency' in x.faults for x in suite))
        self.assertTrue(any(x.dependency_ready_at is not None and 16 <= x.dependency_ready_at <= 24 for x in suite))
        self.assertGreater(len({x.relapse_after for x in suite}), 1)

    def test_invalid_incident(self):
        invalid = [dict(faults=('evil',)), dict(faults=('credentials', 'credentials')),
                   dict(wording='heldout'), dict(max_ticks=0), dict(seed=True),
                   dict(verification_ticks=0), dict(restart_duration=-1),
                   dict(dependency_ready_at=-1), dict(noise=1), dict(log_variant=5)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                Incident(**(dict(id='i', faults=()) | change))
        with self.assertRaises(ValueError):
            Incident.from_dict(dict(id='i', faults=[], arbitrary='ignored'))

    def test_invalid_generator(self):
        for args in [('bad', 2, 0), ('train', -1, 0), ('train', True, 0), ('train', 2, False)]:
            with self.subTest(args=args), self.assertRaises((ValueError, TypeError)):
                generate_suite(*args)
        self.assertEqual([], generate_suite('train', 0, 0))

    def test_permanent_dependency_is_unrecoverable(self):
        self.assertFalse(Incident('x', ('dependency',), 'test', 1, dependency_ready_at=None).recoverable)
        self.assertTrue(Incident('x', ('credentials',), 'test', 1, dependency_ready_at=None).recoverable)

    def test_recoverability_accounts_for_necessary_action_budget(self):
        self.assertTrue(Incident('x', ('credentials', 'connections'), max_ticks=16).recoverable)
        self.assertFalse(Incident('x', ('credentials', 'connections'), max_ticks=15).recoverable)
        self.assertFalse(Incident('x', ('dependency',), dependency_ready_at=47).recoverable)

    def test_verification_covers_every_generated_relapse(self):
        for incident in generate_suite('test', 90, 11):
            self.assertEqual(12, incident.verification_ticks)
            self.assertGreater(incident.verification_ticks, incident.relapse_after)
        with self.assertRaises(ValueError):
            Incident('x', ('connections',), relapse_after=11, verification_ticks=6)

    def test_generation_seed_changes_real_dynamics(self):
        a, b = generate_suite('test', 120, 11), generate_suite('test', 120, 22)
        def dynamics(suite):
            return [(x.dependency_ready_at, x.relapse_after, x.restart_duration) for x in suite]
        self.assertNotEqual(dynamics(a), dynamics(b))
        self.assertEqual({1, 2, 3}, {x.restart_duration for x in a})
        self.assertTrue(all(3 <= x.relapse_after <= 11 for x in a))
        fast = {x.dependency_ready_at for x in a if x.dependency_ready_at is not None and x.dependency_ready_at < 16}
        slow = {x.dependency_ready_at for x in a if x.dependency_ready_at is not None and x.dependency_ready_at >= 16}
        self.assertGreater(len(fast), 1)
        self.assertGreater(len(slow), 1)
        self.assertTrue(all(6 <= value <= 12 for value in fast))
        self.assertTrue(all(16 <= value <= 24 for value in slow))

    def test_manifest_is_frozen_and_prefix_stable(self):
        from dataclasses import FrozenInstanceError
        incident = Incident('i', ('dependency', 'credentials'))
        self.assertEqual(('credentials', 'dependency'), incident.faults)
        with self.assertRaises(FrozenInstanceError):
            incident.seed = 3
        self.assertEqual(generate_suite('test', 20, 9), generate_suite('test', 50, 9)[:20])
