import unittest
from jev_recovery.environment import Action, RecoveryEnv, Scenario, TEST_FAULTS, TRAIN_FAULTS, LOGS
from jev_recovery.policy import Checklist, QPolicy, train


class EnvironmentTests(unittest.TestCase):
    def test_restart_does_not_fix_credentials(self):
        env = RecoveryEnv(Scenario(('credentials',)))
        env.step(Action.RESTART)
        self.assertFalse(env.healthy())
        env.step(Action.ROLLBACK)
        env.step(Action.STOP)
        self.assertTrue(env.success)

    def test_restart_relapse_fails_independent_stop_probes(self):
        env = RecoveryEnv(Scenario(('connections',)))
        env.step(Action.RESTART)
        self.assertTrue(env.healthy())
        env.step(Action.STOP)
        self.assertFalse(env.success)
        self.assertGreater(env.downtime, 0)

    def test_connection_recovery_needs_durable_fix_and_restart(self):
        env = RecoveryEnv(Scenario(('connections',)))
        env.step(Action.ROLLBACK)
        self.assertFalse(env.healthy())
        env.step(Action.RESTART)
        env.step(Action.STOP)
        self.assertTrue(env.success)

    def test_dependency_restart_before_recovery_does_nothing(self):
        env = RecoveryEnv(Scenario(('dependency',)))
        env.step(Action.RESTART)
        self.assertFalse(env.healthy())
        while env.tick < 7:
            env.step(Action.WAIT)
        env.step(Action.RESTART)
        env.step(Action.STOP)
        self.assertTrue(env.success)

    def test_stop_probes_must_fit_budget(self):
        env = RecoveryEnv(Scenario((), max_ticks=4))
        env.step(Action.STOP)
        self.assertFalse(env.success)

    def test_hosted_call_accounting_is_separate_from_cache(self):
        from types import SimpleNamespace
        class Advisor:
            def diagnose(self, logs):
                return SimpleNamespace(probabilities={c: .1 for c in ('credentials', 'connections', 'dependency')},
                    source='hosted-jev', cache_hit=False, estimated_cost_usd=.002, input_tokens=100)
        env = RecoveryEnv(Scenario(()), Advisor())
        env.step(Action.JEV)
        self.assertEqual(env.api_calls, 1)
        self.assertEqual(env.input_tokens, 100)
        self.assertEqual(env.api_cost_usd, .002)

    def test_budget_enforced(self):
        env = RecoveryEnv(Scenario(('credentials',), max_ticks=3))
        for _ in range(3):
            env.step(Action.WAIT)
        self.assertTrue(env.done)
        self.assertFalse(env.success)
        with self.assertRaises(RuntimeError):
            env.step(Action.WAIT)

    def test_holdouts_disjoint(self):
        self.assertFalse(set(TRAIN_FAULTS) & set(TEST_FAULTS))
        self.assertFalse(set(LOGS['train'].values()) & set(LOGS['heldout'].values()))

    def test_checklist_recovers_all_combinations(self):
        for faults in TEST_FAULTS:
            env = RecoveryEnv(Scenario(faults, 'heldout'))
            policy = Checklist()
            while not env.done:
                env.step(policy.choose(env))
            self.assertTrue(env.success, faults)

    def test_policy_training_reproducible_and_evaluation_frozen(self):
        a = train(123, episodes=100)
        b = train(123, episodes=100)
        self.assertEqual(a.q, b.q)
        before = repr(a.q)
        env = RecoveryEnv(Scenario(('credentials', 'dependency'), 'heldout'))
        while not env.done:
            env.step(a.choose(env))
        self.assertEqual(before, repr(a.q))
        self.assertNotIn(Action.JEV, QPolicy().allowed)


if __name__ == '__main__':
    unittest.main()
