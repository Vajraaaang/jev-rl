import copy
import json
import unittest

from jev_recovery.jev import Diagnosis, DiagnosticProvider
from jev_recovery.sandbox import ACTIONS, IncidentEnv
from jev_recovery.scenarios import Incident, generate_suite


def env(faults=(), **kwargs):
    return IncidentEnv(Incident('unit', faults, 'train', 1, **kwargs))


class SandboxTests(unittest.TestCase):
    def test_only_public_evidence_and_defensive_copies(self):
        a, b = env(('credentials',)), env(('connections',))
        self.assertEqual(a.observation(), b.observation())
        observed = a.observation()
        for forbidden in ('faults', 'seed', 'id', 'wording', 'recoverable'):
            self.assertNotIn(forbidden, observed)
        self.assertTrue(all(v is None for v in observed['evidence'].values()))
        observed['evidence']['config'] = {'valid': True}
        self.assertIsNone(a.observation()['evidence']['config'])
        json.dumps(a.observation())

    def test_credentials_need_rollback_and_restart_does_nothing(self):
        sandbox = env(('credentials',))
        sandbox.step('restart')
        self.assertFalse(sandbox.healthy())
        sandbox.step('config')
        self.assertFalse(sandbox.observation()['evidence']['config']['valid'])
        sandbox.step('rollback')
        self.assertTrue(sandbox.healthy())
        _, _, done = sandbox.step('stop')
        self.assertTrue(done)
        self.assertTrue(sandbox.metrics()['recovered'])

    def test_pool_restart_temporarily_helps_then_relapses(self):
        sandbox = env(('connections',), relapse_after=4)
        sandbox.step('restart')
        self.assertTrue(sandbox.healthy())
        for _ in range(4):
            sandbox.step('wait')
        self.assertFalse(sandbox.healthy())
        sandbox.step('rollback')
        self.assertFalse(sandbox.healthy(), 'rollback cannot flush already saturated pool')
        sandbox.step('restart')
        sandbox.step('stop')
        self.assertTrue(sandbox.metrics()['recovered'])

    def test_independent_stop_catches_relapse(self):
        sandbox = env(('connections',), relapse_after=4, verification_ticks=6)
        sandbox.step('restart')
        self.assertTrue(sandbox.healthy())
        sandbox.step('stop')
        self.assertFalse(sandbox.metrics()['recovered'])
        self.assertEqual('verification_failed', sandbox.metrics()['termination'])
        self.assertEqual(6, len(sandbox.trace[-1]['outcome']['health_probes']))

    def test_default_stop_catches_slow_relapse(self):
        sandbox = env(('connections',), relapse_after=11)
        sandbox.step('restart')
        sandbox.step('stop')
        self.assertFalse(sandbox.metrics()['recovered'])
        self.assertEqual(12, len(sandbox.trace[-1]['outcome']['health_probes']))

    def test_dependency_requires_readiness_then_breaker_reset(self):
        sandbox = env(('dependency',), dependency_ready_at=5)
        sandbox.step('restart')
        self.assertFalse(sandbox.healthy())
        while sandbox.tick < 5:
            sandbox.step('wait')
        sandbox.step('dependency')
        self.assertTrue(sandbox.observation()['evidence']['dependency']['reachable'])
        self.assertFalse(sandbox.healthy(), 'reachable upstream leaves latched circuit open')
        sandbox.step('restart')
        sandbox.step('stop')
        self.assertTrue(sandbox.metrics()['recovered'])

    def test_permanent_dependency_escalates(self):
        sandbox = env(('dependency',), dependency_ready_at=None)
        sandbox.step('rollback')
        sandbox.step('restart')
        sandbox.step('escalate')
        self.assertFalse(sandbox.metrics()['recovered'])
        self.assertEqual('escalated', sandbox.metrics()['termination'])

    def test_jev_cannot_read_uncollected_logs(self):
        class Recorder:
            def __init__(self):
                self.calls = []
            def diagnose(self, logs):
                self.calls.append(logs)
                return Diagnosis(dict(credentials=.9, connections=.1, dependency=.1), 'test', 'fixed', 0, 0., False)
        provider = Recorder()
        sandbox = IncidentEnv(Incident('x', ('credentials',), 'train', 1), provider)
        sandbox.step('jev')
        self.assertEqual([], provider.calls)
        self.assertEqual(1, sandbox.metrics()['safety_violations'])
        sandbox.step('logs')
        captured = sandbox.observation()['evidence']['logs']
        sandbox.step('rollback')
        sandbox.step('jev')
        self.assertEqual([captured], provider.calls, 'consult uses captured evidence even if stale')

    def test_diagnostic_unavailability_observed_and_costed(self):
        sandbox = env(('credentials',))
        sandbox.step('logs')
        _, reward, _ = sandbox.step('jev')
        self.assertFalse(sandbox.observation()['evidence']['diagnosis']['available'])
        self.assertGreater(sandbox.metrics()['tool_cost_units'], 0)
        self.assertLess(reward, 0)
        self.assertFalse(sandbox.metrics()['recovered'])

    def test_provider_failure_is_not_success_and_secret_not_leaked(self):
        class Failing:
            def diagnose(self, logs):
                raise RuntimeError('secret-token-should-not-leak')
        sandbox = IncidentEnv(Incident('x', ('credentials',), 'test', 1), Failing())
        sandbox.step('logs')
        sandbox.step('jev')
        self.assertNotIn('secret-token', json.dumps(sandbox.trace))
        self.assertFalse(sandbox.observation()['evidence']['diagnosis']['available'])

    def test_provider_provenance_and_usage(self):
        class HostedStub:
            def diagnose(self, logs):
                return Diagnosis(dict(credentials=.9, connections=.1, dependency=.1), 'hosted-jev', 'jev-1.13.0', 100, .0000042, False)
        sandbox = IncidentEnv(Incident('x', ('credentials',), 'test', 1), HostedStub())
        sandbox.step('logs')
        sandbox.step('jev')
        self.assertEqual(1, sandbox.metrics()['api_calls'])
        self.assertEqual(100, sandbox.metrics()['input_tokens'])
        self.assertAlmostEqual(.0000042, sandbox.metrics()['api_cost_usd'])
        self.assertEqual('hosted-jev', sandbox.trace[-1]['provider']['source'])

    def test_cached_usage_is_not_charged_as_fresh_hosted_call(self):
        class CachedStub:
            def diagnose(self, logs):
                return Diagnosis(dict(credentials=.9, connections=.1, dependency=.1), 'hosted-jev', 'jev-1.13.0', 100, .0000042, True)
        sandbox = IncidentEnv(Incident('x', ('credentials',), 'test', 1), CachedStub())
        sandbox.step('logs')
        sandbox.step('jev')
        self.assertEqual(1, sandbox.metrics()['cache_hits'])
        self.assertEqual(0, sandbox.metrics()['api_calls'])
        self.assertEqual(0, sandbox.metrics()['input_tokens'])
        self.assertEqual(0, sandbox.metrics()['api_cost_usd'])
        self.assertEqual(100, sandbox.trace[-1]['provider']['input_tokens'])

    def test_perfect_diagnosis_cannot_grade_its_own_success(self):
        class Confident:
            def diagnose(self, logs):
                return Diagnosis(dict(credentials=0., connections=0., dependency=0.), 'test', 'fixed', 0, 0., False)
        sandbox = IncidentEnv(Incident('x', ('credentials',), 'test', 1), Confident())
        sandbox.step('logs')
        sandbox.step('jev')
        sandbox.step('stop')
        self.assertFalse(sandbox.metrics()['recovered'])
        self.assertFalse(any(sandbox.trace[-1]['outcome']['health_probes']))

    def test_bad_provider_payload_fails_closed(self):
        cases = [dict(credentials=float('nan'), connections=.5, dependency=.5),
                 dict(credentials=2., connections=.5, dependency=.5),
                 dict(credentials=True, connections=.5, dependency=.5),
                 dict(credentials=.5, connections=.5)]
        for probabilities in cases:
            class Invalid:
                def diagnose(self, logs):
                    return Diagnosis(probabilities, 'test', 'fixed', 0, 0., False)
            sandbox = IncidentEnv(Incident('x', ('credentials',), 'test', 1), Invalid())
            sandbox.step('logs')
            sandbox.step('jev')
            self.assertFalse(sandbox.observation()['evidence']['diagnosis']['available'])
            json.dumps(sandbox.trace, allow_nan=False)

    def test_repair_invalidation_does_not_rewrite_collected_logs(self):
        sandbox = env(('credentials', 'connections', 'dependency'))
        for action in ('logs', 'config', 'connections', 'dependency'):
            sandbox.step(action)
        logs = sandbox.observation()['evidence']['logs']
        sandbox.step('rollback')
        self.assertIsNone(sandbox.observation()['evidence']['config'])
        self.assertIsNone(sandbox.observation()['evidence']['connections'])
        sandbox.step('restart')
        self.assertIsNone(sandbox.observation()['evidence']['dependency'])
        self.assertEqual(logs, sandbox.observation()['evidence']['logs'])

    def test_manifest_and_action_replay_is_identical(self):
        for incident in generate_suite('test', 24, 20):
            a, b = IncidentEnv(incident), IncidentEnv(Incident.from_dict(incident.to_dict()))
            for action in ('logs', 'config', 'rollback', 'connections', 'dependency', 'restart', 'wait', 'stop'):
                self.assertEqual(a.step(action), b.step(action))
                if a.done:
                    break
            self.assertEqual(a.metrics(), b.metrics())
            self.assertEqual(a.trace, b.trace)

    def test_stop_probe_cost_is_independent_of_diagnosis(self):
        sandbox = env()
        _, reward, done = sandbox.step('stop')
        self.assertTrue(done)
        self.assertEqual(12, sandbox.tick)
        self.assertTrue(sandbox.metrics()['recovered'])
        self.assertEqual([True] * 12, sandbox.trace[-1]['outcome']['health_probes'])
        self.assertAlmostEqual(39.7, reward)

    def test_restart_has_real_downtime_and_unnecessary_cost(self):
        sandbox = env(restart_duration=3)
        sandbox.step('restart')
        self.assertEqual(3, sandbox.metrics()['downtime_ticks'])
        self.assertEqual(1, sandbox.metrics()['unnecessary_interventions'])
        self.assertEqual(1, sandbox.metrics()['interventions'])
        self.assertTrue(sandbox.healthy())

    def test_horizon_strict_and_verification_not_shortened(self):
        sandbox = env(max_ticks=5, verification_ticks=6)
        sandbox.step('stop')
        self.assertEqual(5, sandbox.tick)
        self.assertFalse(sandbox.metrics()['recovered'])
        self.assertEqual('verification_budget_exhausted', sandbox.metrics()['termination'])
        with self.assertRaises(RuntimeError):
            sandbox.step('wait')
        sandbox = env(max_ticks=1, restart_duration=3)
        sandbox.step('restart')
        self.assertEqual(1, sandbox.tick)
        self.assertFalse(sandbox.metrics()['recovered'])

    def test_invalid_actions_do_not_mutate(self):
        sandbox = env()
        before = copy.deepcopy(sandbox.metrics())
        for action in ('delete-production', 3, None, [], True):
            with self.assertRaises((ValueError, TypeError)):
                sandbox.step(action)
        self.assertEqual(before, sandbox.metrics())

    def test_reward_components_sum_and_trace_serializes(self):
        sandbox = env(('credentials', 'connections'))
        for action in ('logs', 'config', 'rollback', 'restart', 'stop'):
            before = sandbox.observation()
            after, reward, _ = sandbox.step(action)
            event = sandbox.trace[-1]
            self.assertEqual(before, event['observation_before'])
            self.assertEqual(after, event['observation_after'])
            self.assertAlmostEqual(reward, sum(event['reward_components'].values()))
        self.assertAlmostEqual(sandbox.metrics()['reward'], sum(x['reward'] for x in sandbox.trace))
        json.dumps(sandbox.trace)

    def test_seeded_noisy_logs_deterministic_and_wording_disjoint(self):
        collected = {}
        for split in ('train', 'validation', 'test'):
            incident = Incident(split, ('credentials', 'connections', 'dependency'), split, 40, log_variant=2, noise=True)
            a, b = IncidentEnv(incident), IncidentEnv(incident)
            self.assertEqual(a.logs(), b.logs())
            collected[split] = a.logs()
        self.assertEqual(3, len(set(collected.values())))

    def test_exhaustive_suite_terminates_safely(self):
        for incident in generate_suite('test', 60, 11):
            sandbox = IncidentEnv(incident, DiagnosticProvider())
            for i in range(incident.max_ticks):
                if sandbox.done:
                    break
                sandbox.step(ACTIONS[i % len(ACTIONS[:-2])])
            self.assertTrue(sandbox.done)
            self.assertLessEqual(sandbox.tick, incident.max_ticks)
            self.assertGreaterEqual(sandbox.metrics()['downtime_ticks'], 0)
            self.assertLessEqual(sandbox.metrics()['downtime_ticks'], sandbox.tick)
