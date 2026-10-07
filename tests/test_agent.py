import copy
import unittest

try:
    from jev_recovery.agent import (RecoveryAgent, ChecklistPolicy, RandomPolicy,
                                    GreedyRepairPolicy, admissible_actions)
except ImportError:
    RecoveryAgent = None


def observation(**updates):
    result = dict(schema_version=2, tick=0, remaining_ticks=48, verification_ticks=6,
                  health=False, healthy_streak=0, evidence={k: None for k in
                  ('logs', 'config', 'connections', 'dependency', 'diagnosis')},
                  action_counts={}, last_action=None, last_outcome=None)
    result.update(updates)
    return result


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(RecoveryAgent, 'v2 agent runtime must exist')

    def test_policy_receives_defensive_copy_and_no_environment(self):
        class MutatingPolicy:
            def select(self, public):
                public['evidence']['config'] = {'valid': True}
                return 'config'
        public = observation()
        before = copy.deepcopy(public)
        decision = RecoveryAgent(MutatingPolicy()).decide(public)
        self.assertEqual(public, before)
        self.assertEqual(decision.action, 'config')
        self.assertNotIn('faults', decision.evidence)

    def test_fresh_evidence_is_not_reinspected_without_a_state_change(self):
        public = observation()
        public['evidence']['config'] = {'valid': True, 'connection_release_enabled': True}
        public['evidence']['connections'] = {'exhausted': False}
        self.assertNotIn('config', admissible_actions(public))
        self.assertNotIn('connections', admissible_actions(public))
        public['action_counts'] = {'rollback': 1, 'config': 1, 'connections': 1}
        public['evidence']['config'] = public['evidence']['connections'] = None
        self.assertIn('config', admissible_actions(public))
        self.assertIn('connections', admissible_actions(public))

    def test_consult_requires_collected_logs_and_failed_diagnosis_is_not_retried(self):
        public = observation()
        self.assertNotIn('jev', admissible_actions(public))
        public['evidence']['logs'] = 'database error'
        self.assertIn('jev', admissible_actions(public))
        public['action_counts']['jev'] = 1
        public['evidence']['diagnosis'] = {'available': False}
        self.assertNotIn('jev', admissible_actions(public))

    def test_intervention_limit_and_repeated_evidence_are_blocked(self):
        public = observation(action_counts={'rollback': 3, 'restart': 3, 'logs': 1})
        allowed = admissible_actions(public)
        self.assertNotIn('restart', allowed)
        self.assertNotIn('rollback', allowed)
        self.assertNotIn('logs', allowed)
        class BadPolicy:
            def select(self, public): return 'restart'
        agent = RecoveryAgent(BadPolicy())
        decision = agent.decide(public)
        self.assertEqual(decision.action, 'escalate')
        self.assertEqual(len(agent.safety_blocks), 1)

    def test_invalid_policy_and_policy_error_gracefully_escalate(self):
        class BrokenPolicy:
            def select(self, public): raise RuntimeError('unavailable')
        decision = RecoveryAgent(BrokenPolicy()).decide(observation())
        self.assertEqual(decision.action, 'escalate')
        self.assertIn('policy', decision.reason.lower())

    def test_stop_requires_health_and_verification_budget(self):
        self.assertNotIn('stop', admissible_actions(observation()))
        self.assertNotIn('stop', admissible_actions(observation(health=True, remaining_ticks=5)))
        self.assertIn('stop', admissible_actions(observation(health=True)))

    def test_checklist_composes_repairs_and_rechecks_dependency(self):
        public = observation()
        policy = ChecklistPolicy()
        self.assertEqual(policy.select(public), 'config')
        public['evidence']['config'] = {'valid': True, 'connection_release_enabled': False}
        self.assertEqual(policy.select(public), 'rollback')
        public['evidence']['config']['connection_release_enabled'] = True
        self.assertEqual(policy.select(public), 'connections')
        public['evidence']['connections'] = {'exhausted': True}
        public['evidence']['dependency'] = {'reachable': False, 'circuit_open': True, 'observed_at': 0}
        public['tick'] = 1
        self.assertEqual(policy.select(public), 'wait')
        public['tick'] = 3
        self.assertEqual(policy.select(public), 'dependency')
        public['evidence']['dependency']['reachable'] = True
        self.assertEqual(policy.select(public), 'restart')

    def test_checklist_recovers_every_feasible_sandbox_stress_combination(self):
        from jev_recovery.scenarios import generate_suite
        from jev_recovery.sandbox import IncidentEnv
        for incident in generate_suite('test', 36, 99):
            env = IncidentEnv(incident)
            result = RecoveryAgent(ChecklistPolicy()).run(env)
            self.assertEqual(result['recovered'], incident.recoverable, incident.id)
            self.assertEqual(result['safety_violations'], 0)
            self.assertFalse(result['safety_blocks'])
            self.assertLessEqual(result['interventions'], 6)

    def test_failed_diagnosis_accounted_separately_from_success(self):
        from jev_recovery.scenarios import Incident
        from jev_recovery.sandbox import IncidentEnv
        class ConsultPolicy:
            def select(self, public):
                if public['evidence']['logs'] is None: return 'logs'
                if public['evidence']['diagnosis'] is None: return 'jev'
                return 'escalate'
        result = RecoveryAgent(ConsultPolicy()).run(IncidentEnv(Incident('failed-advisor', ('credentials',))))
        self.assertEqual(result['diagnosis_failures'], 1)
        self.assertFalse(result['recovered'])
        self.assertGreater(result['tool_cost_units'], 0)
        self.assertEqual(result['safety_violations'], 0)

    def test_hidden_root_fields_filtered_before_policy(self):
        class InspectPolicy:
            def select(self, public):
                assert 'faults' not in public and 'incident' not in public
                return 'config'
        public = observation()
        public.update(faults=('credentials',), incident='secret')
        self.assertEqual(RecoveryAgent(InspectPolicy()).decide(public).action, 'config')

    def test_random_reproducible_and_baselines_masked(self):
        a, b = RandomPolicy(4), RandomPolicy(4)
        public = observation()
        self.assertEqual([a.select(public) for _ in range(20)], [b.select(public) for _ in range(20)])
        self.assertIn(GreedyRepairPolicy().select(public), admissible_actions(public))


if __name__ == '__main__': unittest.main()
