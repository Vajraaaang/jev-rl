import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    from jev_recovery.learning import LinearQPolicy, train_policy
except ImportError:
    LinearQPolicy = None

from test_agent import observation


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(LinearQPolicy, 'linear local controller must exist')

    def test_meaningful_factorized_features(self):
        policy = LinearQPolicy(use_jev=True)
        public = observation()
        first = policy.features(public)
        public['evidence']['config'] = {'valid': False, 'connection_release_enabled': False}
        public['evidence']['diagnosis'] = {'available': True, 'probabilities': {
            'credentials': .9, 'connections': .2, 'dependency': .1}}
        second = policy.features(public)
        self.assertNotEqual(first, second)
        self.assertEqual(second['config_bad'], 1)
        self.assertEqual(second['jev_credentials'], .9)
        without = LinearQPolicy(use_jev=False).features(public)
        self.assertEqual(without['jev_credentials'], 0)

    def test_checkpoint_roundtrip_frozen_evaluation_and_integrity(self):
        policy = LinearQPolicy(use_jev=True, seed=9)
        policy.weights['rollback']['config_bad'] = 4.0
        public = observation()
        public['evidence']['config'] = {'valid': False, 'connection_release_enabled': False}
        before = copy.deepcopy(policy.weights)
        self.assertEqual(policy.select(public), 'rollback')
        self.assertEqual(before, policy.weights)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            policy.save(path)
            loaded = LinearQPolicy.load(path)
            self.assertEqual(loaded.weights, policy.weights)
            self.assertEqual(loaded.select(public), 'rollback')
            data = json.loads(path.read_text())
            data['weights']['rollback']['config_bad'] += 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): LinearQPolicy.load(path)
        with self.assertRaises(ValueError): policy.save('relative.json')

    def test_unseen_observation_uses_learned_values_not_checklist(self):
        policy = LinearQPolicy(seed=2)
        public = observation()
        policy.weights['wait']['bias'] = 3
        self.assertEqual(policy.select(public), 'wait')
        self.assertNotIn('jev', policy.allowed_actions(public))

    def test_validation_selection_is_frozen_and_preserves_full_training_budget(self):
        from jev_recovery.scenarios import generate_suite
        incidents = generate_suite('train', 12, 7)
        validation = generate_suite('validation', 6, 8)
        final, baseline = train_policy(incidents, 20, 8, False)
        selected, summary = train_policy(incidents, 20, 8, False,
                                         validation_incidents=validation, validation_interval=5)
        self.assertEqual(summary['episodes'], 20)
        self.assertEqual(summary['incident_schedule_sha256'], baseline['incident_schedule_sha256'])
        self.assertEqual(summary['final_training_weights_sha256'], final.checksum())
        self.assertEqual([item['after_episode'] for item in summary['validation_scores']], [5, 10, 15, 20])
        best = max(summary['validation_scores'], key=lambda item: item['mean_reward'])
        self.assertEqual(summary['selected_after_episode'], best['after_episode'])
        self.assertEqual(selected.training_episodes, best['after_episode'])
        self.assertEqual(selected.checksum(), best['weights_sha256'])
        self.assertEqual(summary['selection_metric'], 'mean_validation_reward')
        self.assertTrue(summary['selection_provenance']['greedy_frozen_evaluation'])
        self.assertEqual(summary['selection_provenance']['validation_ids'], [i.id for i in validation])
        again, repeated = train_policy(incidents, 20, 8, False,
                                       validation_incidents=validation, validation_interval=5)
        self.assertEqual(selected.weights, again.weights)
        self.assertEqual(summary, repeated)

    def test_validation_ties_choose_earliest_snapshot_and_final_budget_is_evaluated(self):
        from unittest.mock import patch
        from jev_recovery.scenarios import generate_suite
        incidents = generate_suite('train', 6, 1)
        validation = generate_suite('validation', 6, 2)
        with patch('jev_recovery.agent.RecoveryAgent.run', return_value={'reward': 1.0, 'recovered': False}):
            policy, summary = train_policy(incidents, 11, 3, True,
                                          validation_incidents=validation, validation_interval=4)
        self.assertEqual(summary['selected_after_episode'], 4)
        self.assertEqual(policy.training_episodes, 4)
        self.assertEqual([item['after_episode'] for item in summary['validation_scores']], [4, 8, 11])
        self.assertEqual(summary['episodes'], 11)

    def test_validation_rejects_test_split_and_invalid_candidate_plan(self):
        from jev_recovery.scenarios import generate_suite
        incidents = generate_suite('train', 6, 1)
        with self.assertRaises(ValueError):
            train_policy(incidents, 10, 1, False, validation_incidents=generate_suite('test', 6, 2))
        with self.assertRaises(ValueError):
            train_policy(incidents, 10, 1, False, validation_incidents=[])
        for invalid in (0, -1, True):
            with self.assertRaises(ValueError):
                train_policy(incidents, 10, 1, False,
                             validation_incidents=generate_suite('validation', 6, 2), validation_interval=invalid)

    def test_training_reproducible_real_weight_updates_and_equal_budget(self):
        from jev_recovery.scenarios import generate_suite
        incidents = generate_suite('train', 12, 7)
        a, summary_a = train_policy(incidents, 30, 8, False)
        b, summary_b = train_policy(incidents, 30, 8, False)
        self.assertEqual(a.weights, b.weights)
        self.assertEqual(summary_a, summary_b)
        self.assertTrue(any(value != 0 for row in a.weights.values() for value in row.values()))
        self.assertEqual(summary_a['episodes'], 30)
        self.assertEqual(summary_a['seed'], 8)
        _, assisted = train_policy(incidents, 30, 8, True)
        self.assertEqual(assisted['episodes'], summary_a['episodes'])
        self.assertEqual(assisted['incident_schedule_sha256'], summary_a['incident_schedule_sha256'])


if __name__ == '__main__': unittest.main()
