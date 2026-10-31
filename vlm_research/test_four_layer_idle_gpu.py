import unittest
from unittest.mock import patch
from four_layer_attention import run
from four_layer_attention.idle_gpu import schedule


class IdleGpuTests(unittest.TestCase):
    def test_resource_only_remap_restores_all_globals(self):
        originals = run.GPUS, run.ENV_KEYS, run.preflight, run.atomic_json
        algorithms = run.worker, run.supplement_generate, run.cohort, run.rank_for, run.smoke, run.CONDITIONS
        captured = []
        with patch.object(run, 'preflight', side_effect=lambda rank: rank) as checked, \
             patch.object(run, 'atomic_json', side_effect=lambda path, payload: captured.append(payload)):
            with schedule(0, 3, {'audit': True}):
                self.assertEqual(run.GPUS[0], originals[0][2])
                self.assertEqual(run.preflight(0), 2)
                self.assertEqual(set(run.ENV_KEYS), set(originals[1]) - {'cuda_visible_devices'})
                payload = {'environment': {'physical_gpu': 1, 'cuda_visible_devices': originals[0][2]}}
                run.atomic_json(run.OUT / 'shard0/smoke.json', payload)
                self.assertEqual(captured[0]['environment']['physical_gpu'], 3)
                self.assertEqual(captured[0]['environment']['logical_worker'], 0)
                self.assertEqual(captured[0]['environment']['cuda_visible_devices'], originals[0][2])
                self.assertEqual(algorithms,
                    (run.worker, run.supplement_generate, run.cohort, run.rank_for, run.smoke, run.CONDITIONS))
            checked.assert_called_once_with(2)
        self.assertEqual(originals, (run.GPUS, run.ENV_KEYS, run.preflight, run.atomic_json))

    def test_exception_restores_and_gpu0_is_forbidden(self):
        originals = run.GPUS, run.ENV_KEYS, run.preflight, run.atomic_json
        with self.assertRaisesRegex(RuntimeError, 'cleanup'):
            with schedule(0, 3, {}): raise RuntimeError('cleanup')
        self.assertEqual(originals, (run.GPUS, run.ENV_KEYS, run.preflight, run.atomic_json))
        for physical in (0, 4):
            with self.assertRaises(ValueError):
                with schedule(0, physical, {}): pass


if __name__ == '__main__': unittest.main()
