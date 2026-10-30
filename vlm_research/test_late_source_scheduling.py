"""CPU tests for the operational-only three-GPU amendment."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from paper_rq1_late_attention_sources import launch
from paper_rq1_late_attention_sources.scheduling import SHARD_GPU, queues, configuration
from paper_rq1_late_attention_sources.protocol import GPUS


class Tests(unittest.TestCase):
    def test_all_frozen_shards_once_and_gpu0_excluded(self):
        q=queues()
        self.assertEqual(sorted(rank for ranks in q.values() for rank in ranks),list(range(4)))
        self.assertEqual(set(q),{1,2,3})
        self.assertEqual(q[1],(0,3))
        self.assertTrue(configuration()['same_image_full_ablation_same_gpu'])
        self.assertEqual(set(SHARD_GPU.values()),{1,2,3})

    def test_same_physical_gpu_for_all_stages_and_sequential_shards(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(launch,'OUT',Path(folder)),\
                patch.object(launch.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
            for stage in ['smoke','full','ablation']:
                self.assertIsNone(launch.run_queue(stage,1,(0,3)))
            self.assertEqual([c.args[0][-1] for c in run.call_args_list],['0','3']*3)
            self.assertTrue(all(c.kwargs['env']['CUDA_VISIBLE_DEVICES']==GPUS[1] for c in run.call_args_list))

    def test_failure_stops_queue_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(launch,'OUT',Path(folder)),\
                patch.object(launch.subprocess,'run',return_value=SimpleNamespace(returncode=2)) as run:
            result=launch.run_queue('smoke',1,(0,3))
            self.assertEqual(result,dict(rank=0,physical_gpu=1,returncode=2))
            self.assertEqual(run.call_count,1)
            self.assertFalse(list(Path(folder).rglob('completion.json')))


if __name__=='__main__':unittest.main()
