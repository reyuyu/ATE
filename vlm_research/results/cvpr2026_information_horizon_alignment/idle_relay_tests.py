"""CPU-only scheduler gates; all GPU and process APIs are mocked."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vlm_research.information_horizon_alignment import supervise as relay
from vlm_research.information_horizon_alignment.common import digest, source_hashes, verify


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(idle_max_memory_mib=1024, idle_max_utilization_percent=5)
        self.gpu = dict(pids=[], memory_mib=64, utilization=0)

    def test_idle_card(self):
        self.assertTrue(relay.idle(self.gpu, self.policy))

    def test_external_process_blocks_even_low_utilization(self):
        self.assertFalse(relay.idle(dict(self.gpu, pids=[123]), self.policy))

    def test_active_or_large_graphics_memory_blocks(self):
        self.assertFalse(relay.idle(dict(self.gpu, utilization=99), self.policy))
        self.assertFalse(relay.idle(dict(self.gpu, memory_mib=2048), self.policy))

    def test_pending_smoke_precedes_all_formal_work(self):
        self.assertEqual(relay.plan('B', [False]*4, [True, False, True, False]),
                         [(1, True), (3, True)])

    def test_completed_shards_never_relaunched(self):
        self.assertEqual(relay.plan('A', [True, False, True, False], [True]*4),
                         [(1, False), (3, False)])
        self.assertEqual(relay.plan('B', [True]*4, [True]*4), [])

    def test_launch_uuid_and_no_proxy_without_touching_sqlite(self):
        with tempfile.TemporaryDirectory() as folder:
            child = Mock(pid=321)
            with patch.object(relay, 'archive_attempt', return_value=Path(folder)), \
                    patch.object(relay.subprocess, 'Popen', return_value=child) as start:
                result = relay.launch('A', 3, False, dict(self.gpu))
            self.assertIs(result['process'], child)
            args, kwargs = start.call_args
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], relay.UUIDS[3])
            self.assertNotIn('HTTPS_PROXY', kwargs['env'])
            self.assertIn('--rank', args[0])
            self.assertNotIn('--smoke-only', args[0])
            self.assertTrue((Path(folder) / 'admission.json').exists())

    def test_signal_only_controller_children(self):
        child = Mock(pid=321)
        child.poll.return_value = None
        with patch.object(relay.os, 'killpg') as kill:
            relay.stop_own({3: dict(process=child)})
        kill.assert_called_once_with(321, relay.signal.SIGINT)

    def test_frozen_inference_protocols_unchanged(self):
        for stage in ('A', 'B'):
            verify(stage)
        policy = relay.read(relay.OUT / 'idle_relay_policy.json')
        self.assertEqual(digest(source_hashes()), policy['freeze_source_sha256'])


if __name__ == '__main__':
    unittest.main()
