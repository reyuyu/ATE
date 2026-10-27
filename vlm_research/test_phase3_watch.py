"""Watchdog tests with mocked scheduler/GPU calls; never launch the experiment."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import phase3_watch as watch


class WatchdogTest(unittest.TestCase):
    def test_threshold_and_any_single_gpu(self):
        self.assertIsNone(watch.choose_gpu([{'index': 0, 'free_mib': 15999, 'utilization_percent': 0}]))
        rows = [{'index': 0, 'free_mib': 6000, 'utilization_percent': 100},
                {'index': 1, 'free_mib': 24000, 'utilization_percent': 0}]
        self.assertEqual(watch.choose_gpu(rows)['index'], 1)
        rows[0]['free_mib'] = 16000
        self.assertEqual(watch.choose_gpu(rows)['index'], 1)

    def test_gpu_csv_parse(self):
        output = subprocess.CompletedProcess([], 0, '0, GPU-a, NVIDIA RTX 4090, 6928, 100\n1, GPU-b, NVIDIA RTX 4090, 23000, 0\n', '')
        with patch.object(watch.subprocess, 'run', return_value=output):
            rows = watch.read_gpus()
        self.assertEqual(rows[1]['uuid'], 'GPU-b')
        self.assertEqual(rows[1]['free_mib'], 23000)

    def test_install_preserves_other_cron_entries_and_deduplicates(self):
        old = '@reboot my-existing-job\n*/15 * * * * old-command ' + watch.MARKER + '\n'
        read = subprocess.CompletedProcess([], 0, old, '')
        with patch.object(watch.subprocess, 'run', side_effect=[read, subprocess.CompletedProcess([], 0)]) as run:
            watch.cron_update(Path('/tmp/ate-watch-test'), True)
        written = run.call_args_list[1].kwargs['input']
        self.assertIn('@reboot my-existing-job\n', written)
        self.assertEqual(written.count(watch.MARKER), 1)
        self.assertIn('*/15 * * * * /usr/bin/python3 ', written)
        self.assertNotIn('old-command', written)

    def test_remove_preserves_other_cron_entries(self):
        old = '@reboot original\n*/15 * * * * command ' + watch.MARKER + '\n'
        read = subprocess.CompletedProcess([], 0, old, '')
        with patch.object(watch.subprocess, 'run', side_effect=[read, subprocess.CompletedProcess([], 0)]) as run:
            watch.cron_update(Path('/tmp/ate-watch-test'), False)
        self.assertEqual(run.call_args_list[1].kwargs['input'], '@reboot original\n')

    def test_occupied_gpu_does_not_run_or_modify_cron(self):
        with tempfile.TemporaryDirectory() as temp:
            state_dir = Path(temp)
            with patch.object(watch, 'read_gpus', return_value=[{'index': 0, 'free_mib': 6000}]), \
                 patch.object(watch, 'execute_experiment') as execute, patch.object(watch, 'cron_update') as cron:
                result = watch.tick(state_dir)
            self.assertEqual(result['status'], 'waiting')
            execute.assert_not_called()
            cron.assert_not_called()
            self.assertTrue((state_dir / 'status.json').exists())

    def test_free_gpu_runs_once_and_terminal_state_stops_polling(self):
        def complete(state_dir, state, gpu):
            state.update(status='completed', selected_gpu=gpu['index'])
        with tempfile.TemporaryDirectory() as temp:
            state_dir = Path(temp)
            with patch.object(watch, 'read_gpus', return_value=[{'index': 1, 'free_mib': 24000, 'utilization_percent': 0}]) as read, \
                 patch.object(watch, 'execute_experiment', side_effect=complete) as execute, \
                 patch.object(watch, 'cron_update') as cron:
                first = watch.tick(state_dir)
                second = watch.tick(state_dir)
            self.assertEqual(first['selected_gpu'], 1)
            self.assertEqual(second['status'], 'completed')
            self.assertEqual(read.call_count, 1)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(cron.call_count, 2)

    def test_interrupted_run_is_not_silently_repeated(self):
        with tempfile.TemporaryDirectory() as temp:
            state_dir = Path(temp)
            watch.write_json(state_dir / 'status.json', {'status': 'running', 'run_directory': 'partial'})
            with patch.object(watch, 'read_gpus') as read, patch.object(watch, 'execute_experiment') as execute, \
                 patch.object(watch, 'cron_update') as cron:
                state = watch.tick(state_dir)
            self.assertEqual(state['status'], 'failed')
            read.assert_not_called()
            execute.assert_not_called()
            cron.assert_called_once_with(state_dir, False)

    def test_wrong_branch_prevents_automatic_publication(self):
        with patch.object(watch.subprocess, 'check_output', return_value='other-branch\n'), \
             patch.object(watch.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Expected main'):
                watch.publish(Path('/tmp/unused'), True, Path('/tmp/unused-state'))
        run.assert_not_called()

    def test_cuda_resource_race_returns_to_waiting_without_publication(self):
        def reject(command, log_path, env=None, timeout=None):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
            watch.write_json(log_path.parent / 'resource_preflight.json', {'ready': False})
            return 2
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'project'
            (project / 'results/phase3_knockout').mkdir(parents=True)
            state_dir = Path(temp) / 'state'
            state_dir.mkdir()
            state = {'status': 'waiting'}
            with patch.object(watch, 'PROJECT', project), patch.object(watch, 'notify'), \
                 patch.object(watch, 'run_logged', side_effect=reject), patch.object(watch, 'publish') as publish:
                watch.execute_experiment(state_dir, state, {'index': 1, 'free_mib': 16000})
            self.assertEqual(state['status'], 'waiting')
            publish.assert_not_called()
            self.assertTrue(Path(state['last_resource_recheck']).is_dir())
            self.assertNotIn('run_directory', state)

    def test_success_requires_complete_run_validation_and_regression_before_publish(self):
        def complete(command, log_path, env=None, timeout=None):
            if command[1].endswith('knockout_probe.py'):
                folder = log_path.parent
                watch.write_json(folder / 'run.json', {'completed': True, 'cutoffs': watch.CUTOFFS,
                                                       'samples_completed': ['sample']})
                target = folder / 'samples/sample'
                target.mkdir(parents=True)
                watch.write_json(target / 'correctness_validation.json', {'passed': True})
            return 0
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'project'
            (project / 'results/phase3_knockout').mkdir(parents=True)
            (project / 'configs').mkdir()
            watch.write_json(project / 'configs/phase3_samples.json', {'samples': [{'id': 'sample'}]})
            state_dir = Path(temp) / 'state'
            state_dir.mkdir()
            state = {'status': 'waiting'}
            with patch.object(watch, 'PROJECT', project), patch.object(watch, 'notify'), \
                 patch.object(watch, 'run_logged', side_effect=complete) as run, \
                 patch.object(watch, 'publish', return_value='test-commit') as publish:
                watch.execute_experiment(state_dir, state, {'index': 1, 'free_mib': 24000})
            self.assertEqual(run.call_count, 2)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['commit'], 'test-commit')
            self.assertTrue(state['github_remote_verified'])
            publish.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
