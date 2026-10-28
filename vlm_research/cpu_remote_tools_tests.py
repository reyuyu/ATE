"""Focused path/download controls, separate from the frozen 101-test suite."""
import copy
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from attention_inheritance_remote import remap_sample
from handoff.restore_checkout import blob_sha
from handoff.restore_server_resources import retrieve
from handoff.verify_history_remote import resolve_entry


class RemoteControlsTests(unittest.TestCase):
    def test_runtime_remap_preserves_frozen_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b'original-image-test-fixture'
            (root / 'image_2324597.jpg').write_bytes(data)
            sample = dict(question_id='15133706', img_id='2324597', image_path='/old/image.jpg',
                          image_sha256=hashlib.sha256(data).hexdigest(), candidates=['left', 'right'])
            before = copy.deepcopy(sample)
            actual = remap_sample(sample, root)
            self.assertEqual(sample, before)
            self.assertEqual(actual['image_path'], str(root / 'image_2324597.jpg'))
            actual['candidates'].append('not-a-real-candidate')
            self.assertEqual(sample['candidates'], ['left', 'right'])

    def test_missing_or_changed_image_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = dict(img_id='001', image_sha256='0' * 64)
            with self.assertRaises(FileNotFoundError):
                remap_sample(sample, root)
            (root / 'image_001.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                remap_sample(sample, root)

    def test_existing_resource_requires_hash_before_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'weight'
            path.write_bytes(b'weight-bytes')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('handoff.restore_server_resources.get_hf_file_metadata') as request:
                result = retrieve('repo', 'weight', 'pin', root, 'https://example.invalid',
                                  path.stat().st_size, digest, transport='aria2')
                self.assertEqual(result, path)
                request.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'mismatched'):
                    retrieve('repo', 'weight', 'pin', root, 'https://example.invalid',
                             path.stat().st_size, '0' * 64, transport='aria2')
                self.assertEqual(path.read_bytes(), b'weight-bytes')

    def test_wrong_revision_never_starts_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('handoff.restore_server_resources.get_hf_file_metadata',
                       return_value=SimpleNamespace(commit_hash='wrong', size=10, etag='0' * 64)), \
                 patch('handoff.restore_server_resources.subprocess.run') as process:
                with self.assertRaisesRegex(ValueError, 'metadata disagrees'):
                    retrieve('repo', 'weight', 'pin', Path(directory), 'https://example.invalid',
                             10, '0' * 64, transport='aria2')
                process.assert_not_called()

    def test_wrong_size_or_hash_never_starts_transfer(self):
        for size, digest in [(11, '0' * 64), (10, '1' * 64)]:
            with self.subTest(size=size, digest=digest), tempfile.TemporaryDirectory() as directory:
                with patch('handoff.restore_server_resources.get_hf_file_metadata',
                           return_value=SimpleNamespace(commit_hash='pin', size=size, etag=digest)), \
                     patch('handoff.restore_server_resources.subprocess.run') as process:
                    with self.assertRaisesRegex(ValueError, 'metadata disagrees'):
                        retrieve('repo', 'weight', 'pin', Path(directory), 'https://example.invalid',
                                 10, '0' * 64, transport='aria2')
                    process.assert_not_called()

    def test_git_blob_hash_is_not_plain_file_hash(self):
        self.assertEqual(blob_sha(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
        self.assertNotEqual(blob_sha(b'test'), hashlib.sha1(b'test').hexdigest())

    def test_history_paths_only_remap_known_frozen_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual, is_image = resolve_entry('/old/GQA/image_001.jpg', root, {'image_001.jpg': 'hash'})
            self.assertTrue(is_image)
            self.assertEqual(actual, root / 'image_001.jpg')
            with self.assertRaisesRegex(ValueError, 'Unknown absolute path'):
                resolve_entry('/old/model.safetensors', root, {'image_001.jpg': 'hash'})
            with self.assertRaisesRegex(ValueError, 'escapes project'):
                resolve_entry('../../outside.json', root, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
