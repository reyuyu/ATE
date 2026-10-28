"""Focused path/download controls, separate from the frozen 101-test suite."""
import copy
import hashlib
import io
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from attention_inheritance_remote import remap_sample
from handoff.restore_checkout import blob_sha
from handoff.restore_server_resources import (
    ARCHIVE, ARCHIVE_BYTES, ARCHIVE_SHA256, DATA_ID, DATA_REVISION,
    MODEL_ID, MODEL_REVISION, MODELSCOPE_PINS, modelscope_source, retrieve,
)
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

    def test_modelscope_copies_require_exact_metadata(self):
        cases = [(DATA_ID, ARCHIVE, DATA_REVISION, ARCHIVE_BYTES, ARCHIVE_SHA256, 'dataset'),
                 (MODEL_ID, 'model-00001-of-00003.safetensors', MODEL_REVISION, 10, '0' * 64, None)]
        for repo, filename, revision, size, digest, kind in cases:
            with self.subTest(repo=repo):
                pin = MODELSCOPE_PINS[(kind or 'model', repo, revision)]
                item = dict(Path=filename, Revision=pin, Size=size, Sha256=digest)
                payload = dict(Code=200, Data=dict(Files=[item]))
                with patch('handoff.restore_server_resources.urlopen',
                           return_value=io.BytesIO(json.dumps(payload).encode())) as request:
                    source = modelscope_source(repo, filename, revision, size, digest, kind)
                self.assertEqual(source['canonical_revision'], revision)
                self.assertEqual(source['provider_revision'], pin)
                self.assertIn('/resolve/' + pin + '/', source['url'])
                self.assertIn('Revision=' + pin, request.call_args.args[0])

    def test_modelscope_bad_metadata_never_starts_download(self):
        item = dict(Path=ARCHIVE, Revision=MODELSCOPE_PINS[('dataset', DATA_ID, DATA_REVISION)],
                    Size=ARCHIVE_BYTES, Sha256=ARCHIVE_SHA256)
        variants = [dict(item, Size=1), dict(item, Sha256='0' * 64),
                    dict(item, Revision='master')]
        listings = [dict(Code=200, Data=dict(Files=[entry])) for entry in variants]
        listings.extend([dict(Code=200, Data=dict(Files=[item, item])),
                         dict(Code=200, Data=dict(Files=[])), dict(Code=500)])
        for listing in listings:
            with self.subTest(listing=listing), tempfile.TemporaryDirectory() as directory:
                with patch('handoff.restore_server_resources.urlopen',
                           return_value=io.BytesIO(json.dumps(listing).encode())), \
                     patch('handoff.restore_server_resources.subprocess.run') as process:
                    with self.assertRaisesRegex(ValueError, 'ModelScope metadata'):
                        retrieve(DATA_ID, ARCHIVE, DATA_REVISION, Path(directory), '',
                                 ARCHIVE_BYTES, ARCHIVE_SHA256, 'dataset', 'modelscope')
                    process.assert_not_called()

    def test_modelscope_unknown_pin_never_contacts_network(self):
        with patch('handoff.restore_server_resources.urlopen') as request:
            for revision, digest in [('master', ARCHIVE_SHA256), (DATA_REVISION, None)]:
                with self.assertRaisesRegex(ValueError, 'No hash-pinned'):
                    modelscope_source(DATA_ID, ARCHIVE, revision, ARCHIVE_BYTES, digest, 'dataset')
            request.assert_not_called()

    def test_modelscope_reuses_control_and_checks_final_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = root / 'weight.transfer'
            transfer.write_bytes(b'partial')
            control = root / 'weight.transfer.aria2'
            control.write_bytes(b'preserved-range-map')
            content = b'complete-original-weight'
            digest = hashlib.sha256(content).hexdigest()
            source = dict(url='https://modelscope.cn/test-pinned-url')
            def complete(command, **kwargs):
                self.assertTrue(kwargs['check'])
                self.assertIn('--continue=true', command)
                self.assertIn('--out=weight.transfer', command)
                self.assertEqual(command[-1], source['url'])
                self.assertEqual(control.read_bytes(), b'preserved-range-map')
                transfer.write_bytes(content)
            with patch('handoff.restore_server_resources.modelscope_source', return_value=source), \
                 patch('handoff.restore_server_resources.subprocess.run', side_effect=complete), \
                 patch('handoff.restore_server_resources.get_hf_file_metadata') as hf:
                result = retrieve('repo', 'weight', 'pin', root, '', len(content), digest,
                                  transport='modelscope')
            hf.assert_not_called()
            self.assertEqual(result.read_bytes(), content)
            audit = json.loads((root / '.download-sources/weight.modelscope.json').read_text())
            self.assertTrue(audit['resume_control_present'])
            self.assertEqual(audit['resume_control_sha256'], hashlib.sha256(control.read_bytes()).hexdigest())

    def test_modelscope_bad_content_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = root / 'weight.transfer'
            transfer.write_bytes(b'bad')
            with patch('handoff.restore_server_resources.modelscope_source',
                       return_value=dict(url='https://modelscope.cn/test')), \
                 patch('handoff.restore_server_resources.subprocess.run'):
                with self.assertRaisesRegex(ValueError, 'final size/SHA256'):
                    retrieve('repo', 'weight', 'pin', root, '', 3, '0' * 64, transport='modelscope')
            self.assertFalse((root / 'weight').exists())
            self.assertEqual(transfer.read_bytes(), b'bad')

    def test_modelscope_failed_process_retains_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = root / 'weight.transfer'
            transfer.write_bytes(b'partial')
            with patch('handoff.restore_server_resources.modelscope_source',
                       return_value=dict(url='https://modelscope.cn/test')), \
                 patch('handoff.restore_server_resources.subprocess.run',
                       side_effect=subprocess.CalledProcessError(1, ['aria2c'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    retrieve('repo', 'weight', 'pin', root, '', 10, '0' * 64, transport='modelscope')
            self.assertFalse((root / 'weight').exists())
            self.assertEqual(transfer.read_bytes(), b'partial')

    def test_modelscope_keeps_small_files_in_original_hf_cache(self):
        with patch('handoff.restore_server_resources.hf_hub_download', return_value='/cached/config.json') as hf, \
             patch('handoff.restore_server_resources.modelscope_source') as alternative:
            result = retrieve(MODEL_ID, 'config.json', MODEL_REVISION, Path('/cached'),
                              'https://hf-mirror.com', 950, None, transport='modelscope')
            self.assertEqual(result, Path('/cached/config.json'))
            self.assertTrue(hf.call_args.kwargs['local_files_only'])
            self.assertEqual(hf.call_args.kwargs['revision'], MODEL_REVISION)
            alternative.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
