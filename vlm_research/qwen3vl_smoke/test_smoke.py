import tempfile
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from common import question_positions, sha256, write


class HelpersTest(unittest.TestCase):
    def test_question_mapping_with_different_template_lengths(self):
        for prefix in ("abc", "a" * 77):
            text = prefix + "Question?" + "\nAnswer shortly."
            offsets = [(i, i + 1) for i in range(len(text))]
            self.assertEqual(question_positions(offsets, text, "Question?"), list(range(len(prefix), len(prefix) + 9)))

    def test_question_not_duplicate(self):
        with self.assertRaises(AssertionError):
            question_positions([(0, 1)], "Q Q", "Q")

    def test_atomic_json_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            write(path, {"x": 1})
            self.assertEqual(len(sha256(path)), 64)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_unicode_tokens_in_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode.json"
            env = dict(os.environ, LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", CUDA_VISIBLE_DEVICES="")
            subprocess.run([sys.executable, "-c", "from common import write; import sys; write(sys.argv[1], {'token': chr(266)})", str(path)],
                           cwd=Path(__file__).parent, env=env, check=True)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["token"], "\u010a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
