"""CPU-only query selection, no-op, cleanup, frozen sampling and score checks."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from .experiment import AttentionBypass, CONDITIONS, OUT, gqa_prediction, hook_counts, select, write


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"o_proj": nn.Identity()}) for _ in range(36)])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=[
            SimpleNamespace(self_attn=SimpleNamespace(o_proj=x["o_proj"])) for x in self.layers]))

    def forward(self, input_ids=None, inputs_embeds=None, cache_position=None):
        result = inputs_embeds.clone() if inputs_embeds is not None else input_ids.float().unsqueeze(-1)
        outputs = []
        for layer in self.layers:
            outputs.append(layer["o_proj"](result + 1))
        return outputs


class Checks(unittest.TestCase):
    def test_fixed_conditions(self):
        self.assertEqual(len(CONDITIONS), 7)
        self.assertEqual(CONDITIONS["W2"], [2, 3, 4, 5])
        self.assertEqual(CONDITIONS["W5"], [16, 17, 18, 19])

    def test_prefill_locality(self):
        model = Toy()
        values = torch.randn(1, 9, 7)
        for name, layers in CONDITIONS.items():
            before = hook_counts(model)
            with AttentionBypass(model, layers, 9):
                result = model(inputs_embeds=values, cache_position=torch.arange(9))
            for i, output in enumerate(result):
                self.assertTrue(torch.equal(output[:, :-1], (values+1)[:, :-1]))
                self.assertTrue(torch.equal(output[:, -1], torch.zeros_like(values[:, -1]) if i in layers else (values+1)[:, -1]))
            self.assertEqual(hook_counts(model), before)

    def test_all_decode_queries(self):
        model = Toy()
        with AttentionBypass(model, CONDITIONS["W0"], 5) as hook:
            model(input_ids=torch.ones(1, 5), cache_position=torch.arange(5))
            for pos in range(5, 11):
                result = model(input_ids=torch.ones(1, 1), cache_position=torch.tensor([pos]))
                self.assertTrue(all(torch.count_nonzero(result[i]) == 0 for i in range(4)))
            self.assertEqual(len(hook.events), 7*4)
            self.assertEqual([x["phase"] for x in hook.calls], ["prefill"]+["decode"]*6)

    def test_noop_exact(self):
        model = Toy()
        ids = torch.randn(1, 8)
        native = model(input_ids=ids, cache_position=torch.arange(8))
        with AttentionBypass(model, CONDITIONS["W5"], 8, noop=True):
            noop = model(input_ids=ids, cache_position=torch.arange(8))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(native, noop)))

    def test_bad_cache_stops_and_cleans(self):
        model = Toy()
        before = hook_counts(model)
        with self.assertRaises(AssertionError):
            with AttentionBypass(model, [0], 5):
                model(input_ids=torch.ones(1, 1), cache_position=torch.tensor([5]))
        self.assertEqual(hook_counts(model), before)

    def test_exception_cleanup(self):
        model = Toy()
        before = hook_counts(model)
        with self.assertRaises(RuntimeError):
            with AttentionBypass(model, [0], 5):
                raise RuntimeError("fixture")
        self.assertEqual(hook_counts(model), before)

    def test_sampling_independent_of_output(self):
        rows = [dict(task=t, question_id=str(i), image_sha256=f"{t}-{i}", correct=i % 2)
                for t in "abcdef" for i in range(50)]
        chosen = select(rows)
        changed = [dict(r, correct=1-r["correct"]) for r in reversed(rows)]
        self.assertEqual([(r["task"], r["question_id"]) for r in chosen],
                         [(r["task"], r["question_id"]) for r in select(changed)])
        self.assertEqual(len(chosen), 240)
        self.assertEqual(len({r["image_sha256"] for r in chosen}), 240)

    def test_existing_scoring_not_permissive(self):
        self.assertEqual(gqa_prediction("Dog."), "dog")
        self.assertEqual(gqa_prediction("two cats"), "two cats")
        self.assertNotEqual(gqa_prediction("The answer is red."), "red")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(OUT / "cpu_tests.json", dict(passed=result.wasSuccessful(), tests=result.testsRun,
          failures=len(result.failures), errors=len(result.errors), device="CPU only"))
    raise SystemExit(not result.wasSuccessful())
