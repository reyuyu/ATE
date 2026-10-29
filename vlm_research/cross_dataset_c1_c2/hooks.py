"""Extend the unchanged prefill lifecycle with original C1 o_proj zero_query."""
from benchmark_interventions.hooks import PrefillIntervention
from knockout_probe import model_layers


class AttentionPrefill(PrefillIntervention):
    def __init__(self, model, inputs, mapping, originals, noop=False):
        super().__init__(model, inputs, mapping, 'Full', originals, noop=noop)
        self.condition = 'C1'

    def __enter__(self):
        try:
            super().__enter__()
            for index in (2, 3, 4):
                module = model_layers(self.model)[index].self_attn.o_proj
                self.handles.append(module.register_forward_hook(self.zero_attention(index)))
            return self
        except BaseException:
            self.cleanup()
            raise

    def zero_attention(self, index):
        def hook(module, args, out):
            if self.noop:
                return out
            result, audit = self.originals['c2'].zero_query(out, self.p)
            self.records.append(dict(layer=index, kind='attention_after_o_proj', **audit))
            return result
        return hook

    def finish(self, module, args, out):
        if not self.noop and [r['layer'] for r in self.records] != [2, 3, 4]:
            raise AssertionError('C1 must edit exactly L2/L3/L4 o_proj at original P-1')
        return super().finish(module, args, out)
