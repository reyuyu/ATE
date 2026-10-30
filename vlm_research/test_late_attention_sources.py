import copy
import unittest
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
import transformers.models.llama.modeling_llama as llama
from benchmarks.smoke import hook_footprint
from paper_rq1_late_attention_sources.engine import partition,validate_partition,group_head,contributions,SourceTrace,step,step_inputs,cache_fingerprint
from paper_rq1_late_attention_sources.protocol import LAYERS,conditions,FAMILY


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config=LlamaConfig(hidden_size=16,intermediate_size=24,num_attention_heads=2,num_key_value_heads=1,
            num_hidden_layers=32,vocab_size=64,max_position_embeddings=1024,attention_dropout=0.)
        config._attn_implementation='eager'
        self.language_model=LlamaForCausalLM(config).half().eval()
    def forward(self,**kwargs):return self.language_model(**kwargs)


def prompt():
    ids=[1,3,4]+[12]*576+[5,6,7]
    return dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))


class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(9)

    def test_dynamic_partition_and_consumed_token(self):
        p=prompt();g=partition(p,3,image_token_id=12)
        self.assertEqual(g['BOS'],[0]);self.assertEqual(g['IMAGE'],list(range(3,579)))
        self.assertEqual(g['PROMPT_TEXT'],[1,2,579,580,581])
        self.assertEqual(g['GENERATED_HISTORY'],[582,583,584]);self.assertEqual(g['RECENT'],[584])
        self.assertEqual(g['OLDER_HISTORY'],[582,583]);validate_partition(g,585)

    def test_partition_rejects_padding_overlap_missing(self):
        p=prompt();g=partition(p,1,image_token_id=12)
        for bad in ['overlap','missing','wrong_recent']:
            x=copy.deepcopy(g)
            if bad=='overlap':x['BOS'].append(1)
            elif bad=='missing':x['PROMPT_TEXT'].pop()
            else:x['RECENT']=[0]
            with self.assertRaises(ValueError):validate_partition(x,583)
        p['attention_mask'][0,2]=0
        with self.assertRaises(ValueError):partition(p,1,image_token_id=12)
        with self.assertRaises(ValueError):partition(prompt(),0,image_token_id=12)

    def test_four_groups_bias_and_split_rounding_bounds(self):
        g=partition(prompt(),4,image_token_id=12);n=586
        p=torch.randn(2,n).softmax(-1).half();v=torch.randn(2,n,4).half()
        projection=torch.nn.Linear(8,8,bias=True).half()
        native=(p.unsqueeze(-2)@v).squeeze(-2).reshape(1,1,8)
        output=projection(native)
        vectors,rows,audit=contributions(projection,p,v,g,native,output,torch.randn(8),26)
        self.assertEqual(len(rows),6);self.assertLessEqual(audit['group_projection_max_error_to_bound'],1)
        self.assertLessEqual(audit['history_split_max_error_to_bound'],1)
        self.assertAlmostEqual(sum(r['mass'] for r in rows[:4]),1,places=3)
        expected=sum(torch.nn.functional.linear(group_head(p,v,g[name]),projection.weight.float(),None) for name in ['BOS','IMAGE','PROMPT_TEXT','GENERATED_HISTORY'])
        torch.testing.assert_close(sum(vectors[name] for name in ['BOS','IMAGE','PROMPT_TEXT','GENERATED_HISTORY']),expected)
        with torch.no_grad():projection.bias.add_(1)
        vectors2,_,_=contributions(projection,p,v,g,native,projection(native),torch.randn(8),26)
        for name in vectors:self.assertTrue(torch.equal(vectors[name],vectors2[name]))

    def test_empty_older_is_exact_zero(self):
        g=partition(prompt(),1,image_token_id=12)
        p=torch.randn(2,583).softmax(-1).half();v=torch.randn(2,583,4).half()
        self.assertTrue(torch.equal(group_head(p,v,g['OLDER_HISTORY']),torch.zeros(1,1,8)))
        projection=torch.nn.Linear(8,8,bias=False).half()
        native=(p.unsqueeze(-2)@v).squeeze(-2).reshape(1,1,8)
        vec,rows,_=contributions(projection,p,v,g,native,projection(native),torch.randn(8),31)
        self.assertTrue(torch.equal(vec['RECENT'],vec['GENERATED_HISTORY']))
        older=next(r for r in rows if r['source']=='OLDER_HISTORY')
        self.assertEqual(older['mass'],0);self.assertIsNone(older['cosine_with_full_attention'])

    def test_noop_observation_grouped_kv_and_cache(self):
        model=Toy();p=prompt();_,history,_=step(model,p,None)
        inputs=step_inputs(p,[8]);g=partition(p,1,image_token_id=12)
        before=cache_fingerprint(history);hooks=hook_footprint(model);original=llama.eager_attention_forward
        full,_,_=step(model,inputs,history)
        observer,_,trace=step(model,inputs,history,g,observe=True)
        self.assertTrue(torch.equal(full,observer));self.assertEqual(len(trace['rows']),44)
        for mode in ['Whole','Image','Prompt','History','BOS','Recent','Older']:
            same,_,record=step(model,inputs,history,g,mode,26,noop=True)
            self.assertTrue(torch.equal(full,same));self.assertEqual([r['layer'] for r in record['edits']],[26])
        same,_,_=step(model,inputs,history,g,'Older',26)
        self.assertTrue(torch.equal(full,same))
        self.assertEqual(before,cache_fingerprint(history));self.assertEqual(hooks,hook_footprint(model))
        self.assertIs(original,llama.eager_attention_forward)

    def test_source_subtraction_retains_denominator_and_bias(self):
        model=Toy();g=partition(prompt(),3,image_token_id=12)
        projection=model.language_model.model.layers[26].self_attn.o_proj
        p=torch.randn(2,585).softmax(-1).half();v=torch.randn(2,585,8).half()
        native=(p.unsqueeze(-2)@v).squeeze(-2).reshape(1,1,16)
        full=projection(native);p_before=p.clone();v_before=v.clone()
        for mode,name in [('Image','IMAGE'),('Prompt','PROMPT_TEXT'),('History','GENERATED_HISTORY'),('BOS','BOS'),('Recent','RECENT'),('Older','OLDER_HISTORY')]:
            with SourceTrace(model,g,mode,26) as trace:
                trace.pv[26]=p,v
                changed=projection(native)
            expected=(full.float()-torch.nn.functional.linear(group_head(p,v,g[name]),projection.weight.float(),None)).half()
            self.assertTrue(torch.equal(changed,expected))
        self.assertTrue(torch.equal(p,p_before));self.assertTrue(torch.equal(v,v_before))

    def test_exception_cleanup_and_prefill_rejected(self):
        model=Toy();g=partition(prompt(),1,image_token_id=12)
        original=llama.eager_attention_forward;hooks=hook_footprint(model)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with SourceTrace(model,g,observe=True):raise RuntimeError('fixture')
        self.assertIs(original,llama.eager_attention_forward);self.assertEqual(hooks,hook_footprint(model))
        with self.assertRaises(ValueError):step(model,prompt(),None,g,'Whole',26)

    def test_conditions_and_budget_no_extra_layers(self):
        self.assertEqual(LAYERS,[3,14,16,25,26,27,28,29,30,31]);self.assertEqual(FAMILY,41)
        self.assertEqual(sum(len(conditions(l)) for l in LAYERS),46)
        with self.assertRaises(ValueError):SourceTrace(Toy(),{},'BOS',14)
        with self.assertRaises(ValueError):SourceTrace(Toy(),{},'Image',10)


if __name__=='__main__':unittest.main()
