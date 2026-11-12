"""Read-only audit of the exact failed source and artifacts; CPU only."""
import pathlib,json,subprocess,csv,collections,ast
from .common import *
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 paths=['vlm_research/qwen3vl_c3v2_physical/'+n for n in ['engine.py','worker.py','tests.py']]
 names=['protocol.json','gpu_smoke_validation.json','physical_execution_audit.json','actual_matrix_shapes.csv','STOP.json','report_zh.md','source_manifest.json','execution_selection.json','model_manifest.json']
 names += [f'shard{r}/smoke_failure.json' for r in range(2)]
 names += [f'shard{r}/smoke/{ORIGINAL}__{c}.json' for r in range(2) for c in CONDITIONS]
 audit=[]
 for rel in paths:
  body=subprocess.check_output(['git','show',BASE+':'+rel],cwd=ROOT);assert sha(ROOT/rel)==__import__('hashlib').sha256(body).hexdigest()
  ast.parse(body);audit.append(dict(path=rel,sha256=sha(ROOT/rel),bytes=len(body),commit=BASE))
 for name in names:
  p=OLD/name;body=subprocess.check_output(['git','show',BASE+':vlm_research/results/qwen3vl_c3v2_physical_transfer/'+name],cwd=ROOT);assert p.read_bytes()==body
  audit.append(dict(path=str(p),sha256=sha(p),bytes=len(body),commit=BASE))
 write(OUT/'historical_bindings.json',audit)
 rows=[]
 for c in ['N','F','B','W22-functional','P','C','A16-functional']:
  physical=c in ['P','C'];skip=c in ['B','C'];bypass=c in ['F','B','C','A16-functional'];late=c in ['F','B','W22-functional','P','C'];native=c=='N'
  rows.append(dict(condition=c,late_R_image_mask=late,answer_image_mask=not native,physical_exit_L22=physical,attention_bypass_L1_L6=bypass,true_skip=skip,prefill_early_rows=122,prefill_attention_L1_L6_rows=121 if skip else 122,prefill_late_rows=26 if physical else 122,prefill_KV_return='current full KV except skipped attention prefix121' if not native else 'native',persistent_KV='text-only, skipped layers absent' if skip else 'text-only all layers' if physical else 'native dense all layers',decode_position='original logical 122+step-1',prefill_mrope='native3axes; original text indices gathered at exit' if physical else 'native3axes unchanged',deepstack='native injections after L0/L1/L2'))
 with (OUT/'execution_paths.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 observations=dict(
  baseline='No first-divergence claim can be made from saved scalar logits and operator shapes alone.',
  F_B='Functionally matched effective outputs: same permissions and bypass; physical B removes P-1 QKV and future unused KV. Prefill 122->121 attention can alter BF16 shapes.',
  W22_P='Despite W22 not being a pure Native-relative single factor, historical W22 and P DO have matching R and answer image permissions and no bypass; valid exit-isolation pair.',
  B_C='Same physical skip and image permissions; only exit and consequent late shapes differ. Both store text-only KV.',
  cache_prefill='CompactKV.update returns unmodified fresh KV for current prefill, while storing text subset for future decode. Thus persistent storage alone should not change prefill attention values.',
  skip_KV='No future attention queries exist in L1-L6. Prefix queries cannot causally use P-1. Absent persistent cache there is semantically sufficient.',
  positions='Original logical cache positions and gathered original rotary factors are preserved by source inspection; require runtime tensor equality to exclude bugs.',
  mask='Dense finite-min future mask versus explicit -inf image mask; physical text submatrix should preserve all causal/legal entries. CPU truth-table tests and runtime alignment required.',
  deepstack='Three native injections before L22; no injection intentionally removed.',
  first_possible_shape_divergence='L1 attention prefix 121 vs full122; later L22 text-only26 vs full122 affects QKV, attention reductions, MLP. This is a hypothesis, not a measured cause.',
  old_probe_limit='Original array vectors were not persisted; this task must persist complete logits and relevant stage tensors with SHA.',
  diagnostics_scope='No quality, NLL, benchmark or layer search in this task.')
 write(OUT/'cpu_code_audit.json',observations)
 selection=read(OLD/'execution_selection.json');b=read(OLD/'input_bindings.json')
 write(OUT/'sample_manifest.json',dict(samples=[sample(k) for k in selection['smoke']],input_bindings={k:b[k] for k in selection['smoke']},original=ORIGINAL,secondary_diagnostic=selection['smoke'][1],selection='Exactly historical min/median/max input smoke; no replacements'))
 write(OUT/'protocol.json',dict(base_commit=BASE,old_execution_commit=FROZEN,model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',dtype='bfloat16',transformers='4.57.6',attention='eager',max_new_tokens=32,conditions=CONDITIONS,matched_pairs=PAIRS,original_case=ORIGINAL,secondary_case=selection['smoke'][1],smoke_cases=selection['smoke'],probe_layers=list(range(36)),raw_attention_heads=[0,16,31],raw_attention_queries='retained original text positions, with P-1 separately labelled; all heads summarized, raw selected heads only',probes=['layer_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','rope_q','rope_k','scores_before_mask','effective_mask','scores_after_mask','attention_probabilities','attention_AV_output','o_proj_raw','attention_effective_output','attention_residual','mlp_output','layer_output','final_norm','lm_head_input'],numeric_metrics=['max_abs','mean_abs','RMSE','relative_L2 eps1e-12','Top1','Top20 union logits','all generated tokens'],first_divergence='First layer then execution-order operator with any nonzero aligned text error; also report all common positions before exit and P-1 separately',amplification='First mean absolute aligned-text error > 1e-3, and full per-layer values; diagnostic marker does not replace acceptance',tolerances=read(OLD/'protocol.json')['smoke_tolerance'],probe_gate='Every effective logit vector with/without probes torch.equal; invalid probes stop diagnostic analysis',factor_gate='Do all four frozen diagnostic comparisons even when an old numeric threshold fails; do not stop first comparison',shadow=dict(layers=SHADOW_LAYERS,ops=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],inputs='Frozen F operator input, compare full122 vs prefix121 at L1-L6 attention ops; full122 vs original text26 at L22-L23; identical shared input rows',precision='BF16 unchanged; local FP32 operator diagnostic permitted, not a candidate replacement'),repair_policy='Only after first divergence located; no threshold/layer/prompt/dtype/weight changes; no nearly-dense fallback accepted as physical fix',gpu_authorization='PENDING_CURRENT_USER_REPLY',gpu_budget='Freeze absolute start and +2h deadline before first GPU; new GPU work stops 30min before; never reset on retries',stop='No formal accuracy or latency benchmark under any outcome; any repair-original-case failure stops expanding samples',success_status='NUMERICAL_SMOKE_VALIDATED only if original then all fixed cases pass on authorized cards',raw_tensor_storage='Private /data experiment resource directory; per-file SHA, shape and dtype indexed; do not claim Git contains external arrays'))
 print('CPU read-only source and artifact audit passed',len(audit),'bindings; no GPU work',flush=True)
if __name__=='__main__':main()
