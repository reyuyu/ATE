from pathlib import Path
import json,hashlib,subprocess

r=Path('/data/lk/ATE-worktrees/layer-semantic-joint-writing-240')
p=r/'vlm_research/results/layer_semantic_joint_writing_240'
def read(n):return json.loads((p/n).read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def write(n,v):
    target=p/n;assert not target.exists();target.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
c=read('completion.json');assert c['status']=='COMPLETE'
for n,h in c['artifacts'].items():assert sha(p/n)==h,n
scientific={n:h for n,h in c['artifacts'].items() if n.endswith(('.csv','.npz')) or n.startswith(('checkpoints/','reused_checkpoints/')) or n in ['protocol.json','validation.json','token_group_manifest.json','sample_manifest.json','input_mappings.json']}
before=sha(p/'report_zh.md');text=(p/'report_zh.md').read_text(encoding='utf-8');assert '## 完成后的研究判读' not in text
interpretation='''## 完成后的研究判读

**Layer Sparsity与Token Sparsity在本批16层组合的点估计上可以同时成立，但更激进的12层或8层组合未达到预设质量界限，不能从两种单独稀疏条件的良好净表现推出联合安全。** Joint-16为201/240（83.750%），相对All-Writer的202/240为−0.417pp，达到探索性−1pp和−2pp点估计界限；区间[−4.167,+3.333]仍允许较大总体损失，不构成非劣效证明。

**“约25.78%的关键R Token在12/32层看图、约9.67%预算即可接近All”的设想，本轮没有达到预设−2pp点估计要求。** Joint-12为195/240（81.250%），较All低2.917pp，95%CI[−7.500,+1.250]；18题原对变错、11题原错变对。对应Layer-only为201/240，Semantic-only为202/240，联合分别少答对6题和7题。交互为−2.500pp，CI[−6.667,+1.667]。存在5题All、Layer、Semantic全部正确而Joint错误；若仅要求两单独条件正确，则有7题联合错误。这些逐题反例表明两种稀疏性不能无条件组合，但交互CI跨0，尚不能确证总体负交互。

Joint-16相对Semantic-only只改变3题完整答案的首Token，损害2题、纠正1题，差值−0.417pp，CI[−2.083,+0.833]；相对Layer-only为+0.417pp，CI[−3.333,+4.167]。其交互为+0.417pp，CI[−1.250,+2.083]，本样本联合独有损害为0；这既不是普遍零损害保证，也不能忽略其相对All仍有11题损害和10题纠正。历史Semantic-only本身同样有10题损害、10题纠正，净准确率相同不代表答案相同。

Joint-8为194/240（80.833%），相对All为−3.333pp，CI[−7.917,+1.250]，同样未达到−2pp界限；联合独有损害3题，两单独均正确而联合错误4题。交互为−1.667pp，CI[−5.833,+2.500]。三个交互区间均跨0，因此本轮既不证明可加，也不证明不存在额外损害。

任务层面，QueryAttr由All的31/40降到Joint-12的28/40和Joint-8的26/40，即−7.5pp和−12.5pp；这是每任务40题的描述性差异，不能作为新的分层选择规则。例如09502671在All、Layer-only、Semantic-only均答Surfing，Joint-12改为Skating；02420578三组对照均答Right，Joint-12改为Left。所有反例均保留，不筛题、不调整层集合或Token集合。

预算需要区分平均方式：25.7755%是每题Semantic-k/R比例的宏平均，乘12/32得到9.6658%；按240题的实际R→Image连接数汇总，Joint-12是9.9297%。Joint-16对应宏平均12.8878%、汇总13.2396%，是本轮更值得留待独立数据验证的联合候选。所有值只描述读图权限预算；全部Attention矩阵、Image KV和层计算仍然存在，未实现实际FLOPs、延迟或显存节省。

所有条件均已屏蔽P−1和Decode对原始图像的直接访问，因此Joint-16保留的较高性能支持“少量文本Query在部分层获得直接视觉访问后，非图像中继可支持后续作答”的功能证据。未选中R仍可通过其他文本连接间接获取视觉信息；本轮不证明视觉写入只发生在被选中的Token或层，也不直接测量信息存储位置。没有追加独立数据、激活修补或优化实现。

'''
text=text.replace('## 冻结协议与历史复用',interpretation+'## 冻结协议与历史复用',1)
(p/'report_zh.md').write_text(text,encoding='utf-8')
gpu=read('gpu_completion.json');pre=read('pre_freeze_packaging_note.json');cpu=read('cpu_validation.json')
notes=f'''# Execution notes

The user authorized exactly three new conditions (720 main generations). Five historical controls (1200 records) were audited and copied byte-for-byte, with provenance retained. No new token selection was performed.

Execution source commit: {c['code_commit']}. Final CPU validation: {cpu['tests']} tests passed. Three fixed real-7B smoke questions passed, including 54 smoke generations. Every effective HF/native/no-op logit and old/new baseline logit remained strictly equal; every prohibited actual attention probability was exactly zero. No precision, data, threshold, layer set or token policy was changed.

Only physical GPU1, UUID GPU-b08489af-0a8a-eb48-55ba-8346733af417, was exposed. It was idle at the recorded immediate preflight; GPU0 had become occupied before selection and was left untouched. One model replica generated conditions serially. The worker PID was {gpu['pid']}; load, smoke and main collection took {gpu['seconds']:.4f} seconds. Peak allocated memory was {gpu['memory']['peak_allocated_mib']:.3f} MiB. The worker exited normally; GPU1 returned to 27 MiB and 0% use. See gpu_release.json for the timestamped snapshot.

Before freezing the source or loading a GPU model, a CRLF formatting issue prevented the source commit. An incorrectly sequenced freeze invocation then rejected the uncommitted source as intended. Normalizing only CRLF to LF preserved the AST. The immutable JSON writer also rejected replacing an earlier passed CPU audit; that audit was archived under a separate name, and all 68 tests and the historical audit were rerun successfully. These packaging diagnostics and both earlier audit records are retained under pre_freeze* and failures/. No scientific CPU test assertion, numerical GPU gate, mask check, cache check or formal generation failed. The frozen execution source was never subsequently edited.

The historical semantic module was imported verbatim from its referenced Git commit; its pre-existing trailing whitespace was preserved to retain exact source identity. New execution sources passed the Git whitespace check after LF normalization.

All 720 new answers ended with EOS, with no empty answer, truncation or repetition marker. CPU analysis replayed 1920 checkpoints, nine paired comparisons, three predeclared four-condition interactions and actual connection budgets; it rechecked full model-file SHA256 hashes and exact CSV cells. An independent implementation regenerated all bootstrap draws and verified all intervals, harm/correction counts, budget ratios, frozen token sets and complete negative cases. The three PNG figures were visually inspected; matching PDF exports are included.

The final editorial step adds interpretation and these notes only. Scientific CSV files, checkpoints, raw arrays, input manifests, validation records and execution sources remain unchanged. Raw first-token arrays stay in the server resource directory and are not included on GitHub. No additional experiment, physical image exit, KV pruning, router, kernel or cross-model work was started.
'''
assert not (p/'execution_notes.md').exists();(p/'execution_notes.md').write_text(notes,encoding='utf-8')
for n,h in scientific.items():assert sha(p/n)==h,n
assert not subprocess.check_output(['git','diff',c['code_commit'],'--','vlm_research/layer_semantic_joint_writing','vlm_research/semantic_visual_writers'],cwd=r,text=True).strip()
write('delivery_editorial_audit.json',dict(passed=True,report_before_sha256=before,report_after_sha256=sha(p/'report_zh.md'),scientific_artifacts_unchanged=scientific,
    frozen_source_unchanged=True,independent_statistics_passed=True,three_pngs_visually_verified=True,scope='Chinese interpretation and execution notes only'))
c['artifacts']={str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ['worker.lock','completion.json']}
(p/'completion.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('EDITORIAL PASS: scientific records and frozen source unchanged; explicit Joint-12 negative finding recorded')
