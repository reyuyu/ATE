import collections,shutil
from transformers import AutoTokenizer
from .common import *
RULES=dict(condition='Native Full, all original attention edges open. All-Writer only for two predeclared smoke controls; never a Semantic hard-mask observation.',
    samples='Exactly the16 original selected_cases.csv rows,12 harm and4 rescue; unchanged order; ranks alternate row index,8 each.',
    units='R strictly last original Image position+1 through P-2. P-1 separate, excluded from all R group ranks.',
    groups='Reuse original frozen token manifests and whole-word groups. No regrouping, no new token selection.',
    categories='Question=frozen group.question. Instruction=group offsets overlap the literal instruction substring after Question in prompt_text. All other R groups=Assistant/template, including boundary whitespace. P-1 separate.',
    primary_rank='Within each case, band and scope: descending arithmetic mean ImageMass over all layers in band, all32 heads, and all subwords of the intact group; ties by first position then group_id. Never rank by a single-head maximum.',
    secondary='Also provide excess/enrichment means and subword mean/max, head mean/median/max. Undefined enrichment remains blank with explicit reason; no infinity or clipping.',
    orange='Top5 omitted Question whole groups by primary L0-15 rank within each case; relative top5, not an absolute high/causal/enrichment>1 threshold.',
    figure_A='Case-level mean over layer/head/subword for four available scopes, plus paired layer-banded descriptions; absent categories missing, not zero. Descriptive16-case distribution only.',
    figure_C='Per-case/per-band/per-scope top5, then case-kind descriptive summaries. No positional alignment across cases; no inferential claims from token/head pseudo-replication.',
    tags='Pre-existing fixed text-only spaCy3.8.7/en_core_web_sm3.8.0 parse from noun-article commit11fe70e82d6b608da21d9bd375ce49abf390d4d5 supplies noun annotation only; immutable word groups remain those of Semantic experiment. Literal articles; frozen relation-word set; last2 Question groups tagged question_end, nonexclusive.',
    smoke='First case in each rank: Native plain vs no-op capture logits torch.equal; Native captured vs no-op captured attention exact; original All-Writer R->Image probability tensors torch.equal all32layers/allheads/allR/all576keys.',
    forward_budget='16 Native Full cases plus6 validation-only Prefills (plain/no-op/All-Writer on2cases), total22 single-Prefill forwards. No decode, no free generation, no new accuracy scoring.',
    causal_boundary='High attention mass is allocation, not causal importance. Under answer-stage image blocking, L31 R outputs cannot propagate to first-answer P-1 logits.')
REL=set('of to in on at under above below behind beside between near next from with by through left right front'.split())
def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'ranking_rules.json',RULES)
    selected=HISTORY/'selected_cases.csv';assert sha256(selected)==gitsha(selected)
    order=list(csv.DictReader(selected.open()));old=historical('case_manifest.json')['cases'];mapping={v['question_id']:v for v in old};assert len(order)==16
    # Extra annotation is frozen pre-attention, never used for ranking or selection.
    pospath=Path('/data/lk/ATE-worktrees/noun-article-visual-writers-pilot/vlm_research/results/noun_article_visual_writers_pilot/pos_parse_all3000.json')
    expected=hashlib.sha256(subprocess.check_output(['git','show','11fe70e82d6b608da21d9bd375ce49abf390d4d5:vlm_research/results/noun_article_visual_writers_pilot/pos_parse_all3000.json'],cwd=pospath.parents[3])).hexdigest();assert sha256(pospath)==expected
    hashpath=ROOT/'results/gqa3000_token_visual_writing/input_hashes.json';assert sha256(hashpath)==gitsha(hashpath,SEM);inputs_hashes=read(hashpath)
    pos=read(pospath);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);out=[];images=set()
    for i,row in enumerate(order):
        q=row['question_id'];v=mapping[q];s=v['sample'];spec=v['spec'];assert row['kind']==v['kind'] and row['image_sha256']==s['image_sha256'];assert s['image_sha256'] not in images;images.add(s['image_sha256'])
        assert row['All-Writer_answer']==v['rows']['All-Writer']['text'] and row['Semantic-k_answer']==v['rows']['Semantic-k']['text'];assert sha256(s['image_path'])==s['image_sha256']
        orig=ROOT/'results/gqa3000_token_visual_writing/token_manifests'/(s['task']+'__'+q+'.json');assert sha256(orig)==gitsha(orig,SEM) and read(orig)==spec
        enc=tok(v['prompt_expanded'],return_offsets_mapping=True);assert enc['input_ids']==s['input_mapping']['prompt_ids'];assert all(list(enc['offset_mapping'][t['position']])==t['offset'] for t in v['tokens'])
        m=s['input_mapping'];R=list(range(m['visual_positions'][-1]+1,m['p_minus_one']));assert spec['relay_positions']==R and flatten(spec['groups'])==R and intact(spec['writers']['Semantic-k'],spec['groups'])
        instruction=s['prompt_text'][len(s['question']):].strip();istart=v['prompt_expanded'].index(instruction,v['question_char_offset'][1]);iend=istart+len(instruction)
        tokens={t['position']:t for t in v['tokens']};groups=[];qgroups=[g for g in spec['groups'] if g['question']];qend={p for g in qgroups[-2:] for p in g['positions']};parse=pos[q];assert parse['question']==s['question']
        for gi,g in enumerate(spec['groups']):
            p=g['positions'];offsets=[tokens[k]['offset'] for k in p];scope='Question' if g['question'] else 'Instruction' if any(b>istart and a<iend for a,b in offsets) else 'Assistant/template';S=set(p)<=set(spec['writers']['Semantic-k'])
            tag=[];words={w.lower() for w in g['words']}
            if words&{'a','an','the'}:tag.append('article')
            if words&REL:tag.append('relation')
            if set(p)&qend:tag.append('question_end')
            if not words and g['decoded'].strip():tag.append('punctuation')
            qs=v['question_char_offset'][0]
            if g['question'] and any(t['noun'] and any(b>qs+t['start'] and a<qs+t['end'] for a,b in offsets) for t in parse['tokens']):tag.append('noun')
            groups.append(dict(**g,group_id=gi,scope=scope,selected=S,tags=tag,offsets=offsets))
        img=OUT/'images'/Path(v['gallery_image']).name;img.parent.mkdir(exist_ok=True);shutil.copyfile(s['image_path'],img)
        out.append(dict(question_id=q,rank=i%2,smoke=i<2,kind=v['kind'],sample=s,spec=spec,groups=groups,tokens=v['tokens'],prompt_expanded=v['prompt_expanded'],question_char_offset=v['question_char_offset'],history_answers={k:v['rows'][k] for k in ['All-Writer','Semantic-k']},gallery_image=str(img.relative_to(OUT)),pos_parse=parse,input_hashes=inputs_hashes[s['task']+'__'+q]))
    atomic_json(OUT/'case_manifest.json',dict(cases=out,selected_cases_sha256=sha256(selected),case_source_commit=BASE,semantic_source_commit=SEM,pos_annotation_sha256=sha256(pospath),pos_annotation_commit='11fe70e82d6b608da21d9bd375ce49abf390d4d5'))
    shutil.copyfile(selected,OUT/'selected_cases.csv')
    atomic_json(OUT/'cpu_input_validation.json',dict(passed=True,cases=16,harm=12,rescue=4,unique_images=16,rank_counts=dict(collections.Counter(v['rank'] for v in out)),original_inputs_and_groups_exact=True,native_tokenizer_offsets_exact=True,answers_reused_not_generated=True,p_minus_one_excluded_from_R=True,ranking_rules_sha256=sha256(OUT/'ranking_rules.json'),cuda_initialized=torch.cuda.is_initialized(),binding=binding()))
    assert not torch.cuda.is_initialized();print('FROZEN CASES',[(v['question_id'],v['rank'],v['smoke']) for v in out],flush=True)
if __name__=='__main__':main()
