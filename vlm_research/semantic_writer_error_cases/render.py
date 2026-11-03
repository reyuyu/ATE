"""Portable offline HTML and one-case-per-page PDF, using frozen real images."""
import argparse,base64,csv,hashlib,html,json,mimetypes,os
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from PIL import Image

PALETTE={'selected':('#DCF3E5','#166534'),'question':('#FFEDD5','#9A3412'),'template':('#E5E7EB','#4B5563')}
SEEDS=[20261103,20261107,20261111]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(t):return html.escape(str(t))
def text(g):return g['decoded'].strip() or ('[newline]' if '\n' in g['decoded'] else '[space]')
def group_state(c,g):return 'selected' if set(g['positions'])<=set(c['spec']['writers']['Semantic-k']) else 'question' if g['question'] else 'template'
def answers(c):
    r=c['rows'];return [('Gold',c['sample']['answer'],None),('All-Writer',r['All-Writer']['text'],int(r['All-Writer']['correct'])),('Semantic-k',r['Semantic-k']['text'],int(r['Semantic-k']['correct'])),('Question-Last-k',r['Question-Last-k']['text'],int(r['Question-Last-k']['correct']))]+[(f'Q-Random {seed}',r[f'Question-Random-k-{seed}']['text'],int(r[f'Question-Random-k-{seed}']['correct'])) for seed in SEEDS]
def write_html(out,cases,plan):
    chunks=[]
    for i,c in enumerate(cases):
        s=c['sample'];qid=c['question_id'];spec=c['spec'];k=len(spec['writers']['Semantic-k']);R=len(spec['relay_positions']);tokens={t['position']:t for t in c['tokens']}
        img=out/c['gallery_image'];mime=mimetypes.guess_type(img.name)[0] or 'image/jpeg';uri='data:'+mime+';base64,'+base64.b64encode(img.read_bytes()).decode()
        chips=[]
        for g in spec['groups']:
            detail=' | '.join(f"{p}: id={tokens[p]['token_id']} {tokens[p]['piece']} offsets={tokens[p]['offset']}" for p in g['positions'])
            chips.append(f'<span class="chip {group_state(c,g)}" title="{esc(detail)}"><b>{esc(text(g))}</b><small>p={esc(",".join(map(str,g["positions"])))} · n={len(g["positions"])}</small></span>')
        answerrows=''.join(f'<tr><th>{esc(label)}</th><td>{esc(value)}</td><td>{"" if ok is None else "✓" if ok else "✗"}</td></tr>' for label,value,ok in answers(c))
        tokenrows=''.join(f'<tr class="{group_state(c,next(g for g in spec["groups"] if t["position"] in g["positions"]))}"><td>{t["position"]}</td><td>{t["token_id"]}</td><td>{esc(t["piece"])}</td><td>{esc(t["offset"])}</td><td>{"Question" if t["question"] else "Template"}</td><td>{"selected" if t["selected"] else "omitted"}</td></tr>' for t in c['tokens'])
        singles=plan.get(qid,{}).get('singles',[])
        candidates='；'.join(f'{v["id"]}: {text(v["group"])} @{v["added"]}' for v in singles) or '纠正反例：不纳入补词测试。'
        omitted=' | '.join(text(g) for g in c['semantic_core_dropped_groups']) or '无：所有原词法核心候选均保留'
        chunks.append(f'''<article class="case" id="case-{qid}" data-task="{esc(s['task'])}" data-kind="{c['kind']}">
<header><span class="number">{i+1:02d}</span><div><h2>{esc(qid)} <small>{esc(s['task'])}</small></h2><p>{'错误案例：All 对 / Semantic 错' if c['kind']=='harm' else '纠正反例：All 错 / Semantic 对'} · {'图片不重叠确认子集' if c['confirmatory'] else '历史探索图片；必含例外'}</p></div></header>
<h3>{esc(s['question'])}</h3><div class="pair"><figure><img src="{uri}" alt="GQA original image {qid}" loading="lazy"><figcaption>原始图片 · 等比显示 · <a href="{esc(c['gallery_image'])}">原文件</a></figcaption></figure><table class="answers"><tbody>{answerrows}</tbody></table></div>
<p class="budget"><b>k / R = {k} / {R} = {100*k/R:.2f}%</b> · Question Token = {len(spec['question_positions'])} · 核心候选预算缩减 = {spec['natural']['budget_reduction_for_whole_word_suffix']} · P−1 = {s['input_mapping']['p_minus_one']}</p>
<p class="offset">Question字符区间 [{c['question_char_offset'][0]}, {c['question_char_offset'][1]})；位置为0-based、图像展开后的实际Prompt位置。</p>
<div class="legend"><span class="selected">绿色：Semantic实际选中</span><span class="question">橙色：未选Question</span><span class="template">灰色：未选指令/模板</span></div>
<div class="tokens">{''.join(chips)}</div><p class="scope">仅展示 R 接收位置；576 个 Image Token 与 P−1 不属于本次可补回位置。n 为该完整词组的实际 Subword Token 数。</p>
<div class="note"><b>只读归因与待检验假说</b><p>{esc(c['interpretation'])}</p><p>真实遗漏的核心组：{esc(omitted)}</p></div>
<div class="pending"><b>补词因果验证：未执行（启动检查时四卡均有其他任务）</b><p>{esc(candidates)}</p><p>单组、全部Question、全部模板、等预算位置对照与条件性替换均没有新生成答案；不能据此判断补回是否成功。</p></div>
<details><summary>查看每个实际Token的ID、位置、Subword和字符Offset</summary><div class="scroll"><table><thead><tr><th>Position</th><th>ID</th><th>Subword</th><th>Offset</th><th>Scope</th><th>Status</th></tr></thead><tbody>{tokenrows}</tbody></table></div></details>
</article>''')
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATE · Semantic-k 错误案例画廊</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#17212c;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:32px 20px}.intro{padding:26px 30px;background:#132c42;color:#fff;border-radius:12px}.intro h1{margin:0 0 12px;font-size:29px}.intro p{margin:9px 0}.notice{background:#fff3d4;color:#553900;padding:12px 16px;border-radius:7px}.controls{padding:13px 0;background:#f3f5f7;display:flex;gap:10px;align-items:center;flex-wrap:wrap}select,input{font:inherit;padding:8px;border:1px solid #bdc6d1;border-radius:6px}input{flex:1;min-width:200px}.case{margin:18px 0 28px;padding:26px 30px;background:white;border:1px solid #dce1e7;border-radius:12px;box-shadow:0 4px 18px #17212c08}.case header{display:flex;gap:14px;align-items:center}.number{font-size:28px;color:#71839a}h2{margin:0;font-size:23px}h2 small{font-size:14px;color:#607082}header p{margin:2px 0;color:#607082}h3{font-size:21px;line-height:1.4;margin:17px 0}.pair{display:grid;grid-template-columns:minmax(230px,45%) 1fr;gap:26px;align-items:center}figure{margin:0;text-align:center;background:#f8fafc;border-radius:7px;padding:10px}figure img{max-width:100%;max-height:320px;object-fit:contain}figcaption,.scope,.offset{font-size:12px;color:#647386}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #e5e9ef}th{font-weight:600}.answers tr:nth-child(3){background:#eaf6ee}.budget{margin:20px 0 5px}.offset{margin:0 0 10px}.legend{display:flex;gap:8px;flex-wrap:wrap;font-size:12px;margin:12px 0}.legend span{padding:4px 8px;border-radius:4px}.selected{background:#dcf3e5;color:#166534}.question{background:#ffedd5;color:#9a3412}.template{background:#e5e7eb;color:#4b5563}.tokens{display:flex;gap:6px;flex-wrap:wrap;padding:12px;border:1px solid #e5e9ef;border-radius:6px}.chip{display:inline-flex;flex-direction:column;padding:5px 7px;border-radius:4px;min-width:42px}.chip b{font-size:14px;white-space:pre-wrap}.chip small{font:10px/1.5 ui-monospace,monospace}.note{margin:18px 0 12px;padding:14px 17px;background:#f0f5fa;border-left:3px solid #607f9f}.note p,.pending p{margin:6px 0}.pending{font-size:13px;padding:13px 17px;background:#fff8e7;border-left:3px solid #d3a03a}details{margin-top:15px}summary{cursor:pointer;color:#385a79}.scroll{overflow:auto;margin-top:10px}.scroll table{font:12px/1.4 ui-monospace,monospace}footer{font-size:12px;color:#607082;text-align:center;padding:25px}a{color:#245779}[hidden]{display:none!important}@media(max-width:700px){.pair{grid-template-columns:1fr}.case{padding:20px}.intro{padding:20px}.intro h1{font-size:23px}.case h3{font-size:18px}}@media print{.controls{display:none}.case{break-before:page;box-shadow:none;border:0;margin:0;padding:10px}.intro{color:black;background:white}.pending{break-inside:avoid}details{display:none}}
</style></head><body><main><section class="intro"><h1>Semantic-k 为什么答错？</h1><p>12 个分任务错误案例 + 4 个纠正反例。图片、完整答案和实际 Token 集合均复用冻结的 GQA3000 正式实验。</p><p>4 / 12 例存在词法核心组遗漏，8 / 12 例保留全部核心候选；另有答案表面形式不匹配。计数仅描述这些挑选案例，不能外推全3000题。</p><p class="notice">本次交付为只读分析。候选已预先锁定，但启动前四张 GPU 均被其他任务占用，因此未执行任何补词因果测试。所有恢复成功率均未知。</p></section><div class="controls"><select id="kind"><option value="">全部16例</option><option value="harm">仅12个错误案例</option><option value="rescue">仅4个纠正反例</option></select><select id="task"><option value="">全部任务</option>'''+''.join(f'<option>{t}</option>' for t in ['ChooseAttr','ChooseCat','ChooseRel','CompareAttr','LogicalObj','QueryAttr'])+'''</select><input id="search" placeholder="搜索问题、Question ID 或归因说明" aria-label="搜索案例"><span id="count">16 / 16</span></div>'''+''.join(chunks)+'''<footer>ATE · 参考正式Commit 01c82b1d39360fc73bca3d10d12886672ffee370 · 全部颜色来自真实Manifest，未将假说当作因果结论。</footer></main><script>
const cards=[...document.querySelectorAll('.case')],kind=document.getElementById('kind'),task=document.getElementById('task'),search=document.getElementById('search');function filter(){let n=0;for(const c of cards){c.hidden=!!((kind.value&&kind.value!==c.dataset.kind)||(task.value&&task.value!==c.dataset.task)||(search.value&&!c.textContent.toLowerCase().includes(search.value.toLowerCase())));if(!c.hidden)n++}document.getElementById('count').textContent=n+' / 16'}for(const el of [kind,task,search])el.addEventListener('input',filter);
</script></body></html>'''
    (out/'gallery.html').write_text(doc,encoding='utf-8')

def font_setup():
    latin=next(p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('C:/Windows/Fonts/arial.ttf')] if p.exists())
    cjk=next(p for p in [Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),Path('C:/Windows/Fonts/msyh.ttc')] if p.exists())
    pdfmetrics.registerFont(TTFont('Latin',str(latin)));pdfmetrics.registerFont(TTFont('CJK',str(cjk)))
    return dict(latin=str(latin),cjk=str(cjk))
def make_pdf(out,cases,plan):
    fonts=font_setup();pdf=out/'case_cards.pdf';cv=canvas.Canvas(str(pdf),pagesize=A4);cv.setTitle('ATE Semantic-k Error Case Cards | Read-only attribution')
    W,H=A4;left=30;width=W-60
    def para(txt,x,y,w,size=9,font='CJK',color='#17212c',leading=None):
        style=ParagraphStyle('p',fontName=font,fontSize=size,leading=leading or size*1.42,textColor=colors.HexColor(color),wordWrap='CJK' if font=='CJK' else None)
        p=Paragraph(txt,style);_,h=p.wrap(w,1000);p.drawOn(cv,x,y-h);return y-h
    for i,c in enumerate(cases):
        s=c['sample'];spec=c['spec'];qid=c['question_id'];k=len(spec['writers']['Semantic-k']);R=len(spec['relay_positions'])
        y=H-28;y=para(f'ATE / Semantic-k 错误归因 · {"错误案例" if c["kind"]=="harm" else "纠正反例"} · {i+1:02d}/16',left,y,width,10)
        y-=5;y=para(esc(qid+'  |  '+s['task']),left,y,width,12,'Latin')
        y-=5;y=para(esc(s['question']),left,y,width,11,'Latin',leading=14)
        y-=10;top=y;img=out/c['gallery_image'];iw,ih=Image.open(img).size;boxw=225;boxh=160;scale=min(boxw/iw,boxh/ih)
        cv.drawImage(ImageReader(str(img)),left+(boxw-iw*scale)/2,top-boxh+(boxh-ih*scale)/2,width=iw*scale,height=ih*scale,preserveAspectRatio=True,mask='auto')
        para('原始图片：等比显示，不裁剪',left,top-boxh-3,boxw,7.5)
        ax=left+boxw+13;aw=width-boxw-13;ay=top
        for label,value,ok in answers(c):
            mark='' if ok is None else ' [correct]' if ok else ' [wrong]'
            if label=='Semantic-k':
                cv.setFillColor(colors.HexColor('#EAF6EE'));cv.rect(ax-3,ay-27,aw+6,29,fill=1,stroke=0)
            ay=para(esc(label+mark),ax,ay,aw,7.6,'Latin',color='#576779',leading=9)
            ay=para(esc(value),ax,ay,aw,9.5,'Latin',leading=12);ay-=3
        y=min(top-boxh-17,ay)-8
        y=para(f'k / R = {k} / {R} = {100*k/R:.2f}% · Question Tokens = {len(spec["question_positions"])} · P-1 = {s["input_mapping"]["p_minus_one"]}',left,y,width,8,'Latin')
        y=para(f'图像不重叠确认子集：{"是" if c["confirmatory"] else "否（用户指定的必含案例）"}；核心预算缩减 {spec["natural"]["budget_reduction_for_whole_word_suffix"]} Token；位置均为 0-based。',left,y-2,width,8)
        y-=8
        for j,(state,label) in enumerate([('selected','已选 Semantic'),('question','未选 Question'),('template','未选指令/模板')]):
            x=left+j*178;cv.setFillColor(colors.HexColor(PALETTE[state][0]));cv.roundRect(x,y-14,168,18,3,fill=1,stroke=0);para(esc(label),x+5,y+1,160,8)
        y-=25;x=left;rowheight=23
        for g in spec['groups']:
            label=text(g);pos=','.join(map(str,g['positions']));meta=f'{pos} / n={len(g["positions"])}'
            chipw=max(pdfmetrics.stringWidth(label,'Latin',7.4),pdfmetrics.stringWidth(meta,'Latin',6.3))+10;chipw=min(width,chipw)
            if x+chipw>left+width:y-=rowheight+4;x=left
            bg,fg=PALETTE[group_state(c,g)];cv.setFillColor(colors.HexColor(bg));cv.roundRect(x,y-rowheight,chipw,rowheight,3,fill=1,stroke=0)
            para(esc(label),x+4,y-2,chipw-8,7.4,'Latin',fg,leading=9)
            para(esc(meta),x+4,y-12,chipw-8,6.3,'Latin',fg,leading=8);x+=chipw+4
        y-=rowheight+9;y=para('上方完整展示 R 词组：词组下方为实际位置与 Subword 数。图像Token和 P-1 不属于可补回位置；逐Token ID与字符Offset见HTML展开表。',left,y,width,7.5)
        y-=9;cv.setStrokeColor(colors.HexColor('#CAD6E2'));cv.line(left,y,left+width,y);y-=8
        y=para('只读观察与候选解释（尚无新因果证据）',left,y,width,9.5);y=para(esc(c['interpretation']),left,y-5,width,9,leading=13)
        singles=plan.get(qid,{}).get('singles',[])
        if singles:
            desc='；'.join(f'{text(v["group"])} @{v["added"]}' for v in singles)
            y=para('冻结的单组候选：'+esc(desc),left,y-8,width,8,leading=11)
        assert y>75,('Case page overflow',qid,y)
        cv.setFillColor(colors.HexColor('#FFF3D4'));cv.roundRect(left,35,width,33,4,fill=1,stroke=0)
        para('因果验证未执行：启动前四卡均有其他任务。没有补词、模板补回或等预算替换的恢复结论。',left+8,59,width-16,8)
        para('仅为选定案例，不外推全3000题；历史完整答案和Token集合均未重新生成。',left+8,47,width-16,7)
        cv.showPage()
    cv.save();return fonts
def main():
    a=argparse.ArgumentParser();a.add_argument('--results-dir',required=True);args=a.parse_args();out=Path(args.results_dir)
    cases=read(out/'case_annotations.json')['cases'];plan=read(out/'candidate_plan.json')['cases'];assert len(cases)==16
    for c in cases:assert sha(out/c['gallery_image'])==c['sample']['image_sha256']
    write_html(out,cases,plan);fonts=make_pdf(out,cases,plan)
    (out/'render_validation.json').write_text(json.dumps(dict(passed=True,cases=16,PDF_pages=16,one_case_per_page=True,HTML_self_contained=True,
        original_images_all_exact=True,all_three_random_seed_answers_shown=True,actual_token_positions_and_subword_counts=True,
        causal_not_run_disclosure_on_every_card=True,source_sha256=sha(__file__),fonts=fonts,
        artifacts={name:sha(out/name) for name in ['gallery.html','case_cards.pdf']}),ensure_ascii=False,indent=2),encoding='utf-8')
    print('RENDERED 16 cases, HTML and PDF',pdfmetrics.getRegisteredFontNames())
if __name__=='__main__':main()
