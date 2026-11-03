"""CPU-only input cards; conspicuously no observed attention values."""
import json,html,base64,sys
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
def main():
    P=Path(sys.argv[1]);cases=json.loads((P/'case_manifest.json').read_text(encoding='utf-8'))['cases'];font=next(x for x in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if x.exists());pdfmetrics.registerFont(TTFont('Text',str(font)));pdf=canvas.Canvas(str(P/'case_cards.pdf'),pagesize=A4);W,H=A4;cards=[];layout=[]
    for i,v in enumerate(cases):
        s=v['sample'];q=v['question_id'];S=set(v['spec']['writers']['Semantic-k']);impath=P/v['gallery_image'];im=Image.open(impath);iw,ih=im.size;ratio=min(250/iw,190/ih);dw,dh=iw*ratio,ih*ratio
        def para(t,x,y,w,size=10,color='#172b3a'):
            pp=Paragraph(t,ParagraphStyle('p',fontName='Text',fontSize=size,leading=size*1.35,textColor=color));_,h=pp.wrap(w,900);pp.drawOn(pdf,x,y-h);return y-h
        pdf.setFillColor('#18354a');pdf.rect(0,H-65,W,65,fill=1,stroke=0);pdf.setFillColor('white');pdf.setFont('Text',15);pdf.drawString(30,H-28,f'{i+1:02d} | {q} | {s["task"]}');pdf.setFont('Text',9);pdf.drawString(30,H-47,'INPUT AUDIT PREVIEW - ATTENTION NOT COLLECTED')
        y=para(html.escape(s['question']),30,H-80,W-60,13)-12;pdf.drawImage(ImageReader(im),30,y-dh,width=dw,height=dh)
        yy=para('Gold: <b>'+html.escape(s['answer'])+'</b><br/><br/>Historical All-Writer: '+html.escape(v['history_answers']['All-Writer']['text'])+'<br/>Historical Semantic-k: '+html.escape(v['history_answers']['Semantic-k']['text']),305,y,W-335,11)
        yy=para(f'<br/>Case type: {v["kind"]}<br/>Historical S/R: {len(S)}/{len(v["spec"]["relay_positions"])}<br/>Planned worker: {v["rank"]}<br/>P-1: {s["input_mapping"]["p_minus_one"]} (outside R)',305,yy,W-335,10)
        y=min(y-dh,yy)-18;y=para('GPU preflight STOP: only one truly idle GPU; two required. No model loaded, no attention measured, no high-mass orange markers.',30,y,W-60,10,'#9a4e00')-15
        y=para('Real frozen groups below: green = historical Semantic selected; gray = omitted. Native Full would keep all original attention edges open. Positions are 0-based expanded prompt indices; n is subword count.',30,y,W-60,9)-12
        chips=[]
        for scope in ['Question','Instruction','Assistant/template']:
            y=para(html.escape(scope),30,y,W-60,10)-6;x=30;lineh=34
            for g in [g for g in v['groups'] if g['scope']==scope]:
                txt=g['decoded'].strip() or ('[newline]' if '\n' in g['decoded'] else '[space]');ps=g['positions'];color='#d9f1df' if g['selected'] else '#e7ebef';width=max(43,min(125,pdfmetrics.stringWidth(txt,'Text',9)+12))
                if x+width>W-30:x=30;y-=lineh+4
                pdf.setFillColor(color);pdf.roundRect(x,y-lineh,width,lineh,3,stroke=0,fill=1);pdf.setFillColor('#172b3a');pdf.setFont('Text',9);pdf.drawString(x+5,y-12,txt);pdf.setFont('Text',6.5);pdf.drawString(x+5,y-25,f'@{ps[0]} n={len(ps)}');x+=width+5
                chips.append(f'<span style="background:{color}"><b>{html.escape(txt)}</b><small>{html.escape(scope)} | p={ps} | n={len(ps)}</small></span>')
            y-=lineh+15
        y=para('Image Mass / Excess / Enrichment: NOT COLLECTED.<br/>Layer heatmaps and L8-11 / L0-15 / L16-19 Top5: UNAVAILABLE.<br/>High attention allocation alone would not establish causal importance.',30,y,W-60,10,'#9a4e00');assert y>35,(q,y);layout.append(dict(question_id=q,bottom=y));pdf.setFont('Text',8);pdf.setFillColor('#657585');pdf.drawString(30,20,f'ATE | 0 native forwards | Resource-stop input preview | Page {i+1}/16');pdf.showPage()
        uri='data:image/jpeg;base64,'+base64.b64encode(impath.read_bytes()).decode()
        cards.append(f'<article><h2>{i+1:02d} | {q} | {s["task"]} | {v["kind"]}</h2><h3>{html.escape(s["question"])}</h3><div class="pair"><img src="{uri}" alt="Frozen original GQA image {q}"><div><p>Gold: <b>{html.escape(s["answer"])}</b></p><p>历史 All-Writer：{html.escape(v["history_answers"]["All-Writer"]["text"])}</p><p>历史 Semantic-k：{html.escape(v["history_answers"]["Semantic-k"]["text"])}</p><p>S/R={len(S)}/{len(v["spec"]["relay_positions"])}；P−1={s["input_mapping"]["p_minus_one"]}，不属于R。</p><p class="stop">Attention未采集：无Mass、Excess、Enrichment、热图或Top5结果。</p></div></div><p>绿色＝历史Semantic选中；灰色＝未选。颜色只表示真实冻结集合，不表示本轮观测到高/低Mass。无橙框标记。</p><div class="chips">{"".join(chips)}</div><p>Native Full将保持所有原始连接开放；本轮未运行。Question、Instruction、Assistant/template按原始Offset分类，完整词组不变。</p></article>')
    pdf.save();doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATE 输入审计预览 - Attention未采集</title><style>*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#183045;font:15px/1.6 Arial,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:25px}header{padding:23px;background:#19364a;color:white;border-radius:10px}.stop{padding:13px;background:#fff1ce;color:#805000;border-radius:5px}article{background:white;border-radius:10px;padding:25px;margin:24px 0}.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px}.pair img{width:100%;height:300px;object-fit:contain}.chips{display:flex;gap:7px;flex-wrap:wrap}.chips span{padding:6px;border-radius:4px}.chips small{display:block;font-size:10px}h2{overflow-wrap:anywhere}@media(max-width:650px){main{padding:12px}article{padding:17px}.pair{display:block}.pair img{height:auto;max-height:320px}}@media print{article{break-before:page}}</style><main><header><h1>遗漏Semantic视觉接收者：输入审计预览</h1><p>固定16例，12个错误案例与4个纠正反例；未重新选题。</p><p class="stop">STOPPED_NO_TWO_IDLE_GPUS：仅1张真正空闲GPU，不满足本轮2卡要求。未加载模型、未运行Smoke或Prefill。此画廊仅展示真实冻结输入和历史答案，不能作为Attention结果。</p></header>'''+''.join(cards)+'</main></html>';(P/'gallery.html').write_text(doc,encoding='utf-8');(P/'card_layout_validation.json').write_text(json.dumps(dict(pages=16,input_preview_only=True,attention_not_collected=True,all_bottoms_above_footer=True,layout=layout),indent=2))
if __name__=='__main__':main()
