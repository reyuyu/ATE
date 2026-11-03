"""Portable image cards from actual frozen POS and model results; no inferred answers."""
import json,csv,html,base64,sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from PIL import Image
def main():
    out=Path(sys.argv[1]);p=json.loads((out/'token_plan.json').read_text(encoding='utf-8'));rows=list(csv.DictReader((out/'per_question.csv').open(encoding='utf-8')));lookup={(r['question_id'],r['condition']):r for r in rows}
    font=next(x for x in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if x.exists());pdfmetrics.registerFont(TTFont('Text',str(font)))
    c=canvas.Canvas(str(out/'case_cards.pdf'),pagesize=A4);W,H=A4;pages=[];layout=[]
    for i,q in enumerate(p['historical_ids']):
        v=p['cases'][q];s=v['sample'];N,D,S=map(set,[v['N'],v['D'],v['S']]);R=len(v['spec']['relay_positions']);img=out/v['gallery_image'];im=Image.open(img);iw,ih=im.size;ratio=min(250/iw,195/ih);dw,dh=iw*ratio,ih*ratio
        c.setFillColorRGB(.08,.17,.23);c.rect(0,H-63,W,63,fill=1,stroke=0);c.setFillColorRGB(1,1,1);c.setFont('Text',16);c.drawString(30,H-30,f'{i+1:02d}  {q}  |  {s["task"]}');c.setFont('Text',9);c.drawString(30,H-48,'Noun + Article Visual Writing | historical diagnostic case, outside pilot120')
        def para(text,x,y,width,size=10,color='#17212c'):
            st=ParagraphStyle('p',fontName='Text',fontSize=size,leading=size*1.35,textColor=color);pp=Paragraph(text,st);ww,hh=pp.wrap(width,1000);pp.drawOn(c,x,y-hh);return y-hh
        y=para(html.escape(s['question']),30,H-78,W-60,13)-12
        c.drawImage(ImageReader(im),30,y-dh,width=dw,height=dh,preserveAspectRatio=True)
        yy=para('<b>Gold:</b> '+html.escape(s['answer']),305,y,W-335,12)
        yy=para(f'S={len(S)} / R={R} ({100*len(S)/R:.1f}%)<br/>N={len(N)}, D={len(D)}<br/>N outside S={len(N-S)}<br/>D outside S={len(D-S)}',305,yy-12,W-335,10)
        y=min(y-dh,yy)-18
        y=para('Token groups: green=S; blue=N; orange=D. Labels show membership, actual prompt positions and subword count.',30,y,W-60,9)-9
        chips=[];x=30;lineh=33
        for g in v['spec']['groups']:
            ps=set(g['positions']);txt=g['decoded'].strip() or '[newline]';flags=''.join(k for k,st in [('S',S),('N',N),('D',D)] if ps<=st)
            if not g['question'] and not flags:continue
            color='#ffdfab' if 'D' in flags else '#cfe8fb' if 'N' in flags else '#d9f1df' if 'S' in flags else '#eceff2'
            width=max(47,min(160,pdfmetrics.stringWidth(txt,'Text',10)+14));
            if x+width>W-30:x=30;y-=lineh+3
            c.setFillColor(color);c.roundRect(x,y-lineh,width,lineh,3,fill=1,stroke=0);c.setFillColorRGB(.08,.13,.18);c.setFont('Text',9);c.drawString(x+5,y-12,txt);c.setFont('Text',6.5);c.drawString(x+5,y-25,f'{flags or "-"} @{g["positions"][0]} n={len(ps)}');x+=width+5
            chips.append(f'<span style="background:{color}" title="positions={g["positions"]}"><b>{html.escape(txt)}</b><small>{flags or "-"} | p={g["positions"]} | n={len(ps)}</small></span>')
        y-=lineh+16
        labels=list(v['writers']);answers=[]
        for label in labels:
            r=lookup[(q,label)];value=r['answer'] if r['status'] in ['generated','reused'] else ('NOT COMPARABLE (exact budget)' if r['status']=='unavailable_exact_budget' else 'NOT RUN');ok='correct' if r['correct']=='1' else 'wrong' if r['correct']=='0' else '--'
            answers.append((label,value,ok,r['k']))
        c.setFont('Text',9);c.setFillColorRGB(.08,.13,.18);c.drawString(30,y,'Condition');c.drawString(250,y,'Actual answer');c.drawString(475,y,'ACC / k');y-=16
        for label,value,ok,k in answers:
            c.setFillColorRGB(.15,.25,.35);c.setFont('Text',8);c.drawString(30,y,label)
            yy=para(html.escape(value),250,y+9,215,8);c.setFont('Text',8);c.drawString(475,y,ok+' / '+str(k));y=min(y-16,yy-5)
        parsing='; '.join(f"{t['text']}/{t['pos']} -{t['dep']}-> {t['head_text']}" for t in v['parse']['tokens'] if t['noun'] or t['article'])
        sw=v['writers']['Article-Swap'];rem=sorted(S-set(sw)) if sw is not None else None
        swapline='Article-Swap writers: enable '+str(sorted(D-S))+'; disable '+str(rem)+' (unchanged k and prompt).'
        y=para(swapline,30,y-7,W-60,8)-6
        y=para('Parser audit: '+html.escape(parsing),30,y-3,W-60,8)-6
        y=para('Post-hoc case intervention only. Recovery does not identify a unique visual storage token or imply a population-wide selector improvement.',30,y,W-60,8,'#6d4b00')
        assert y>32,(q,y);layout.append(dict(question_id=q,bottom=y))
        c.setFont('Text',8);c.setFillColorRGB(.4,.45,.5);c.drawString(30,20,f'ATE | 0-based expanded prompt positions | Page {i+1}/12');c.showPage()
        uri='data:image/jpeg;base64,'+base64.b64encode(img.read_bytes()).decode();table=''.join(f'<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{d}</td><td>{k}</td></tr>' for a,b,d,k in answers)
        pages.append(f'<article><h2>{i+1:02d} - {q} | {s["task"]}</h2><h3>{html.escape(s["question"])}</h3><div class="pair"><img src="{uri}"><div><p>Gold: <b>{html.escape(s["answer"])}</b></p><p>S={len(S)} / R={R}; N={len(N)}, D={len(D)}</p><p>New N positions: {sorted(N-S)}<br>New D positions: {sorted(D-S)}</p><p>{html.escape(swapline)}</p></div></div><p>Green=S; blue=N; orange=D. Membership and all actual positions appear below.</p><div class="chips">{"".join(chips)}</div><table><tr><th>Condition</th><th>Actual answer</th><th>ACC</th><th>k</th></tr>{table}</table><details><summary>Full POS / dependency and token mapping</summary><pre>{html.escape(json.dumps(v["parse"],indent=2))}</pre></details><p class="caution">Exploratory historical case; never included in pilot120 statistics.</p></article>')
    c.save();(out/'gallery.html').write_text('<!doctype html><html><meta charset="utf-8"><title>ATE Noun + Article Cases</title><style>body{font:15px/1.5 Arial;background:#eef2f5;color:#182b3a;margin:0}main{max-width:1050px;margin:auto;padding:24px}article{background:white;padding:28px;margin:24px 0;border-radius:10px}h1,h2{color:#142f44}.pair{display:grid;grid-template-columns:50% 50%;gap:20px}.pair img{width:100%;height:300px;object-fit:contain}.chips{display:flex;flex-wrap:wrap;gap:6px}.chips span{padding:7px;border-radius:4px}.chips small{display:block;font-size:10px}table{border-collapse:collapse;width:100%;margin-top:20px}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}.caution{color:#826000}pre{white-space:pre-wrap}@media(max-width:600px){.pair{display:block}}</style><main><h1>Noun + Article Visual Writing</h1><p>12 frozen historical error cases. Original images, actual whole-token groups and complete answers.</p>'+''.join(pages)+'</main></html>',encoding='utf-8');(out/'card_layout_validation.json').write_text(json.dumps(dict(pages=12,all_bottoms_above_footer=True,layout=layout)))
    hp=out/'gallery.html';doc=hp.read_text(encoding='utf-8');doc=doc.replace('</style>','*{box-sizing:border-box}.pair{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}td,th{overflow-wrap:anywhere}pre{overflow-wrap:anywhere}@media(max-width:600px){main{padding:12px}article{padding:16px}.pair{display:block}table{font-size:12px;table-layout:fixed}td,th{padding:5px}h2{overflow-wrap:anywhere}}</style>');hp.write_text(doc,encoding='utf-8')
if __name__=='__main__':main()
