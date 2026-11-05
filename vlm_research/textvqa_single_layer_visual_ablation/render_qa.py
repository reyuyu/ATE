"""Render the final vector figures for visual inspection; private QA copies only."""
from pathlib import Path
import subprocess,json,hashlib
from PIL import Image,ImageDraw

SOURCE=Path('/data/lk/ATE-worktrees/textvqa-single-layer-visual-ablation/vlm_research/results/textvqa_single_layer_visual_ablation/figures')
TARGET=Path('/data/lk/ATE-resources/experiments/textvqa_single_layer_visual_ablation/qa_final')
TARGET.mkdir(exist_ok=True)
tiles=[];records=[]
for p in sorted(SOURCE.glob('*.pdf')):
 target=TARGET/p.stem
 subprocess.run(['pdftoppm','-f','1','-singlefile','-scale-to','1600','-png',str(p),str(target)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 png=target.with_suffix('.png')
 im=Image.open(png).convert('RGB');im.thumbnail((1000,650))
 tile=Image.new('RGB',(1020,685),'white');tile.paste(im,((1020-im.width)//2,28));ImageDraw.Draw(tile).text((10,8),p.name,fill='black');tiles.append(tile)
 records.append(dict(pdf=p.name,pdf_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rendered_png=str(png),render_sha256=hashlib.sha256(png.read_bytes()).hexdigest()))
canvas=Image.new('RGB',(2040,685*((len(tiles)+1)//2)),'#dddddd')
for j,im in enumerate(tiles):canvas.paste(im,((j%2)*1020,(j//2)*685))
canvas.save(TARGET/'contact.png')
(TARGET/'render_manifest.json').write_text(json.dumps(dict(records=records,render_tool='pdftoppm',PDF_scale_to_pixels=1600,manual_review_pending=True),indent=2)+'\n')
print(str(TARGET/'contact.png'))
