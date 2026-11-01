"""Real CPU processor geometry; only merged-token indices and small overlays retained."""
import hashlib
import inspect
import json
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.v2 import functional as F
from transformers import AutoProcessor
from transformers.image_utils import SizeDict
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from types import SimpleNamespace

from ..qwen3vl_early_windows.experiment import MODEL, prepare
from .preflight import OUT, HISTORY, SOURCES, dump, sha


def reconstruct(pixels, grid, patch, merge, temporal):
    t, gh, gw = grid
    assert t == 1 and gh % merge == gw % merge == 0
    # Native stream: merged-row, merged-col, intra-row, intra-col, C, time, patch-y, patch-x.
    block = pixels.reshape(gh//merge, gw//merge, merge, merge, 3, temporal, patch, patch)
    assert torch.equal(block[...,0,:,:], block[...,1,:,:]) and temporal == 2
    return block[:,:,:,:,:,0].permute(4,0,2,5,1,3,6).reshape(3,gh*patch,gw*patch)


def roi_cells(boxes, width, height, rows, cols):
    indices = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c*width/cols, (c+1)*width/cols
            y0, y1 = r*height/rows, (r+1)*height/rows
            if any(min(x1,x+w)>max(x0,x) and min(y1,y+h)>max(y0,y)
                   for x,y,w,h in (b['xywh'] for b in boxes)):
                indices.append(r*cols+c)
    return indices


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), 'CPU-only geometry validation must hide CUDA'
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    ip = proc.image_processor
    assert (ip.patch_size,ip.merge_size,ip.temporal_patch_size)==(16,2,2)
    assert type(ip).__name__ == 'Qwen2VLImageProcessorFast'
    boxes = json.loads((OUT/'resolved_boxes.json').read_text())
    samples = json.loads((HISTORY/'sample_manifest.json').read_text())['samples']
    pool = []
    for task in SOURCES:
        available = [s for s in samples if s['task']==task]
        pool += sorted(available,key=lambda s:hashlib.sha256(('roi_cpu_geometry_v1:'+s['input_binding']).encode()).hexdigest())[:4]
    assert len(pool)==24
    smoke_keys={f'{s["task"]}__{s["question_id"]}' for s in pool}
    panels, audits, mappings = [], [], {}
    for s in samples:
        inputs, layout = prepare(proc,s)
        assert layout == s['layout'], 'Actual official processor differs from frozen history'
        grid = layout['image_grid_thw']
        t,gh,gw = grid
        with Image.open(s['image_path']) as handle:
            image = handle.convert('RGB')
        width,height = image.size
        rh,rw = smart_resize(height,width,factor=32,
                            min_pixels=ip.size['shortest_edge'],max_pixels=ip.size['longest_edge'])
        assert (rh,rw)==(gh*16,gw*16) and t==1
        raw = F.pil_to_tensor(image).unsqueeze(0)
        resized = ip.resize(raw,SizeDict(height=rh,width=rw),interpolation=F.InterpolationMode.BICUBIC)
        expected = ip.rescale_and_normalize(resized,ip.do_rescale,ip.rescale_factor,
                                           ip.do_normalize,tuple(ip.image_mean),tuple(ip.image_std))[0]
        restored = reconstruct(inputs['pixel_values'],grid,16,2,2)
        assert torch.equal(expected,restored), 'Raw patch order reconstruction failed'
        rows,cols = gh//2,gw//2
        assert rows*cols == layout['image_tokens'] == len(layout['image_positions'])
        stub=SimpleNamespace(spatial_merge_size=2,rotary_pos_emb=lambda n:torch.arange(n,dtype=torch.float32)[:,None])
        actual_coords=Qwen3VLVisionModel.rot_pos_emb(stub,torch.tensor([grid]))
        expected_coords=torch.tensor([[2*r+dr,2*c+dc] for r in range(rows) for c in range(cols)
                                     for dr in range(2) for dc in range(2)])
        assert torch.equal(actual_coords,expected_coords), 'Native vision spatial ordering differs'
        target = boxes[f'{s["task"]}__{s["question_id"]}']
        roi = roi_cells(target,width,height,rows,cols)
        assert roi and min(roi)>=0 and max(roi)<rows*cols
        key=f'{s["task"]}__{s["question_id"]}'
        mappings[key]=dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
            input_binding=s['input_binding'],grid_thw=grid,merged_grid=[rows,cols],
            roi_token_indices=roi,roi_fraction=len(roi)/(rows*cols),bbox_objects=target,
            random_seed=int(hashlib.sha256(('random_roi_v1:'+s['input_binding']).encode()).hexdigest()[:16],16),
            exact_pixel_reconstruction=True,native_rotary_patch_order_exact=True)
        if key not in smoke_keys:
            continue
        # Each numbered cell is one merged LLM image token, NOT one 16x16 raw patch.
        display = image.copy()
        display.thumbnail((640,560))
        tint = Image.new('RGBA',display.size,(0,0,0,0)); draw=ImageDraw.Draw(tint)
        sx,sy=display.width/width,display.height/height
        for idx in range(rows*cols):
            r,c=divmod(idx,cols)
            rect=[c*display.width/cols,r*display.height/rows,(c+1)*display.width/cols,(r+1)*display.height/rows]
            draw.rectangle(rect,fill=(240,35,120,75) if idx in roi else None,outline=(255,255,255,100),width=1)
        for b in target:
            x,y,w,h=b['xywh'];draw.rectangle([x*sx,y*sy,(x+w)*sx,(y+h)*sy],outline=(0,255,75,255),width=3)
        rendered=Image.alpha_composite(display.convert('RGBA'),tint).convert('RGB')
        panel=Image.new('RGB',(640,640),'white');panel.paste(rendered,(0,52))
        d=ImageDraw.Draw(panel);d.text((8,6),f'{s["task"]} {s["question_id"]} | grid {rows}x{cols} | ROI {len(roi)}',fill='black')
        d.text((8,23),s['question'][:95],fill='black')
        path=OUT/'mapping_overlays'/f'{s["task"]}__{s["question_id"]}.png'
        path.parent.mkdir(parents=True,exist_ok=True);panel.save(path)
        panels.append(panel.resize((480,480)))
        audits.append(dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
            grid_thw=grid,merged_grid=[rows,cols],image_tokens=layout['image_tokens'],
            roi_token_indices=roi,bbox_objects=target,exact_pixel_reconstruction=True,
            official_processor_matches_frozen_layout=True,visual_overlay=str(path.relative_to(OUT))))
    sheet=Image.new('RGB',(480*4,480*6),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,((i%4)*480,(i//4)*480))
    sheet.save(OUT/'mapping_overview.png')
    dump(OUT/'roi_mapping.json',mappings)
    dump(OUT/'processor_validation.json',dict(status='cpu_geometry_passed',
        count=240,visualized=24,per_task_visualized=4,smoke_keys=sorted(smoke_keys),
        selection='All 240 CPU geometry checks; four per task selected by frozen input hash for GPU smoke/overlays',
        scope='CPU pixel packing and geometry only; NO model forward/logit/Attention/DeepStack validation',
        processor_class=type(ip).__name__,patch_size=16,merge_size=2,temporal_patch_size=2,
        merger_source_sha256=sha(inspect.getfile(Qwen3VLVisionPatchMerger)),
        processor_source_sha256=sha(inspect.getfile(type(ip))),samples=audits))
    print('240 actual processor checks passed; 24 GPU smoke cases and overlays frozen.',flush=True)


if __name__=='__main__':
    main()
