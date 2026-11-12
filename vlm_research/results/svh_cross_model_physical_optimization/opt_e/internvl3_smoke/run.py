"""One BF16/Eager instance, four official chat cases and fixed hook checks."""
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from .common import OUT, ROOT, read, sha256, write
from .engine import cache_audit, hooks_count, official_preprocessor, partitions, replay, tensor_hash

RAW = Path('/data/lk/ATE-resources/experiments/internvl3_8b_single_gpu_smoke')


def command(*args):
    return subprocess.check_output(list(args), text=True).strip()


class ResourceMonitor:
    def __init__(self):
        self.rows, self.errors, self.stage = [], [], 'load'
        self.stop = threading.Event()

    def sample(self):
        try:
            raw = command('nvidia-smi','-i','1','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits')
            for line in raw.splitlines():
                pid, uuid, memory = [p.strip() for p in line.split(',')]
                if int(pid) == os.getpid():
                    self.rows.append(dict(time=time.time(), stage=self.stage, pid=int(pid), uuid=uuid, memory_mib=int(memory)))
        except Exception as error:
            self.errors.append(str(error))

    def __enter__(self):
        def monitor():
            while not self.stop.wait(.5):
                self.sample()
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stop.set()
        self.thread.join()
        write(OUT/'gpu1_resource_log.json',dict(rows=self.rows, errors=self.errors,
            peak_process_memory_mib=max((r['memory_mib'] for r in self.rows), default=0),
            pid=os.getpid(), cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))


class ChatCapture:
    """Capture actual official chat inputs and its first LM prefill, unchanged."""
    def __init__(self, model):
        self.model, self.saved, self.calls, self.vision = model, {}, [], []

    def __enter__(self):
        from transformers.cache_utils import DynamicCache
        self.stack = ExitStack()
        original = self.model.generate
        def generate(*args, **kwargs):
            assert not args
            self.input_ids = kwargs['input_ids'].detach().cpu().clone()
            self.attention_mask = kwargs['attention_mask'].detach().cpu().clone()
            output = original(*args, **kwargs)
            assert isinstance(output, torch.Tensor) and output.shape[0] == 1
            self.generated_ids = output[0].detach().cpu().tolist()
            return output
        self.stack.enter_context(patch.object(self.model, 'generate', generate))
        def before(module, args, kwargs):
            assert not args
            if not self.saved:
                for key, value in kwargs.items():
                    if key == 'past_key_values':
                        assert isinstance(value, DynamicCache) and value.get_seq_length() == 0
                        self.saved[key] = 'DynamicCache(empty)'
                    elif isinstance(value, torch.Tensor):
                        self.saved[key] = value.detach().cpu().clone()
                    else:
                        assert value is None or isinstance(value, (bool, int, float, str))
                        self.saved[key] = value
        def after(module, args, kwargs, output):
            current = output.logits[0, -1].detach().cpu().clone()
            assert torch.isfinite(current).all()
            n = int(kwargs['inputs_embeds'].shape[1]) if kwargs.get('inputs_embeds') is not None else int(kwargs['input_ids'].shape[1])
            if not self.calls:
                self.first_logits = current
                self.prefill_cache = cache_audit(output.past_key_values, len(self.model.language_model.model.layers), n)
            self.calls.append(dict(query_length=n, cache_length=int(output.past_key_values.get_seq_length())))
        self.stack.callback(self.model.language_model.register_forward_pre_hook(before, with_kwargs=True).remove)
        self.stack.callback(self.model.language_model.register_forward_hook(after, with_kwargs=True).remove)
        def vision(module, args, output):
            tensor = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
            self.vision.append(dict(module='vision' if module is self.model.vision_model else 'mlp1',
                                    shape=list(tensor.shape), sha256=tensor_hash(tensor)))
        self.stack.callback(self.model.vision_model.register_forward_hook(vision).remove)
        self.stack.callback(self.model.mlp1.register_forward_hook(vision).remove)
        return self

    def __exit__(self, *args):
        self.stack.close()


def architecture(model, tokenizer):
    layers = model.language_model.model.layers
    assert len(layers) == model.config.llm_config.num_hidden_layers
    modules = dict(model.named_modules())
    records = []
    for index, layer in enumerate(layers):
        path = f'language_model.model.layers.{index}'
        assert modules[path] is layer
        projections = {}
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            module_path = path+'.self_attn.'+name
            module = model.get_submodule(module_path)
            assert module is getattr(layer.self_attn, name)
            projections[name] = dict(path=module_path, shape=list(module.weight.shape))
        record = dict(layer=index, path=path, attention=path+'.self_attn', projections=projections,
            input_layernorm=path+'.input_layernorm', post_attention_layernorm=path+'.post_attention_layernorm', mlp=path+'.mlp')
        records.append(record)
        print('DECODER', path, 'o_proj', projections['o_proj']['path'], flush=True)
    config = model.language_model.config
    vision_layers = model.vision_model.encoder.layers
    assert len(vision_layers) == model.config.vision_config.num_hidden_layers
    return dict(model_class=type(model).__name__, parameter_count=sum(p.numel() for p in model.parameters()),
        vision=dict(path='vision_model', layers_path='vision_model.encoder.layers', layer_count=len(vision_layers),
                    parameter_count=sum(p.numel() for p in model.vision_model.parameters()),
                    class_name=type(model.vision_model).__name__, patch_size=model.patch_size,
                    use_flash_attn=[layer.attn.use_flash_attn for layer in vision_layers]),
        language=dict(path='language_model',class_name=type(model.language_model).__name__,layer_count=len(layers),
            parameter_count=sum(p.numel() for p in model.language_model.parameters()),hidden_size=config.hidden_size,
            query_heads=config.num_attention_heads,kv_heads=config.num_key_value_heads,
            grouped_query_attention=config.num_attention_heads != config.num_key_value_heads,
            head_dimension=layers[0].self_attn.head_dim,attention_implementation=config._attn_implementation, layers=records),
        injection='Vision CLS removed -> pixel_shuffle(v2, downsample_ratio=0.5) -> mlp1 -> replace IMG_CONTEXT embeddings before language layer0; no DeepStack',
        pixel_shuffle_method='pixel_shuffle',projection_path='mlp1',projection=str(model.mlp1),
        image_size=model.config.force_image_size,downsample_ratio=model.downsample_ratio,
        image_tokens_per_tile=model.num_image_token,template=model.template,
        special_token_ids={token:tokenizer.convert_tokens_to_ids(token) for token in ('<img>','</img>','<IMG_CONTEXT>','<|im_start|>','<|im_end|>','<|endoftext|>')},
        tokenizer_bos=tokenizer.bos_token_id,tokenizer_eos=tokenizer.eos_token_id,
        configured_language_bos=config.bos_token_id,configured_language_eos=config.eos_token_id,
        official_chat_eos=tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip()),
        stopping='Official chat overrides eos_token_id with actual conversation separator; greedy max_new_tokens=32')


def expanded_prompt(model, sample, tiles):
    template = model.conv_template.copy()
    template.system_message = model.system_message
    template.append_message(template.roles[0], sample['request'])
    template.append_message(template.roles[1], None)
    return template.get_prompt().replace('<image>', '<img>'+'<IMG_CONTEXT>'*(model.num_image_token*tiles)+'</img>', 1)


@torch.no_grad()
def main():
    from transformers import AutoModel, AutoTokenizer
    protocol = read(OUT/'protocol.json')
    assert read(OUT/'download_validation.json')['passed'], 'No GPU before complete download validation'
    assert read(OUT/'remote_code_review.json')['passed'] and read(OUT/'cpu_tests.json')['passed']
    assert all(sha256(ROOT/path) == value for path, value in protocol['source_sha256'].items())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and torch.cuda.device_count() == 1
    processes = command('nvidia-smi','-i','1','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits')
    assert not processes, f'GPU1 busy; stop without touching other GPUs: {processes}'
    uuid = command('nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader')
    write(OUT/'gpu1_preflight.json',dict(snapshot=command('nvidia-smi'),uuid=uuid,physical_gpu=1,
        visible_devices='1',process_device='cuda:0',pid=os.getpid()))
    directory = Path(protocol['local_path'])
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=False, local_files_only=True)
    fast = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
    RAW.mkdir(parents=True, exist_ok=True)
    model = None
    status = dict(passed=False,scope=protocol['scope'],errors=[])
    start = time.perf_counter()
    try:
        with ResourceMonitor() as monitor:
            model = AutoModel.from_pretrained(directory, trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=False,
                device_map={'':'cuda:0'}, use_safetensors=True).eval()
            assert {str(p.device) for p in model.parameters()} == {'cuda:0'}
            assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
            assert {str(b.device) for b in model.buffers()} == {'cuda:0'}
            assert model.language_model.config._attn_implementation == 'eager'
            assert not any(layer.attn.use_flash_attn for layer in model.vision_model.encoder.layers)
            mapping = command('nvidia-smi','-i','1','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
            assert any(line.split(',')[0].strip() == str(os.getpid()) and uuid in line for line in mapping.splitlines())
            versions = [(name,p._version,p.data_ptr()) for name,p in model.named_parameters()]
            baseline_hooks = hooks_count(model)
            audit = architecture(model,tokenizer)
            write(OUT/'architecture_audit.json',audit)
            load_image,_ = official_preprocessor(directory)
            native, layouts, replays = [], [], []
            for case, sample in enumerate(read(OUT/'sample_manifest.json')['samples']):
                monitor.stage = f'chat_{case}_max{sample["max_num"]}'
                assert sha256(sample['image_path']) == sample['image_sha256']
                pixels = load_image(sample['image_path'], max_num=sample['max_num']).to('cuda:0', dtype=torch.bfloat16)
                before_pixels = tensor_hash(pixels)
                began = time.perf_counter()
                with ChatCapture(model) as capture:
                    answer = model.chat(tokenizer,pixels,sample['request'],
                        dict(do_sample=False,num_beams=1,max_new_tokens=32))
                assert tensor_hash(pixels) == before_pixels
                assert hooks_count(model) == baseline_hooks
                prompt = expanded_prompt(model,sample,pixels.shape[0])
                ids = capture.input_ids[0].tolist()
                assert tokenizer(prompt)['input_ids'] == ids
                layout = partitions(tokenizer,fast,prompt,sample['question'],ids,model.img_context_token_id)
                assert layout['image_tokens'] == pixels.shape[0]*model.num_image_token
                assert int(capture.first_logits.argmax()) == capture.generated_ids[0]
                assert capture.vision[0]['module'] == 'vision' and capture.vision[1]['module'] == 'mlp1' and len(capture.vision) == 2
                assert len(capture.calls) == len(capture.generated_ids)
                assert capture.calls[0]['query_length'] == len(ids)
                assert all(step['query_length'] == 1 and step['cache_length'] == len(ids)+index
                    for index,step in enumerate(capture.calls[1:],1))
                eos_id = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
                assert answer.strip(), 'Empty answer'
                key = f'{sample["task"]}__{sample["question_id"]}__max{sample["max_num"]}'
                raw = RAW / (key+'.safetensors')
                save_file(dict(first_logits=capture.first_logits,inputs_embeds=capture.saved['inputs_embeds']),str(raw))
                row = dict(task=sample['task'],question_id=sample['question_id'],image_sha256=sample['image_sha256'],
                    max_num=sample['max_num'],tiles=int(pixels.shape[0]),thumbnail_added=pixels.shape[0]>1,
                    input_length=len(ids),image_tokens=layout['image_tokens'],answer=answer,
                    generated_token_ids=capture.generated_ids,first_token=capture.generated_ids[0],
                    eos=capture.generated_ids[-1] == eos_id,truncated=len(capture.generated_ids)==32 and capture.generated_ids[-1] != eos_id,
                    seconds=time.perf_counter()-began,cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                    vision_trace=capture.vision,cache_trace=capture.calls,prefill_cache=capture.prefill_cache,
                    first_logits_sha256=tensor_hash(capture.first_logits),raw_path=str(raw),raw_sha256=sha256(raw))
                native.append(row)
                layouts.append(dict(question_id=sample['question_id'],prompt=prompt,input_ids=ids,**layout))
                if case < 2:
                    replays.append((capture.saved,capture.first_logits,layout))
                print(f'CHAT {key} tiles={row["tiles"]} visual={row["image_tokens"]} answer={answer!r} eos={row["eos"]}',flush=True)
                write(OUT/'smoke_generations.json',dict(passed=True,cases=native,not_an_accuracy_benchmark=True))
                write(OUT/'token_layout_audit.json',dict(passed=True,cases=layouts))
                del pixels,capture
            checks = []
            for case,(saved,native_logits,layout) in enumerate(replays):
                monitor.stage = f'hook_{case}'
                input_sha = {k:tensor_hash(v) for k,v in saved.items() if isinstance(v,torch.Tensor)}
                full,reference,cache,_ = replay(model,saved)
                assert torch.equal(full,native_logits), 'Replay must exactly match actual official native chat first logits'
                noop,_,_,_ = replay(model,saved,layer=2,noop=True)
                assert torch.equal(full,noop), 'No-op logits must be exact'
                interventions=[]
                for layer in (2,len(model.language_model.model.layers)-1):
                    off,trace,off_cache,_ = replay(model,saved,layer=layer)
                    for index,(a,b) in enumerate(zip(trace,reference)):
                        assert a['attention_prefix_sha256'] == b['attention_prefix_sha256']
                        assert a['output_prefix_sha256'] == b['output_prefix_sha256']
                        assert torch.count_nonzero(a['attention']) == 0 if index == layer else torch.count_nonzero(a['attention']) > 0
                    interventions.append(dict(layer=layer,target_row_zero=True,all_other_prefill_queries_exact=True,
                        all_mlp_and_residual_additions_valid=True,cache=off_cache,
                        first_token=int(off.argmax()),native_first_token=int(full.argmax()),
                        max_logit_difference=float((full.float()-off.float()).abs().max()),first_logits_sha256=tensor_hash(off)))
                checks.append(dict(case=case,native_chat_replay_max_abs=0.,noop_max_abs=0.,interventions=interventions,
                    native_cache=cache,input_embeddings_unchanged=True,vision_reused_from_unchanged_native_embeddings=True))
                assert input_sha == {k:tensor_hash(v) for k,v in saved.items() if isinstance(v,torch.Tensor)}
                if case == 0:
                    observed,_,_,observer = replay(model,saved,layout=layout)
                    assert torch.equal(full,observed)
                    assert sorted(observer.rows) == [0,len(model.language_model.model.layers)-1]
                    write(OUT/'attention_mass_smoke.json',dict(passed=True,layers=sorted(observer.rows),
                        source_order=['anchor','image','question','template','self'],head_masses=observer.rows,
                        row_sum_max_errors=observer.errors,noop_logits_max_abs=0.,
                        row_only=True,full_attention_matrices_saved=False))
            from .engine import P1OutputHook,MassObserver
            try:
                with P1OutputHook(model.language_model.model.layers[2].self_attn.o_proj,layout['p_minus_1']):
                    raise RuntimeError('intentional hook cleanup')
            except RuntimeError as error:
                assert str(error) == 'intentional hook cleanup'
            import transformers.models.qwen2.modeling_qwen2 as qwen
            original = qwen.eager_attention_forward
            try:
                with MassObserver({0:model.language_model.model.layers[0].self_attn},layout):
                    raise RuntimeError('intentional observer cleanup')
            except RuntimeError as error:
                assert str(error) == 'intentional observer cleanup'
            assert qwen.eager_attention_forward is original and hooks_count(model) == baseline_hooks
            assert versions == [(name,p._version,p.data_ptr()) for name,p in model.named_parameters()]
            write(OUT/'attention_hook_audit.json',dict(passed=True,cases=checks,normal_and_exception_cleanup=True,
                model_weights_and_parameter_storage_unchanged=True,vision_unchanged=True,decode_intervention=False))
            status.update(passed=True,no_nan_inf_oom=True,official_chat_generations=4,
                hook_cases=2,all_model_parameters_bf16_single_gpu=True,seconds=time.perf_counter()-start,
                physical_gpu=1,uuid=uuid,process_device='cuda:0',source_frozen=True)
    except BaseException:
        status['errors'].append(traceback.format_exc())
        raise
    finally:
        write(OUT/'validation.json',status)
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        print('SMOKE finished; process exits and releases GPU1',flush=True)


if __name__ == '__main__':
    main()
