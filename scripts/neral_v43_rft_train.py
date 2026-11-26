"""Explicit local RFT QLoRA pilot. CPU check by default; GPU path gated separately.

No automatic workflow, no default training, no deployment, no old adapter load.
The default CPU path never imports model libraries or stops the service.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, socket, subprocess, threading, time
from pathlib import Path
from scripts.neral_v43_rft_cpu import read_pinned, rows, safe, output_path, protected_hashes, validate_rows, mask, plan, canonical

PIN='5877b73851c986f39f4dbeb470cd8fb08bd2707d'
APPROVAL='V43-RFT-QLORA-SMOKE-ONCE'
CHECKPOINT=Path('D:/neral-qlora/checkpoint')

def load_bundle(bundle,manifest_sha,protected,protected_sha,context):
    manifest=json.loads(read_pinned(bundle/'manifest.json',manifest_sha))
    if manifest['protected_manifest_sha256']!=protected_sha:raise ValueError('protected manifest mismatch')
    deny=protected_hashes(json.loads(read_pinned(protected,protected_sha)))
    items=rows(read_pinned(bundle/'train.jsonl',manifest['train_sha256']))
    version=manifest.get('catalog_version','v1')
    if version not in ('v1','v2') or (version=='v2' and len(items)!=200):
        raise ValueError('unknown or invalid catalog version')
    if len(items)==200:
        from scripts.neral_v43_rft_pilot import holdout_hashes, audit_pilot
        if not manifest.get('development_holdout_path') or not manifest.get('development_holdout_sha256'):
            raise ValueError('pinned independent development holdout required')
        holdout=rows(read_pinned(manifest['development_holdout_path'],manifest['development_holdout_sha256']))
        held=holdout_hashes(holdout,deny)
        if version=='v2':
            from scripts.neral_v44_rft_quality import audit_pinned_quality
            audit_pinned_quality(manifest,items,holdout,deny)
        deny.update(held)
        audit_pilot(items)
    items=validate_rows(items,deny)
    config=plan(len(items),context)
    replay=[x for x in items if x['meta']['kind']=='replay']
    rft=[x for x in items if x['meta']['kind']=='rft']
    if len(replay)!=config['replay_rows'] or {x['meta']['category'] for x in replay}!={'everyday','knowledge','math','code'} or not any(x['meta'].get('repair') is True for x in rft):raise ValueError('replay or repair gate')
    if any(x['meta'].get('reviewed') is not True or x['meta'].get('source')!='independent_replay' for x in replay):raise ValueError('replay provenance')
    if any(x['meta'].get('source')!='fresh_catalog' or x['meta'].get('proof') not in {'sympy','z3','code'} for x in rft):raise ValueError('RFT provenance')
    return items,config,manifest

def checkpoint_check(checkpoint,manifest,sha):
    # Owner-pinned complete snapshot including tokenizer and weight shards.
    expected=json.loads(read_pinned(manifest,sha)); root=safe(checkpoint)
    actual={x.relative_to(root).as_posix() for x in root.rglob('*') if x.is_file()}
    if set(expected)!=actual or not {'config.json','tokenizer.json','model.safetensors.index.json'}<=actual:raise ValueError('checkpoint inventory mismatch')
    if any(x.startswith('adapter') or '/adapter' in x for x in actual):raise ValueError('adapter not allowed')
    for name,hash_expected in expected.items():
        path=safe(root/name)
        if not path.is_relative_to(root) or not isinstance(hash_expected,str):raise ValueError('checkpoint path')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=hash_expected:raise ValueError('checkpoint SHA mismatch')

def gpu_sample():
    sample=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
    if len(sample)!=1:raise RuntimeError('single GPU required')
    name,free,used,total=(s.strip() for s in sample[0].split(','))
    if 'RTX 4070' not in name or 'Ti' in name or int(total)<11000:raise RuntimeError('unexpected GPU')
    return int(free),int(used),int(total)

def owner_stopped():
    if os.name!='nt' or os.environ.get('COMPUTERNAME','').upper()!='W39-GAMINGPC':raise RuntimeError('wrong training computer')
    task=subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',"(Get-ScheduledTask -TaskName 'NERAL-PR558-ThreeTier-909d1419c9ff' -ErrorAction Stop).State"],text=True,timeout=15).strip()
    if task!='Ready':raise RuntimeError('owner not stopped')
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(('127.0.0.1',9690))==0:raise RuntimeError('owner port still listening')
    free,used,total=gpu_sample()
    if free<10240 or used>10752:raise RuntimeError('insufficient preload VRAM')
    return free,used,total

class Watch:
    def __init__(self,cap=10752):self.cap=cap;self.peak=0;self.samples=0;self.failed=False;self.stop=threading.Event()
    def run(self):
        while not self.stop.is_set():
            try:
                _,used,_=gpu_sample();self.samples+=1;self.peak=max(self.peak,used)
                if used>self.cap:self.failed=True
            except Exception:self.failed=True
            if self.stop.wait(.5):break
    def check(self):
        if self.failed or not self.samples:raise RuntimeError('VRAM monitor gate failed')


class SmokeTelemetry:
    """One step only: first microbatch token-mean CE and synchronized wall time.

    No extra model forward pass. The first loss may already be normalized by
    the full accumulation-window token count, so undo that normalization.
    """
    def __init__(self,clock=time.monotonic):
        self.clock=clock
        self.started=None
        self.finished=None
        self.start_loss=None
        self.substeps=0
    def begin(self):
        if self.started is not None:raise RuntimeError('unexpected additional step')
        self.started=self.clock()
    def record_first_loss(self,loss,supervised_tokens,normalizer):
        if self.started is None or self.finished is not None or self.start_loss is not None:
            raise RuntimeError('unexpected first loss')
        if (not math.isfinite(loss) or loss<0 or type(supervised_tokens) is not int
                or supervised_tokens<=0 or not math.isfinite(normalizer) or normalizer<supervised_tokens):
            raise ValueError('invalid initial loss normalization')
        value=loss*normalizer/supervised_tokens
        if not math.isfinite(value):raise ValueError('nonfinite initial loss')
        self.start_loss=value
    def substep(self):
        if self.started is None or self.finished is not None or self.substeps>=9:
            raise RuntimeError('unexpected accumulation substep')
        self.substeps+=1
    def end(self):
        if self.started is None or self.finished is not None or self.substeps!=9 or self.start_loss is None:
            raise RuntimeError('incomplete one-step telemetry')
        end=self.clock()
        if not math.isfinite(end-self.started) or end<self.started:raise RuntimeError('invalid step duration')
        self.finished=end
    def report(self):
        if self.finished is None:raise RuntimeError('missing measured step')
        return {'start_loss':self.start_loss,
                'start_loss_definition':'first_pre_update_microbatch_mean_supervised_token_ce',
                'step_duration_s':self.finished-self.started,
                'gradient_accumulation_microbatches':self.substeps+1}


def tokenizer_check(items,checkpoint,context):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,use_fast=True,trust_remote_code=False)
    encoded=[mask(x,tok,context) for x in items]
    if tok.pad_token_id is None:raise ValueError('missing pad token')
    return tok,encoded


def sft_settings(output_dir, *, cpu=False, bf16=False):
    from trl import SFTConfig
    return SFTConfig(
        output_dir=str(output_dir),use_cpu=cpu,per_device_train_batch_size=1,
        gradient_accumulation_steps=10,max_steps=1,num_train_epochs=1,
        learning_rate=5e-5,warmup_steps=0,optim='adamw_torch' if cpu else 'paged_adamw_8bit',
        bf16=bf16 and not cpu,fp16=not cpu and not bf16,
        gradient_checkpointing=not cpu,gradient_checkpointing_kwargs={'use_reentrant':False},
        save_strategy='no',logging_steps=1,report_to='none',remove_unused_columns=False,
        packing=False,dataset_kwargs={'skip_prepare_dataset':True},
        assistant_only_loss=False,completion_only_loss=False,seed=20261124,
    )


def masked_trainer(trainer_class,model,settings,encoded,tokenizer,collator,callbacks=()):
    from datasets import Dataset
    trainer=trainer_class(model=model,args=settings,train_dataset=Dataset.from_list(encoded),
                          processing_class=tokenizer,data_collator=collator,callbacks=list(callbacks))
    if len(trainer.train_dataset)!=len(encoded):
        raise RuntimeError('TRL altered dataset length')
    for expected,actual in zip(encoded,trainer.train_dataset):
        if any(actual.get(key)!=expected[key] for key in ('input_ids','labels','attention_mask')):
            raise RuntimeError('TRL altered masked training input')
    return trainer

def train_smoke(args,items,config,manifest):
    # Deliberately smoke-only: full training needs a new approval and proof.
    if args.confirm!=APPROVAL or not args.out or not args.owner_receipt or not args.owner_receipt_sha:raise ValueError('explicit smoke and StopOwned receipt required')
    if len(items)!=200:raise ValueError('GPU pilot requires exactly 200 audited training rows')
    out=output_path(args.out)
    if out.exists():raise FileExistsError('one-shot output already exists')
    receipt=json.loads(read_pinned(args.owner_receipt,args.owner_receipt_sha))
    if receipt.get('approval')!=APPROVAL or receipt.get('original_http_status')!=200 or receipt.get('stop_owned') is not True or receipt.get('checkpoint_manifest_sha256')!=args.checkpoint_sha or receipt.get('train_sha256')!=manifest['train_sha256']:raise ValueError('StopOwned receipt mismatch')
    # Checkpoint identity, native tokenizer and full corpus before CUDA.
    checkpoint_check(args.checkpoint,args.checkpoint_manifest,args.checkpoint_sha)
    tok,encoded=tokenizer_check(items,args.checkpoint,args.context)
    free,used,total=owner_stopped()
    import torch
    from transformers import Qwen3_5ForCausalLM,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from trl import SFTTrainer
    from scripts.neral_qlora_train_peft import create_config,prepare_memory_bounded,answer_only_loss,adapter_fingerprint
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('CUDA identity')
    torch.cuda.set_per_process_memory_fraction(min(.95,9*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats();set_seed(20261124)
    watch=Watch();thread=threading.Thread(target=watch.run,daemon=True);thread.start()
    out.mkdir(parents=True,exist_ok=False)
    try:
        model=Qwen3_5ForCausalLM.from_pretrained(str(args.checkpoint),local_files_only=True,trust_remote_code=False,quantization_config=create_config('Qwen/Qwen3.5-9B'),dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,device_map={'':0},low_cpu_mem_usage=True)
        model.config.use_cache=False
        model=prepare_memory_bounded(model,prepare_model_for_kbit_training)
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
        count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not 0<count<=60_000_000 or gpu_sample()[1]>10752:raise RuntimeError('postload memory or adapter gate')
        watch.check()
        before=adapter_fingerprint(model)
        def collate(xs):
            n=max(len(x['input_ids']) for x in xs)
            return {k:torch.tensor([x[k]+[pad]*(n-len(x[k])) for x in xs],dtype=torch.long) for k,pad in [('input_ids',tok.pad_token_id),('attention_mask',0),('labels',-100)]}
        telemetry=SmokeTelemetry()
        class Guard(TrainerCallback):
            def on_step_begin(self,*a,**k):
                watch.check();torch.cuda.synchronize();telemetry.begin()
            def on_substep_end(self,*a,**k):
                watch.check();telemetry.substep()
            def on_step_end(self,*a,**k):
                torch.cuda.synchronize();watch.check();telemetry.end()
        class AnswerTrainer(SFTTrainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                loss,outputs=answer_only_loss(model,inputs,num_items_in_batch)
                if telemetry.start_loss is None:
                    supervised=int((inputs['labels'][:,1:]!=-100).sum().item())
                    denom=float(num_items_in_batch) if num_items_in_batch is not None else float(supervised)
                    telemetry.record_first_loss(float(loss.detach().item()),supervised,denom)
                return (loss,outputs) if return_outputs else loss
        # Explicitly disable TRL's re-tokenization/packing. Unknown TRL APIs
        # fail closed rather than silently changing labels.
        settings=sft_settings(out/'logs',bf16=torch.cuda.is_bf16_supported())
        trainer=masked_trainer(AnswerTrainer,model,settings,encoded,tok,collate,[Guard()])
        train_started=time.monotonic()
        result=trainer.train();torch.cuda.synchronize();watch.check()
        train_duration=time.monotonic()-train_started
        measured=telemetry.report()
        after=adapter_fingerprint(model)
        grad=[float(x['grad_norm']) for x in trainer.state.log_history if isinstance(x.get('grad_norm'),(int,float))]
        if result.global_step!=1 or before==after or not math.isfinite(result.training_loss) or not grad or not math.isfinite(grad[-1]) or grad[-1]<=0:raise RuntimeError('no finite effective optimizer update')
        model.save_pretrained(str(out/'adapter'),safe_serialization=True);tok.save_pretrained(str(out/'adapter'))
        watch.check()
        report={'status':'smoke_adapter_saved_unevaluated','optimizer_steps':1,'train_loss':float(result.training_loss),'grad_norm':grad[-1],'adapter_before':before,'adapter_after':after,'peak_sampled_mib':watch.peak,'peak_reserved_mib':torch.cuda.max_memory_reserved()//2**20,'train_sha256':manifest['train_sha256'],'checkpoint_manifest_sha256':args.checkpoint_sha,'preload_free_mib':free,'trainable_parameters':count,'promoted':False,'rollback_required':True}
        report.update(measured);report['training_duration_s']=train_duration
        with (out/'smoke.json').open('x',encoding='utf8') as f:f.write(canonical(report)+'\n')
    finally:
        watch.stop.set();thread.join(7)
    if thread.is_alive() or watch.failed:raise RuntimeError('GPU monitor incomplete')

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--protected',type=Path,required=True);p.add_argument('--protected-sha',required=True);p.add_argument('--context',type=int,default=1024);p.add_argument('--checkpoint',type=Path,default=CHECKPOINT);p.add_argument('--checkpoint-manifest',type=Path);p.add_argument('--checkpoint-sha');p.add_argument('--check-tokenizer',action='store_true');p.add_argument('--run-gpu',action='store_true');p.add_argument('--confirm');p.add_argument('--owner-receipt',type=Path);p.add_argument('--owner-receipt-sha');p.add_argument('--out',type=Path);a=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_DISABLED='true',PYTHONDONTWRITEBYTECODE='1')
    if not a.run_gpu:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    items,config,manifest=load_bundle(a.bundle,a.manifest_sha,a.protected,a.protected_sha,a.context)
    if not a.run_gpu:
        if a.check_tokenizer:
            if not a.checkpoint_manifest or not a.checkpoint_sha:raise ValueError('checkpoint pin required')
            checkpoint_check(a.checkpoint,a.checkpoint_manifest,a.checkpoint_sha)
            _,encoded=tokenizer_check(items,a.checkpoint,a.context)
            config.update(native_tokenizer_checked=True,max_observed_tokens=max(len(x['input_ids']) for x in encoded),supervised_tokens=sum(sum(x!=-100 for x in y['labels']) for y in encoded))
        print(canonical(config));return
    if not a.checkpoint_manifest or not a.checkpoint_sha:raise ValueError('checkpoint pin required')
    train_smoke(a,items,config,manifest)
    load_bundle(a.bundle,a.manifest_sha,a.protected,a.protected_sha,a.context)
if __name__=='__main__':
    try:main()
    except Exception:raise SystemExit('RFT_GATE_FAILED (private details withheld)')
