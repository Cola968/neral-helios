"""Opt-in, inference-only Qwen2.5 Coder runtime. No Hub requests or adapters."""
import gc
import time

from model_catalog import HF_BASE, validate_hf_snapshot

FAMILY='qwen2.5-coder'
CONTEXT=2048
MAX_NEW=512
MEMORY_BYTES=int(7.5*1024**3)
SYSTEM='You are NERAL, a local coding assistant based on Qwen2.5-Coder. Be precise; state uncertainty.'
# Use our audited template, never a template or executable code from an import.
CHAT_TEMPLATE="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


class LocalQwenError(ValueError):
    pass


class Runtime:
    def __init__(self,model,tokenizer,identity):
        self.model,self.tokenizer=model,tokenizer
        self.sha256=identity
        self.identity=FAMILY+':'+identity
        self.finish_reason=None

    def encode(self,history,prompt,max_new):
        if not 1<=max_new<=MAX_NEW:raise LocalQwenError('Ungültiges Antwortbudget.')
        if any(h.get('model_family')!=FAMILY or h.get('runtime_id')!=self.identity for h in history):
            raise LocalQwenError('Der Verlauf gehört zu einem anderen Modell.')
        messages=[{'role':'system','content':SYSTEM}]
        messages += [{'role':h['role'],'content':h['content']} for h in history]
        messages.append({'role':'user','content':prompt})
        ids=self.tokenizer.apply_chat_template(messages,chat_template=CHAT_TEMPLATE,
                                               tokenize=True,add_generation_prompt=True)
        if not ids or len(ids)+max_new>CONTEXT:
            raise LocalQwenError('Kontext und Antwort überschreiten 2048 Qwen-Token.')
        return ids

    def decode(self,ids):
        return self.tokenizer.decode(ids,skip_special_tokens=True)

    def generate(self,ids,max_new,temperature,top_k,event,on_tokens=None):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        if not 1<=max_new<=MAX_NEW or len(ids)+max_new>CONTEXT:
            raise LocalQwenError('Kontextbudget überschritten.')
        self.finish_reason=None
        if event.is_set():
            self.finish_reason='cancelled'
            return []
        runtime=self
        deadline=time.monotonic()+120
        class Stop(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                if time.monotonic()>=deadline:event.set()
                return event.is_set()
        class Stream:
            def __init__(self):self.prompt=True;self.ids=[]
            def put(self,value):
                if self.prompt:self.prompt=False;return
                self.ids.extend(value.reshape(-1).tolist())
                if on_tokens and (len(self.ids)%4==0 or self.ids[-1]==151645):
                    on_tokens(runtime.decode(self.ids),len(self.ids))
            def end(self):pass
        config=GenerationConfig(max_new_tokens=max_new,do_sample=temperature>0,
                                bos_token_id=151643,eos_token_id=151645,pad_token_id=151643,
                                use_cache=True,num_beams=1,num_return_sequences=1)
        if temperature>0:config.temperature=temperature;config.top_k=top_k
        with torch.inference_mode():
            inputs=torch.tensor([ids],dtype=torch.long,device='cuda:0')
            output=self.model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                                       generation_config=config,streamer=Stream(),
                                       stopping_criteria=StoppingCriteriaList([Stop()]))
            continuation=output[0,len(ids):].tolist()
        self.finish_reason='cancelled' if event.is_set() else 'eos' if continuation and continuation[-1]==151645 else 'length'
        return continuation

    def close(self):
        import torch
        self.model=None
        gc.collect()
        torch.cuda.empty_cache()


def load(folder):
    import torch
    checked=validate_hf_snapshot(folder,verify_hashes=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise LocalQwenError('CUDA mit BF16-Unterstützung ist erforderlich.')
    free,total=torch.cuda.mem_get_info(0)
    if free<MEMORY_BYTES or total<10*1024**3:
        raise LocalQwenError('Zu wenig freier GPU-Speicher. Andere GPU-Modelle zuerst entladen.')
    # Bound this process allocator; CUDA context/other processes are outside it.
    torch.cuda.set_per_process_memory_fraction(MEMORY_BYTES/total,0)
    from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast, BitsAndBytesConfig
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(folder),local_files_only=True,
                                                trust_remote_code=False,token=False)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    candidate=None
    try:
        candidate=Qwen2ForCausalLM.from_pretrained(
            str(folder),local_files_only=True,trust_remote_code=False,token=False,
            use_safetensors=True,quantization_config=quant,torch_dtype=torch.bfloat16,
            device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
        candidate.eval()
        candidate.requires_grad_(False)
        if not getattr(candidate,'is_loaded_in_4bit',False):
            raise LocalQwenError('4-Bit-Laden wurde nicht bestätigt.')
        return Runtime(candidate,tokenizer,checked['sha256'])
    except Exception:
        candidate=None
        gc.collect()
        torch.cuda.empty_cache()
        raise
