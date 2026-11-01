"""Check pinned APIs without weights, network or training; CUDA probe is opt-in."""
import argparse
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path


def check_stack(cuda=False):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    pins = {}
    for line in (root / 'requirements-neral-training.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name, expected = line.split('==')
            actual = metadata.version(name)
            # CPU-only API evidence is explicitly allowed, not a CUDA certification.
            allowed = {expected} if cuda or name != 'torch' else {expected, '2.8.0+cpu'}
            if actual not in allowed:
                raise RuntimeError(f'{name}: expected {allowed}, found {actual}')
            pins[name] = actual
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel
    from neral_adapter_candidate import TARGETS, seal_adapter, validate_adapter
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from tempfile import TemporaryDirectory
    # Construct interfaces only. NO forward/backward, optimizer step or trainer.train.
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'hello': 3}, unk_token='[UNK]')),
        unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=4, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0,
                          target_modules=sorted(TARGETS), task_type='CAUSAL_LM'))
    rows = [{'input_ids': [3, 3, 2], 'completion_mask': [0, 1, 1]}]
    with TemporaryDirectory(prefix='neral-api-check-') as tmp:
        config = SFTConfig(output_dir=tmp, use_cpu=True, bf16=False, fp16=False,
            max_length=512, max_steps=1, report_to='none', packing=False,
            completion_only_loss=True, dataset_kwargs={'skip_prepare_dataset': True})
        trainer = SFTTrainer(model=model, args=config, processing_class=tok,
                             train_dataset=Dataset.from_list(rows))
        batch = trainer.data_collator(rows)
        assert batch['labels'].tolist() == [[-100, 3, 2]]
        assert any(p.requires_grad for p in model.parameters())
        artifact = Path(tmp) / 'adapter'
        model.save_pretrained(artifact, safe_serialization=True)
        base = {'revision': 'a'*40, 'sha256': 'b'*64}
        seal_adapter(artifact, base, 'c'*64, '0.1.0-pilot.1')
        validate_adapter(artifact, base)
        fresh = Qwen2ForCausalLM(model.get_base_model().config)
        loaded = PeftModel.from_pretrained(fresh, str(artifact), is_trainable=False, local_files_only=True)
        assert not any(p.requires_grad for p in loaded.parameters())
    # Verify padding is ignored while EOS remains supervised.
    batch = DataCollatorForLanguageModeling(pad_token_id=1, completion_only_loss=True)(
        rows + [{'input_ids': [3, 2], 'completion_mask': [0, 1]}])
    assert batch['labels'].tolist() == [[-100, 3, 2], [-100, 2, -100]]
    report = {'platform': platform.platform(), 'python': platform.python_version(),
        'packages': pins, 'api_and_completion_mask': 'passed', 'training_executed': False,
        'tiny_random_adapter_roundtrip': 'passed; temporary CPU fixture, no NERAL training',
        'cuda_checked': False}
    if cuda:
        import bitsandbytes as bnb
        if not torch.cuda.is_available() or torch.version.cuda != '12.8' or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Expected CUDA 12.8 wheel and BF16-capable GPU.')
        device = torch.cuda.get_device_properties(0)
        if '4070' not in device.name or device.total_memory < 11 * 1024**3:
            raise RuntimeError('RTX 4070 with 12 GB required for this acceptance profile.')
        with torch.inference_mode():
            matrix = torch.randn(64, 64, device='cuda:0', dtype=torch.bfloat16)
            assert torch.isfinite(matrix @ matrix).all()
            quant, state = bnb.functional.quantize_4bit(matrix, quant_type='nf4', compress_statistics=True)
            restored = bnb.functional.dequantize_4bit(quant, state)
            assert torch.isfinite(restored).all()
            layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                                     quant_type='nf4', compress_statistics=True).to('cuda:0')
            assert torch.isfinite(layer(matrix)).all()
        report.update(cuda_checked=True, gpu=device.name, total_vram_bytes=device.total_memory,
                      torch_cuda=torch.version.cuda, cuda_probe='BF16/NF4 forward passed; backward unverified')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-cuda', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_stack(args.check_cuda), indent=2))
