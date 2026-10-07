"""Fine-tune Qwen or a FUSAR-GPT adapter using images and text, without TLM/AEF."""
import argparse
import json
import os
from pathlib import Path

from lora_utils import resolve_lora_parameters


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--init-adapter", help="Optional local Stage 1 or image/text adapter directory")
    p.add_argument("--lora-rank", type=int,
                   help="New adapter rank (default: 8); must match an existing adapter")
    p.add_argument("--lora-alpha", type=int,
                   help="New adapter alpha (default: 32); must match an existing adapter")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--image-size", type=int, default=504)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def create_or_load_adapter(base, init_adapter=None, lora_rank=None, lora_alpha=None):
    lora_rank, lora_alpha = resolve_lora_parameters(init_adapter, lora_rank, lora_alpha)
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if init_adapter:
        # Continue the saved adapter, including its original configuration.
        model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
    else:
        suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        targets = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)
                   and n.endswith(suffixes) and "visual" not in n.split(".")]
        if not targets:
            raise ValueError("No language LoRA target modules found")
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                               target_modules=targets, r=lora_rank, lora_alpha=lora_alpha,
                               lora_dropout=0.05, bias="none"))
    model.config.use_cache = False
    return model


def make_trainer_class(processor, image_size):
    from transformers import Trainer

    class ImageTextTrainer(Trainer):
        def save_model(self, output_dir=None, _internal_call=False):
            if not self.is_world_process_zero():
                return
            output = Path(output_dir or self.args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            model.save_pretrained(output, safe_serialization=True)
            processor.save_pretrained(output)
            config_path = output / "adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            (output / "image_text_config.json").write_text(
                json.dumps({"image_size": image_size, "use_tlm": False}, indent=2) + "\n", encoding="utf-8")

    return ImageTextTrainer


def make_training_arguments(args):
    import torch
    from transformers import TrainingArguments

    result = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, warmup_ratio=0.05, optim="adamw_torch",
        bf16=args.dtype == "bfloat16", gradient_checkpointing=False,
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        ddp_backend=("nccl" if torch.cuda.is_available() else "gloo")
                    if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
        save_strategy="epoch", save_total_limit=1, logging_steps=1,
        dataloader_num_workers=0, report_to="none", seed=args.seed,
    )
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        result._n_gpu = 1
    return result


def main():
    args = parse_args()
    if args.grad_accum < 1 or args.epochs <= 0 or args.lr <= 0:
        raise ValueError("grad-accum, epochs and lr must be positive")
    if args.image_size <= 0 or args.image_size % 28:
        raise ValueError("image-size must be a positive multiple of 28")
    output = Path(args.output_dir).resolve()
    if args.init_adapter and output == Path(args.init_adapter).resolve():
        raise ValueError("Use a separate output directory to preserve the initial adapter")
    if any((output / name).exists() for name in ("vec64_film.pt", "tlm_config.json")):
        raise ValueError("Use an output directory without existing TLM weights")
    lora_rank, lora_alpha = resolve_lora_parameters(args.init_adapter, args.lora_rank, args.lora_alpha)
    print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
    from data_no_aef import ImageTextDataset, ImageTextCollator

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    if args.dtype == "bfloat16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training needs a supported CUDA GPU; use --dtype float32 for CPU tests")
    set_seed(args.seed)
    source = args.init_adapter if args.init_adapter and (Path(args.init_adapter) / "preprocessor_config.json").exists() else args.base_model
    processor = AutoProcessor.from_pretrained(source)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention)
    model = create_or_load_adapter(base, args.init_adapter, lora_rank, lora_alpha)
    dataset = ImageTextDataset(args.train_json, args.image_size)
    collator = ImageTextCollator(processor, args.image_size, args.max_length)
    trainer = make_trainer_class(processor, args.image_size)(
        model=model, args=make_training_arguments(args), train_dataset=dataset, data_collator=collator)
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(args.output_dir)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        print(f"Saved image/text adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
