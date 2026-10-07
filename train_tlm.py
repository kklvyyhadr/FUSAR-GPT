"""TLM training with one image per GPU; launch DDP with torchrun."""
import argparse
import json
import os
from pathlib import Path

from lora_utils import resolve_lora_parameters


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json", required=True, help="Relative-path JSON/JSONL triplet manifest")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--init-adapter", help="Stage-1 output required for Stage 2")
    p.add_argument("--lora-rank", type=int,
                   help="New adapter rank (default: 8); must match an existing adapter")
    p.add_argument("--lora-alpha", type=int,
                   help="New adapter alpha (default: 32); must match an existing adapter")
    p.add_argument("--epochs", type=float, help="Defaults: Stage 1=30, Stage 2=5")
    p.add_argument("--lr", type=float, help="Defaults: Stage 1=1e-4, Stage 2=1e-5")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--sigma", type=float, default=2.0, help="Only used for a new Stage-1 model")
    p.add_argument("--legacy-sigma", type=float,
                   help="Training sigma for an old adapter without tlm_config.json; released Stage 1 uses 2")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Reduce training memory with non-reentrant gradient checkpointing")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def make_trainer_class(processor):
    from transformers import Trainer

    class TLMTrainer(Trainer):
        def save_model(self, output_dir=None, _internal_call=False):
            if not self.is_world_process_zero():
                return
            output = Path(output_dir or self.args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            model.base_model.save_pretrained(output, safe_serialization=True)
            model.save_tlm(output)
            processor.save_pretrained(output)
            config_path = output / "adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return TLMTrainer


def make_training_arguments(args):
    import torch
    from transformers import TrainingArguments

    result = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.epochs is not None else (30 if args.stage == 1 else 5),
        learning_rate=args.lr if args.lr is not None else (1e-4 if args.stage == 1 else 1e-5),
        warmup_ratio=0.05, optim="adamw_torch", bf16=args.dtype == "bfloat16",
        # Checkpointing is enabled on the underlying Qwen below, not the TLM wrapper.
        gradient_checkpointing=False, remove_unused_columns=False,
        ddp_backend=("nccl" if torch.cuda.is_available() else "gloo")
                    if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
        ddp_find_unused_parameters=False,
        save_strategy="epoch", save_total_limit=1, logging_steps=1,
        dataloader_num_workers=0, report_to="none", seed=args.seed,
    )
    # One GPU per process, including plain python on a host with multiple GPUs.
    # DDP is selected by torchrun; do not fall back to nn.DataParallel packing.
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        result._n_gpu = 1
    return result


def main():
    args = parse_args()
    if args.stage == 2 and not args.init_adapter:
        raise ValueError("Stage 2 requires --init-adapter pointing to Stage 1")
    if args.grad_accum < 1:
        raise ValueError("grad-accum must be positive")
    if args.init_adapter:
        if Path(args.output_dir).resolve() == Path(args.init_adapter).resolve():
            raise ValueError("Use a separate output directory to preserve the initial adapter")
        if not (Path(args.init_adapter) / "tlm_config.json").exists() and args.legacy_sigma is None:
            raise ValueError("Legacy checkpoint: supply --legacy-sigma 2 for the released Stage 1")
    lora_rank, lora_alpha = resolve_lora_parameters(args.init_adapter, args.lora_rank, args.lora_alpha)
    print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}")
    import torch
    from peft import PeftModel, LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
    from data import TripletDataset, TripletCollator
    from tlm import TLMModel

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    if args.dtype == "bfloat16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training needs a supported CUDA GPU; use --dtype float32 for CPU tests")
    set_seed(args.seed)
    processor_source = args.init_adapter if args.init_adapter and (Path(args.init_adapter) / "preprocessor_config.json").exists() else args.base_model
    processor = AutoProcessor.from_pretrained(processor_source)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention
    )
    if args.init_adapter:
        base = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        wrapped = TLMModel.from_adapter(base, args.init_adapter, legacy_sigma=args.legacy_sigma)
    else:
        suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        targets = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)
                   and n.endswith(suffixes) and "visual" not in n.split(".")]
        if not targets:
            raise ValueError("No language LoRA target modules found")
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=targets,
                            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none")
        wrapped = TLMModel(get_peft_model(base, config), sigma=args.sigma)
    wrapped.set_stage(args.stage)
    wrapped.base_model.config.use_cache = False
    if args.gradient_checkpointing:
        wrapped.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    dataset = TripletDataset(args.train_json, wrapped.image_size)
    collator = TripletCollator(processor, wrapped.image_size, args.max_length)

    TLMTrainer = make_trainer_class(processor)
    training_args = make_training_arguments(args)
    trainer = TLMTrainer(model=wrapped, args=training_args, train_dataset=dataset,
                         data_collator=collator)
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(args.output_dir)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        print(f"Saved LoRA + TLM + processor to {args.output_dir}")


if __name__ == "__main__":
    main()
