"""Detection inference using the ORIGINAL TRAINING TLM computation."""
import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", required=True)
    p.add_argument("--adapters-dir", required=True)
    p.add_argument("--save-pred-jsonl", default="outputs/predictions.jsonl")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--legacy-sigma", type=float, help="Required for an old checkpoint without TLM config; original training default=2")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=4096, help="Maximum prompt length")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--overlay-dir", help="Optionally draw parsed predictions on 504x504 images")
    return p.parse_args()


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use python for inference; device_map='auto' distributes one model across visible GPUs")
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from data import TripletDataset, TripletCollator, parse_detections
    from tlm import TLMModel
    from PIL import Image, ImageDraw

    if args.dtype == "bfloat16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires a supported CUDA GPU; use --dtype float32 for CPU tests")
    source = args.adapters_dir if (Path(args.adapters_dir) / "preprocessor_config.json").exists() else args.base_model
    processor = AutoProcessor.from_pretrained(source)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention,
        device_map="auto", low_cpu_mem_usage=True
    )
    base = PeftModel.from_pretrained(base, args.adapters_dir)
    wrapped = TLMModel.from_adapter(base, args.adapters_dir, legacy_sigma=args.legacy_sigma).eval()
    embedding = base.get_input_embeddings()
    device = getattr(getattr(embedding, "_hf_hook", None), "execution_device", None)
    if device is None:
        device = embedding.weight.device
    if torch.device(device).type == "meta":
        raise RuntimeError("Cannot determine the dispatched embedding execution device")
    dataset = TripletDataset(args.input_json, wrapped.image_size, require_answer=False)
    collator = TripletCollator(processor, wrapped.image_size, args.max_length, training=False)
    out_path = Path(args.save_pred_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        Path(args.overlay_dir).mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output, torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            # The TLM hook places AEF on the visual-output device, independently
            # of the text embedding device selected for model inputs.
            batch = {k: v if k.startswith("vec64_") else v.to(device)
                     for k, v in collator([item]).items()}
            prompt_length = batch["input_ids"].shape[1]
            generated = wrapped.generate(**batch, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, use_cache=True)
            # Decode ONLY new tokens. Preserve multiline JSON and every coordinate.
            raw = processor.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
            error = None
            try:
                detections = parse_detections(raw)
            except (ValueError, TypeError) as exc:
                detections, error = None, str(exc)
            row = {"id": item.get("id", index), "coordinate_size": wrapped.image_size,
                   "question": item["question"], "gt": item.get("answer"),
                   "prediction_text": raw, "detections": detections, "parse_error": error}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if args.overlay_dir and detections is not None:
                with Image.open(item["image"]) as img:
                    img = img.convert("RGB").resize((wrapped.image_size,) * 2, Image.Resampling.BICUBIC)
                draw = ImageDraw.Draw(img)
                for obj in detections:
                    draw.rectangle(obj["bbox_2d"], outline="red", width=2)
                    draw.text(tuple(obj["bbox_2d"][:2]), obj["label"], fill="yellow")
                img.save(Path(args.overlay_dir) / f"prediction_{index:03d}.png")
            print(f"{index + 1}/{len(dataset)}: {'JSON parsed' if error is None else error}")
    print(f"Saved {out_path}; no classification-style exact-string accuracy is computed")


if __name__ == "__main__":
    main()
