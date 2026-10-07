"""Image/text data for fine-tuning and inference without AEF."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data import build_prompt, read_records


class ImageTextDataset(Dataset):
    def __init__(self, manifest, image_size=504, require_answer=True):
        self.root = Path(manifest).resolve().parent
        self.items = read_records(manifest)
        if not self.items:
            raise ValueError("Empty manifest")
        for item in self.items:
            for key in ("image", "question"):
                if key not in item:
                    raise ValueError(f"Manifest record requires {key}")
            if require_answer and "answer" not in item:
                raise ValueError("Training requires answer")
            if item.get("coordinate_size", image_size) != image_size:
                raise ValueError("Manifest coordinates do not match image_size; convert annotations explicitly")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = dict(self.items[index])
        item.pop("aef", None)
        item["image"] = str((self.root / item["image"]).resolve())
        return item


class ImageTextCollator:
    def __init__(self, processor, image_size=504, max_length=2048, training=True):
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length
        self.training = training

    def __call__(self, items):
        if len(items) != 1:
            raise ValueError("Use one image per GPU; use gradient accumulation or torchrun for training")
        item = items[0]
        batch = build_prompt(self.processor, item, self.image_size)
        prompt = batch["input_ids"][0][batch["attention_mask"][0].bool()]
        sequence = prompt
        if self.training:
            answer = item["answer"]
            if not isinstance(answer, str):
                answer = json.dumps(answer, ensure_ascii=False)
            tok = self.processor.tokenizer
            response = torch.tensor(tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id],
                                    dtype=torch.long)
            sequence = torch.cat([prompt, response])
            batch["labels"] = torch.cat([torch.full_like(prompt, -100), response]).unsqueeze(0)
        if len(sequence) > self.max_length:
            raise ValueError(f"Sequence exceeds max_length={self.max_length}; increase it instead of truncating boxes")
        batch["input_ids"] = sequence.unsqueeze(0)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        return dict(batch)


def image_size_for_inference(adapter_dir, requested_size=None):
    config = Path(adapter_dir) / "image_text_config.json" if adapter_dir else None
    saved_size = None
    if config is not None and config.exists():
        saved_size = json.loads(config.read_text(encoding="utf-8"))["image_size"]
    if requested_size is not None and saved_size is not None and requested_size != saved_size:
        raise ValueError("image-size differs from the fine-tuning configuration")
    size = requested_size if requested_size is not None else (saved_size or 504)
    if size <= 0 or size % 28:
        raise ValueError("image-size must be a positive multiple of 28")
    return size
