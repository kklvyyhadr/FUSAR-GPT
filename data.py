"""Small, explicit image/text/AEF interface. Paths are relative to the manifest."""
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text) if path.suffix == ".json" else [json.loads(s) for s in text.splitlines() if s.strip()]


def load_aef(path):
    """NPZ: features[S,64], positions[S,2] in (y,x), normalized to [0,1].

    Also accepts original NPY structured fields or numeric [S,70] arrays:
    lon,lat,row,col,x,y,A00,...,A63. Matches the original min/max normalization.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)
            positions = data["positions"].astype(np.float32)
    else:
        arr = np.load(path, allow_pickle=False)
        if arr.dtype.names:
            arr = arr.reshape(-1)
            names = arr.dtype.names
            fields = [n for n in names if n not in ("lon", "lat", "row", "col", "x", "y")]
            fields.sort(key=lambda n: (0, int(n[1:])) if re.fullmatch(r"[aA]\d+", n) else (1, names.index(n)))
            features = np.column_stack([arr[n] for n in fields]).astype(np.float32)
            xs, ys = (arr["x"], arr["y"]) if "x" in names and "y" in names else (arr["col"], arr["row"])
        else:
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.size % 70 == 0:
                arr = arr.reshape(-1, 70)
            if arr.ndim != 2 or arr.shape[1] != 70:
                raise ValueError(f"Expected an [S,70] original AEF array: {path}")
            features = arr[:, 6:].astype(np.float32)
            xs, ys = arr[:, 4], arr[:, 5]
        def norm(a):
            a = np.asarray(a, dtype=np.float32)
            return (a - a.min()) / (a.max() - a.min()) if np.ptp(a) > 0 else np.zeros_like(a)
        positions = np.column_stack([norm(ys), norm(xs)]).astype(np.float32)
    if features.ndim != 2 or features.shape[1] != 64 or len(features) == 0:
        raise ValueError(f"AEF features must have shape [S,64] with S > 0: {path}")
    if positions.shape != (len(features), 2):
        raise ValueError(f"AEF positions must have shape [S,2]: {path}")
    if not np.isfinite(features).all() or not np.isfinite(positions).all() or (positions < 0).any() or (positions > 1).any():
        raise ValueError(f"Invalid AEF values or normalized positions: {path}")
    return torch.from_numpy(features.copy()), torch.from_numpy(positions.copy())


class TripletDataset(Dataset):
    def __init__(self, manifest, image_size=504, require_answer=True):
        self.root = Path(manifest).resolve().parent
        self.items = read_records(manifest)
        if not self.items:
            raise ValueError("Empty manifest")
        for item in self.items:
            for key in ("image", "aef", "question"):
                if key not in item:
                    raise ValueError(f"Manifest record requires {key}")
            if require_answer and "answer" not in item:
                raise ValueError("Training requires answer")
            if item.get("coordinate_size", image_size) != image_size:
                raise ValueError("Manifest coordinates do not match model image_size; convert annotations explicitly")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = dict(self.items[index])
        for key in ("image", "aef"):
            item[key] = str((self.root / item[key]).resolve())
        return item


def build_prompt(processor, item, image_size):
    question = item["question"].replace("<image>", "").strip()
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(item["image"]) as img:
        image = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    return processor(text=[text], images=[image], padding=False, return_tensors="pt")


class TripletCollator:
    def __init__(self, processor, image_size=504, max_length=2048, training=True):
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length
        self.training = training

    def __call__(self, items):
        tok = self.processor.tokenizer
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids, labels, pixels, grids, vectors, positions = [], [], [], [], [], []
        for item in items:
            encoded = build_prompt(self.processor, item, self.image_size)
            prompt = encoded["input_ids"][0]
            prompt = prompt[encoded["attention_mask"][0].bool()]
            if self.training:
                answer = item["answer"]
                if not isinstance(answer, str):
                    answer = json.dumps(answer, ensure_ascii=False)
                response = tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id]
                response = torch.tensor(response, dtype=torch.long)
                sequence = torch.cat([prompt, response])
                label = torch.cat([torch.full_like(prompt, -100), response])
                labels.append(label)
            else:
                sequence = prompt
            if len(sequence) > self.max_length:
                raise ValueError(f"Sequence exceeds max_length={self.max_length}; increase it instead of truncating boxes")
            ids.append(sequence)
            pixels.append(encoded["pixel_values"])
            grids.append(encoded["image_grid_thw"])
            v, p = load_aef(item["aef"])
            vectors.append(v)
            positions.append(p)
        pad_seq = torch.nn.utils.rnn.pad_sequence
        batch = {
            "input_ids": pad_seq(ids, batch_first=True, padding_value=pad),
            "attention_mask": pad_seq([torch.ones_like(x) for x in ids], batch_first=True, padding_value=0),
            "pixel_values": torch.cat(pixels), "image_grid_thw": torch.cat(grids),
            "vec64_seq": pad_seq(vectors, batch_first=True),
            "vec64_pos": pad_seq(positions, batch_first=True),
            "vec64_mask": pad_seq([torch.ones(len(v), dtype=torch.bool) for v in vectors], batch_first=True),
        }
        if self.training:
            batch["labels"] = pad_seq(labels, batch_first=True, padding_value=-100)
        elif len(items) != 1:
            raise ValueError("The inference demo generates one image at a time")
        return batch


def parse_detections(text):
    """Never alter numeric values; retain raw text separately if parsing fails."""
    text = text.strip()
    if text.lower() in ("none", "null"):
        return []
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    objects = json.loads(text)
    if not isinstance(objects, list):
        raise ValueError("Expected a JSON list of detections or None")
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("label"), str):
            raise ValueError("Every detection requires a string label")
        box = obj.get("bbox_2d")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("bbox_2d must contain four coordinates")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in box):
            raise ValueError("bbox_2d coordinates must be finite numbers")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("bbox_2d must satisfy x1<x2 and y1<y2")
    return objects
