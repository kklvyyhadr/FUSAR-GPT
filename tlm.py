"""TLM copied from the original training calculation; inference follows training."""
import json
from pathlib import Path

import torch
from torch import nn


class TLMModel(nn.Module):
    # The underlying model returns a mean loss; Trainer must handle accumulation.
    accepts_loss_kwargs = False
    FORMAT = "tlm_original_train_patch_resize_v1"

    def __init__(self, base_model, sigma=2.0, hidden_mult=4, image_size=504):
        super().__init__()
        self.base_model = base_model
        self.sigma = float(sigma)
        self.hidden_mult = int(hidden_mult)
        self.image_size = int(image_size)
        if self.sigma <= 0 or self.image_size % 28:
            raise ValueError("sigma must be positive; image_size must be a multiple of 28")
        inner = base_model.get_base_model() if hasattr(base_model, "get_base_model") else base_model
        visual = getattr(inner, "visual", None)
        if visual is None:
            visual = getattr(getattr(inner, "model", None), "visual", None)
        if visual is None:
            raise ValueError("Expected Qwen2.5-VL visual module; use the pinned Transformers version")
        self.merge_size = int(visual.spatial_merge_size)
        text_config = getattr(inner.config, "text_config", inner.config)
        channels = int(text_config.hidden_size)
        hidden = max(128, self.hidden_mult * channels)
        self.film_mlp = nn.Sequential(nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels))
        self._context = None
        self._hook_handle = visual.register_forward_hook(self._modulate)

    @property
    def config(self):
        return self.base_model.config

    def _modulate(self, module, inputs, output):
        if self._context is None:
            raise RuntimeError("AEF context is required for every image forward pass")
        if not torch.is_tensor(output) or output.ndim != 2:
            raise RuntimeError("Expected packed visual tokens [sum(tokens_per_image), channels]")
        vectors, positions, mask, grids = self._context
        if len(grids) != 1 or len(vectors) != 1:
            raise ValueError("Use batch_size=1: preserve the original single-image training calculation")
        chunks, offset = [], 0
        for i, (t, hp, wp) in enumerate(grids.detach().cpu().tolist()):
            if t != 1 or hp % self.merge_size or wp % self.merge_size:
                raise ValueError("Only still images with valid Qwen patch grids are supported")
            h, w = hp // self.merge_size, wp // self.merge_size
            count = h * w
            tokens = output[offset:offset + count]
            offset += count
            valid = mask[i].bool()
            if not valid.any():
                raise ValueError("Every image needs at least one valid AEF vector")
            # Accelerate may dispatch the visual tower to a different GPU.
            # Move only the TLM MLP in eval mode, never the dispatched backbone.
            if self.film_mlp[0].weight.device != output.device:
                if self.training:
                    raise RuntimeError("Trainer must place TLM and visual output on the same local device")
                self.film_mlp.to(output.device)
            v = vectors[i].to(device=output.device, dtype=self.film_mlp[0].weight.dtype)
            p = positions[i].to(device=output.device, dtype=torch.float32)
            # Match original BF16 training autocast during inference as well.
            mixed = output.dtype in (torch.bfloat16, torch.float16)
            with torch.autocast(device_type=output.device.type,
                                dtype=output.dtype if mixed else torch.bfloat16, enabled=mixed):
                gamma, beta = self.film_mlp(v).chunk(2, dim=-1)
                gamma, beta = gamma.to(output.dtype), beta.to(output.dtype)
                # Original training: Gaussian on the PRE-MERGE patch grid.
                yy, xx = torch.meshgrid(
                    torch.arange(hp, device=output.device, dtype=torch.float32),
                    torch.arange(wp, device=output.device, dtype=torch.float32), indexing="ij"
                )
                d2 = (p[:, 0, None] * (hp - 1) - yy.flatten()).square()
                d2 = d2 + (p[:, 1, None] * (wp - 1) - xx.flatten()).square()
                weights = torch.exp(-d2 / (2 * self.sigma ** 2))
                weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-6)
                # Preserve the training mask and second normalization, including epsilon.
                weights = weights * mask[i].to(output.device, torch.float32).view(-1, 1)
                weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-6)
                g, b = weights.T @ gamma, weights.T @ beta
                channels = tokens.shape[-1]
                height = int(round(count ** 0.5))
                width = max(1, (count + height - 1) // height)
                def resize(field):
                    field = field.view(1, hp, wp, channels).permute(0, 3, 1, 2).contiguous()
                    field = torch.nn.functional.interpolate(field, size=(height, width),
                                                           mode="bilinear", align_corners=False)
                    field = field.permute(0, 2, 3, 1).reshape(-1, channels)
                    if len(field) != count:
                        idx = torch.linspace(0, len(field) - 1, count, device=field.device).round().long()
                        field = field.index_select(0, idx)
                    return field
                g, b = resize(g), resize(b)
                chunks.append(tokens * (1 + g) + b)
        if offset != len(output):
            raise ValueError("Packed visual token count does not match image_grid_thw")
        return torch.cat(chunks, dim=0)

    def _call(self, method, kwargs):
        vectors = kwargs.pop("vec64_seq")
        positions = kwargs.pop("vec64_pos")
        mask = kwargs.pop("vec64_mask")
        self._context = (vectors, positions, mask, kwargs["image_grid_thw"])
        kwargs.pop("num_items_in_batch", None)
        try:
            return method(**kwargs)
        finally:
            self._context = None

    def forward(self, **kwargs):
        return self._call(self.base_model, kwargs)

    def generate(self, **kwargs):
        if kwargs.get("num_beams", 1) != 1 or kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("This demo supports one generated answer per image")
        return self._call(self.base_model.generate, kwargs)

    def set_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        for p in self.parameters():
            p.requires_grad_(False)
        for name, p in self.base_model.named_parameters():
            if "lora_" in name:
                # Continue all adapters present in the checkpoint, including
                # vision LoRA in the released legacy Stage 1 weights.
                p.requires_grad_(True)
        if stage == 1:
            for p in self.film_mlp.parameters():
                p.requires_grad_(True)

    def save_tlm(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"film_mlp": self.film_mlp.state_dict()}, directory / "vec64_film.pt")
        config = dict(format=self.FORMAT, sigma=self.sigma, hidden_mult=self.hidden_mult,
                      image_size=self.image_size, vec_dim=64)
        (directory / "tlm_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def from_adapter(cls, base_model, directory, legacy_sigma=None):
        directory = Path(directory)
        path = directory / "tlm_config.json"
        if not path.exists():
            if legacy_sigma is None:
                raise ValueError("Legacy checkpoint: supply --legacy-sigma with its TRAINING sigma (original script: 2)")
            config = dict(format=cls.FORMAT, sigma=legacy_sigma, hidden_mult=4, image_size=504, vec_dim=64)
        else:
            config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("format") != cls.FORMAT or config.get("vec_dim") != 64:
            raise ValueError("Unsupported TLM checkpoint format")
        model = cls(base_model, config["sigma"], config["hidden_mult"], config["image_size"])
        state = torch.load(directory / "vec64_film.pt", map_location="cpu", weights_only=True)
        model.film_mlp.load_state_dict(state["film_mlp"], strict=True)
        # Legacy `score` belongs to the unused global fallback, not the local TLM path.
        return model
