"""Shared LoRA arguments for new adapters and checkpoint continuation."""
import json
from pathlib import Path


def add_lora_arguments(parser):
    parser.add_argument("--lora-rank", type=int,
                        help="New adapter rank (default: 8); must match when loading an existing adapter")
    parser.add_argument("--lora-alpha", type=int,
                        help="New adapter alpha (default: 32); must match when loading an existing adapter")


def resolve_lora_parameters(init_adapter=None, rank=None, alpha=None):
    for name, value in (("rank", rank), ("alpha", alpha)):
        if value is not None and value <= 0:
            raise ValueError(f"--lora-{name} must be positive")
    if not init_adapter:
        return (8 if rank is None else rank, 32 if alpha is None else alpha)

    path = Path(init_adapter) / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for flag, key, requested in (("rank", "r", rank), ("alpha", "lora_alpha", alpha)):
        saved = config[key]
        if requested is not None and requested != saved:
            raise ValueError(
                f"--lora-{flag}={requested} conflicts with {path} ({key}={saved}). "
                f"Omit --lora-{flag} or use --lora-{flag} {saved} to continue this adapter. "
                "Existing adapter weights and scaling are preserved."
            )
    return config["r"], config["lora_alpha"]
