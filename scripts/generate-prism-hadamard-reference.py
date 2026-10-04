#!/usr/bin/env python3
"""Generate small Bonsai reference fixtures without downloading model weights.

Requires mlx==0.32.3 and PrismML-Eng/mlx-lm at REFERENCE_REVISION.
Pass --mlx-lm /path/to/checkout and --output /path/to/reference.json.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

REFERENCE_REVISION = "38f27dc24b535928246b64b211b66d38b7a3e17f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-lm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.mlx_lm), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE_REVISION:
        parser.error(f"Expected mlx-lm revision {REFERENCE_REVISION}, got {revision}")
    sys.path.insert(0, str(args.mlx_lm.resolve()))

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.models import prism_hadamard_qwen35 as prism
    from mlx_lm.models import qwen3_5

    if mx.__version__ != "0.32.3":
        parser.error(f"Expected mlx 0.32.3, got {mx.__version__}")
    mx.set_default_device(mx.gpu)
    text = {
        "model_type": "qwen3_5_text", "hidden_size": 1024,
        "intermediate_size": 2048, "num_hidden_layers": 2,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 128,
        "vocab_size": 16, "tie_word_embeddings": False,
        "linear_num_value_heads": 8, "linear_num_key_heads": 2,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default", "rope_theta": 10000000,
            "partial_rotary_factor": 0.25, "mrope_section": [6, 5, 5],
        },
    }
    base = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({
        "model_type": "qwen3_5", "text_config": text,
    }))
    targets = (
        "embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj",
    )
    modules = [
        {"path": path, "block": 0 if path == "lm_head" else 1024,
         "embedding": path == "model.embed_tokens", "dtype": "float16"}
        for path, _ in base.language_model.named_modules()
        if path.split(".")[-1] in targets
    ]
    config = {
        "model_type": "prism_hadamard_qwen35", "schema_version": 2,
        "tensor_namespace": "mlx-vlm-qwen3_5", "gdn_activation_layout": "grouped",
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "components": {"text": True, "vision": True}, "modules": modules,
        "text_config": text, "tie_word_embeddings": False,
        "vision_config": {
            "model_type": "qwen3_5", "depth": 1, "hidden_size": 32,
            "intermediate_size": 64, "out_hidden_size": 1024, "num_heads": 4,
            "patch_size": 2, "spatial_merge_size": 1, "temporal_patch_size": 1,
            "num_position_embeddings": 16,
        },
    }
    model = prism.Model(prism.ModelArgs.from_dict(config))
    weights, recipes = {}, {}
    for name, parameter in sorted(tree_flatten(model.parameters())):
        seed = sum(name.encode("utf-8"))
        if parameter.dtype == mx.uint32:
            pattern = [((i + seed) * 1103515245 + 12345) & 0xFFFFFFFF for i in range(67)]
            dtype = "uint32"
        elif name.endswith(".signs"):
            pattern = [1 if (i + seed) % 3 == 0 else -1 for i in range(31)]
            dtype = "float32"
        elif name.endswith(".scales"):
            pattern, dtype = [1 / 256], "float16"
        elif name.endswith(".biases"):
            pattern, dtype = [-1.5 / 256], "float16"
        elif name.endswith(".A_log"):
            pattern, dtype = [i / 8 for i in range(7)], "float32"
        elif name.endswith(".dt_bias"):
            pattern, dtype = [1], "float16"
        elif parameter.ndim == 1:
            pattern = [1 + ((i + seed) % 13 - 6) / 128 for i in range(13)]
            dtype = "float16"
        else:
            pattern = [((i + seed) * 17 % 31 - 15) / 1024 for i in range(31)]
            dtype = "float16"
        values = np.resize(np.array(pattern, dtype=dtype), parameter.shape)
        weights[name] = mx.array(values)
        recipes[name] = {"shape": list(parameter.shape), "dtype": dtype, "pattern": pattern}

    tokens = [[1, 2, 3, 4, 5], [6], [7], [8, 9, 10], [11]]
    cases = []
    for tied in [False, True]:
        case_config = dict(config, text_config=dict(text, tie_word_embeddings=tied))
        case_config["tie_word_embeddings"] = tied
        if tied:
            case_config["modules"] = [m for m in modules if m["path"] != "lm_head"]
        model = prism.Model(prism.ModelArgs.from_dict(case_config))
        case_weights = {k: v for k, v in weights.items()
                        if not tied or not k.startswith("language_model.lm_head.")}
        model.update(tree_unflatten(model.sanitize(case_weights)))
        model.eval()
        mx.eval(model.parameters())
        cache = model.make_cache()
        steps = []
        for chunk in tokens:
            logits = model(mx.array([chunk]), cache=cache)
            mx.eval(logits)
            steps.append({"tokens": chunk, "logits": logits.astype(mx.float32).reshape(-1).tolist()})
        full = model(mx.array([[t for chunk in tokens for t in chunk]]))
        mx.eval(full)
        cases.append({"tied": tied, "steps": steps,
                      "full_logits": full.astype(mx.float32).reshape(-1).tolist()})

    fixture = {
        "reference_revision": REFERENCE_REVISION, "mlx_version": mx.__version__,
        "config": config, "weights": recipes, "cases": cases,
    }
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
