"""Extract weights from CoreML model.mil + weight.bin and map to PyTorch state_dict.

Usage:
    cd /path/to/Emagic-V-1.3
    pixi run python -m training.convert_coreml_weights

Outputs:
    training/coreml_weights.pt     — raw tensor dict keyed by MIL name
    training/coreml_weights.json   — manifest (name, shape, offset, nbytes)
    training/mapped_state_dict.pt  — PyTorch state_dict for BSRoformer
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Step 1: Parse model.mil for weight declarations
# ---------------------------------------------------------------------------

# BLOBFILE pattern: tensor<fp16, [shape]> name = const()[... BLOBFILE(... offset(N))]
RE_BLOBFILE = re.compile(
    r"tensor<fp16,\s*\[([^\]]+)\]>\s+(\w+_to_fp16)\s*=\s*const\(\)\[[^;]*?"
    r"BLOBFILE\([^;]*?offset\s*=\s*tensor<uint64,\s*\[\]>\((\d+)\)\)",
)

# Inline hex array pattern: tensor<fp16, [N]> name = const()[... val = tensor<fp16, [N]>([vals])]
RE_INLINE_ARRAY = re.compile(
    r"tensor<fp16,\s*\[(\d+)\]>\s+(\w+_to_fp16)\s*=\s*const\(\)\[[^;]*?"
    r"val\s*=\s*tensor<fp16,\s*\[\d+\]>\(\[([^\]]+)\]\)",
)


def parse_shape(shape_str: str) -> tuple[int, ...]:
    """Parse '256, 8' or '1536' into tuple of ints."""
    return tuple(int(x.strip()) for x in shape_str.split(","))


def parse_hex_fp16(val_str: str) -> float:
    """Parse MIL hex float like '0x1.2p-2' or '-0x1.8p-4' to Python float."""
    val_str = val_str.strip()
    try:
        return float.fromhex(val_str)
    except ValueError:
        return float(val_str)


def extract_weights_from_mil(mil_path: str) -> dict:
    """Parse model.mil and return weight metadata.

    Returns dict keyed by MIL name:
        {name: {"shape": tuple, "offset": int|None, "inline_values": list|None}}
    """
    with open(mil_path, "r") as f:
        text = f.read()

    weights = {}

    # BLOBFILE weights (large tensors stored in weight.bin)
    for match in RE_BLOBFILE.finditer(text):
        shape_str, name, offset_str = match.groups()
        shape = parse_shape(shape_str)
        weights[name] = {
            "shape": shape,
            "offset": int(offset_str),
            "inline_values": None,
        }

    # Inline hex arrays (small tensors like gate biases, norm gammas)
    for match in RE_INLINE_ARRAY.finditer(text):
        size_str, name, vals_str = match.groups()
        if name in weights:
            continue  # already found as BLOBFILE
        size = int(size_str)
        vals = [parse_hex_fp16(v) for v in vals_str.split(",")]
        assert len(vals) == size, f"Expected {size} values for {name}, got {len(vals)}"
        weights[name] = {
            "shape": (size,),
            "offset": None,
            "inline_values": vals,
        }

    return weights


def load_weights_from_bin(
    weights_meta: dict, bin_path: str
) -> dict[str, torch.Tensor]:
    """Read weight.bin and extract tensors."""
    with open(bin_path, "rb") as f:
        bin_data = f.read()

    tensors = {}
    for name, meta in weights_meta.items():
        shape = meta["shape"]
        numel = 1
        for s in shape:
            numel *= s

        if meta["offset"] is not None:
            # Read from BLOBFILE — skip 64-byte header (0xBEEF 0xDEAD magic + metadata)
            offset = meta["offset"] + 64
            nbytes = numel * 2  # fp16
            raw = bin_data[offset : offset + nbytes]
            arr = np.frombuffer(raw, dtype=np.float16).copy()
            # Replace fp16 NaN values (occasional training artifacts) with 0
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr[nan_mask] = 0.0
            tensor = torch.from_numpy(arr).float().reshape(shape)
        elif meta["inline_values"] is not None:
            tensor = torch.tensor(meta["inline_values"], dtype=torch.float32).reshape(shape)
        else:
            continue

        tensors[name] = tensor

    return tensors


# ---------------------------------------------------------------------------
# Step 2: Map CoreML names to PyTorch state_dict keys
# ---------------------------------------------------------------------------

def strip_suffix(name: str) -> str:
    """Remove '_to_fp16' suffix from MIL names."""
    if name.endswith("_to_fp16"):
        return name[:-8]
    return name


def map_to_pytorch(coreml_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map CoreML tensor names to BSRoformer PyTorch state_dict keys.

    CoreML naming conventions (from MIL forensics):
        Band split:
            band_split_to_features_{band}_{0=norm, 1=proj}_{gamma|weight|bias}
        Transformer blocks:
            layers_{depth}_{0=time, 1=band}_layers_0_{0=attn, 1=ffn}_*
            Attention: norm_gamma, to_qkv_weight, to_out_0_weight, to_gates_{weight|bias}
            FFN: net_0_gamma, net_1_{weight|bias}, net_4_{weight|bias}
        Mask estimators:
            mask_estimators_{stem}_to_freqs_{band}_0_{0|2}_{weight|bias}
        Shared biases:
            linear_62_bias_0  → QKV bias (shared across all 24 attention layers)
            linear_64_bias_0  → output proj bias (shared across all 24 attention layers)
    """
    state_dict = {}
    unmapped = []

    # Retrieve shared biases first
    qkv_bias = coreml_tensors.get("linear_62_bias_0_to_fp16")
    out_bias = coreml_tensors.get("linear_64_bias_0_to_fp16")

    for mil_name, tensor in coreml_tensors.items():
        name = strip_suffix(mil_name)
        mapped = False

        # --- Band split ---
        m = re.match(r"band_split_to_features_(\d+)_0_gamma", name)
        if m:
            idx = int(m.group(1))
            state_dict[f"band_split.norms.{idx}.gamma"] = tensor
            mapped = True

        m = re.match(r"band_split_to_features_(\d+)_1_weight", name)
        if m:
            idx = int(m.group(1))
            state_dict[f"band_split.projections.{idx}.weight"] = tensor
            mapped = True

        m = re.match(r"band_split_to_features_(\d+)_1_bias", name)
        if m:
            idx = int(m.group(1))
            state_dict[f"band_split.projections.{idx}.bias"] = tensor
            mapped = True

        # --- Transformer blocks ---
        # Attention norm gamma
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_0_norm_gamma", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_norm_attn.gamma"] = tensor
            mapped = True

        # QKV weight
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_0_to_qkv_weight", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_attn.to_qkv.weight"] = tensor
            # Also assign shared QKV bias to this layer
            if qkv_bias is not None:
                state_dict[f"blocks.{depth}.{prefix}_attn.to_qkv.bias"] = qkv_bias.clone()
            mapped = True

        # Output projection weight
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_0_to_out_0_weight", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_attn.to_out.weight"] = tensor
            # Also assign shared output bias
            if out_bias is not None:
                state_dict[f"blocks.{depth}.{prefix}_attn.to_out.bias"] = out_bias.clone()
            mapped = True

        # Gate weight
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_0_to_gates_weight", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_attn.to_gates.weight"] = tensor
            mapped = True

        # Gate bias
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_0_to_gates_bias", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_attn.to_gates.bias"] = tensor
            mapped = True

        # FFN norm gamma
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_1_net_0_gamma", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_norm_ff.gamma"] = tensor
            mapped = True

        # FFN first linear weight
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_1_net_1_weight", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_ff.net.0.weight"] = tensor
            mapped = True

        # FFN first linear bias
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_1_net_1_bias", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_ff.net.0.bias"] = tensor
            mapped = True

        # FFN second linear weight (net_4 in MIL = index 2 in Sequential after GELU)
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_1_net_4_weight", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_ff.net.2.weight"] = tensor
            mapped = True

        # FFN second linear bias
        m = re.match(r"layers_(\d+)_(\d+)_layers_0_1_net_4_bias", name)
        if m:
            depth, axis = int(m.group(1)), int(m.group(2))
            prefix = "time" if axis == 0 else "band"
            state_dict[f"blocks.{depth}.{prefix}_ff.net.2.bias"] = tensor
            mapped = True

        # --- Mask estimators ---
        # First linear (index 0 in Sequential)
        m = re.match(r"mask_estimators_(\d+)_to_freqs_(\d+)_0_0_weight", name)
        if m:
            stem, band = int(m.group(1)), int(m.group(2))
            state_dict[f"mask_estimator.estimators.{stem}.{band}.0.weight"] = tensor
            mapped = True

        m = re.match(r"mask_estimators_(\d+)_to_freqs_(\d+)_0_0_bias", name)
        if m:
            stem, band = int(m.group(1)), int(m.group(2))
            state_dict[f"mask_estimator.estimators.{stem}.{band}.0.bias"] = tensor
            mapped = True

        # Second linear (index 2 in Sequential, after Tanh)
        m = re.match(r"mask_estimators_(\d+)_to_freqs_(\d+)_0_2_weight", name)
        if m:
            stem, band = int(m.group(1)), int(m.group(2))
            state_dict[f"mask_estimator.estimators.{stem}.{band}.2.weight"] = tensor
            mapped = True

        m = re.match(r"mask_estimators_(\d+)_to_freqs_(\d+)_0_2_bias", name)
        if m:
            stem, band = int(m.group(1)), int(m.group(2))
            state_dict[f"mask_estimator.estimators.{stem}.{band}.2.bias"] = tensor
            mapped = True

        # --- Final norm ---
        if name == "final_norm_gamma":
            state_dict["final_norm.gamma"] = tensor
            mapped = True

        # --- Skip known non-weight tensors ---
        if not mapped and any(
            name.startswith(p) for p in (
                "linear_62_bias_0",  # shared QKV bias (handled above)
                "linear_64_bias_0",  # shared out bias (handled above)
                "expand_dims_0",     # STFT cosine basis (not needed, we use torch.stft)
                "expand_dims_1",     # STFT sine basis
                "var_1429",          # time RoPE cos (computed buffer)
                "var_1473",          # time RoPE sin (computed buffer)
                "var_1760",          # band RoPE cos (computed buffer)
                "var_1804",          # band RoPE sin (computed buffer)
                "op_1429",           # alt naming for RoPE
                "op_1473",
                "op_1760",
                "op_1804",
            )
        ):
            mapped = True

        if not mapped:
            unmapped.append(mil_name)

    return state_dict, unmapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    mil_path = project_root / "ke77nsfms3_8024.mlmodelc" / "model.mil"
    bin_path = project_root / "ke77nsfms3_8024.mlmodelc" / "weights" / "weight.bin"
    out_dir = Path(__file__).resolve().parent

    if not mil_path.exists():
        print(f"ERROR: {mil_path} not found")
        sys.exit(1)
    if not bin_path.exists():
        print(f"ERROR: {bin_path} not found")
        sys.exit(1)

    # Step 1: Parse MIL
    print("Parsing model.mil...")
    weights_meta = extract_weights_from_mil(str(mil_path))
    print(f"  Found {len(weights_meta)} weight tensors")

    # Count by category
    cats = {"band_split": 0, "layers": 0, "mask_estimators": 0, "other": 0}
    for name in weights_meta:
        clean = strip_suffix(name)
        if clean.startswith("band_split"):
            cats["band_split"] += 1
        elif clean.startswith("layers_"):
            cats["layers"] += 1
        elif clean.startswith("mask_estimators"):
            cats["mask_estimators"] += 1
        else:
            cats["other"] += 1
    for cat, count in cats.items():
        print(f"    {cat}: {count}")

    # Step 2: Extract weights from weight.bin
    print("Extracting weights from weight.bin...")
    coreml_tensors = load_weights_from_bin(weights_meta, str(bin_path))
    print(f"  Loaded {len(coreml_tensors)} tensors")

    # Save raw extraction
    raw_path = out_dir / "coreml_weights.pt"
    torch.save(coreml_tensors, str(raw_path))
    print(f"  Saved: {raw_path}")

    # Save manifest
    manifest = {}
    for name, meta in weights_meta.items():
        manifest[name] = {
            "shape": list(meta["shape"]),
            "offset": meta["offset"],
            "nbytes": (1 if meta["inline_values"] else 2) * int(np.prod(meta["shape"])),
        }
    manifest_path = out_dir / "coreml_weights.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    # Step 3: Map to PyTorch state_dict
    print("Mapping to PyTorch state_dict...")
    state_dict, unmapped = map_to_pytorch(coreml_tensors)
    print(f"  Mapped {len(state_dict)} PyTorch parameters")

    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped tensors:")
        for u in unmapped[:10]:
            print(f"    {u}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")

    # Step 4: Validate against model
    print("Validating against BSRoformer model...")
    from .bsroformer import BSRoformer

    model = BSRoformer()
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(state_dict.keys())

    # Exclude RoPE buffers (not weights) and STFT window
    buffer_keys = {k for k in model_keys if any(
        b in k for b in ("cos_cached", "sin_cached", "inv_freq", "window")
    )}
    model_param_keys = model_keys - buffer_keys

    missing = model_param_keys - mapped_keys
    extra = mapped_keys - model_param_keys

    if missing:
        print(f"  Missing from CoreML ({len(missing)} params):")
        for k in sorted(missing)[:10]:
            print(f"    {k}")
    else:
        print("  All model parameters have CoreML weights!")

    if extra:
        print(f"  Extra keys not in model ({len(extra)}):")
        for k in sorted(extra)[:10]:
            print(f"    {k}")

    # Check shape mismatches
    mismatches = []
    for key in mapped_keys & model_param_keys:
        model_shape = model.state_dict()[key].shape
        coreml_shape = state_dict[key].shape
        if model_shape != coreml_shape:
            mismatches.append((key, model_shape, coreml_shape))

    if mismatches:
        print(f"  Shape mismatches ({len(mismatches)}):")
        for key, ms, cs in mismatches[:10]:
            print(f"    {key}: model={ms} coreml={cs}")
    else:
        print("  All shapes match!")

    # Save mapped state_dict
    sd_path = out_dir / "mapped_state_dict.pt"
    torch.save(state_dict, str(sd_path))
    print(f"  Saved: {sd_path}")

    # Summary
    total_params = sum(t.numel() for t in state_dict.values())
    total_bytes = total_params * 4  # fp32
    print(f"\nDone! {len(state_dict)} tensors, {total_params:,} params, {total_bytes/1e6:.1f} MB (fp32)")


if __name__ == "__main__":
    main()
