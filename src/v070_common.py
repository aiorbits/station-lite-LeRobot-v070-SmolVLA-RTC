#!/usr/bin/env python3
"""Shared helpers for G0 manifest verification, G1 CUDA/model gating, and G2 comparisons."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SIZE = (640, 480)
CAMERA_MAPPING = {
    "observation.images.cam_high": "observation.images.camera1",
    "observation.images.cam_right_wrist": "observation.images.camera2",
}
PINNED_VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
PINNED_BASE_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file cleanly in fixed-size chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """Compute hex SHA-256 of a C-contiguous view of a NumPy array."""
    c_arr = np.ascontiguousarray(arr)
    return hashlib.sha256(c_arr.tobytes()).hexdigest()


def configure_offline_env(offline: bool) -> dict[str, str]:
    """Set offline HF/Transformers flags prior to importing model libraries."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return {
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }


def check_hf_vlm_cache(pinned_vlm_rev: str = PINNED_VLM_REVISION) -> dict[str, Any]:
    """Inspect active HuggingFace cache for pinned SmolVLM2 snapshot without network access."""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if not hub_cache and os.environ.get("HF_HOME"):
        hub_cache = str(Path(os.environ["HF_HOME"]) / "hub")
    if not hub_cache:
        return {"configured": False, "cache_dir": None, "vlm_snapshot_present": False, "refs_main": None}
    model_dir = Path(hub_cache) / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    snap_dir = model_dir / "snapshots" / pinned_vlm_rev
    refs_main = model_dir / "refs/main"
    ref_text = refs_main.read_text(encoding="utf-8").strip() if refs_main.is_file() else None
    return {
        "configured": True,
        "cache_dir": hub_cache,
        "vlm_model_dir_exists": model_dir.is_dir(),
        "vlm_snapshot_present": snap_dir.is_dir(),
        "pinned_vlm_revision": pinned_vlm_rev,
        "refs_main": ref_text,
        "refs_main_matches_pinned": (ref_text == pinned_vlm_rev) if ref_text is not None else None,
    }


def verify_env_identity(
    expected_prefix: Path | None = None,
    expected_lerobot_version: str | None = None,
    require_site_packages: bool = True,
) -> dict[str, Any]:
    """Verify isolated conda environment, imported lerobot origin, packages, and CUDA GPU."""
    import torch

    actual_prefix = Path(sys.prefix).resolve()
    lerobot_ver = metadata.version("lerobot")
    spec = importlib.util.find_spec("lerobot")
    lerobot_origin = Path(spec.origin).resolve() if spec and spec.origin else None

    if expected_prefix is not None:
        exp_prefix = expected_prefix.resolve()
        if actual_prefix != exp_prefix:
            raise RuntimeError(f"sys.prefix mismatch: actual={actual_prefix}, expected={exp_prefix}")
        if require_site_packages:
            py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
            expected_pkg = exp_prefix / "lib" / py_tag / "site-packages/lerobot"
            if lerobot_origin is None or not lerobot_origin.is_relative_to(expected_pkg):
                raise RuntimeError(f"lerobot origin {lerobot_origin} is outside {expected_pkg}")

    if expected_lerobot_version is not None and lerobot_ver != expected_lerobot_version:
        raise RuntimeError(f"lerobot version mismatch: actual={lerobot_ver}, expected={expected_lerobot_version}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; refusing CPU fallback per project rule")

    probe = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    pkg_names = [
        "lerobot", "torch", "torchvision", "transformers", "accelerate",
        "datasets", "huggingface-hub", "numpy", "Pillow", "opencv-python-headless",
        "num2words", "scipy",
    ]
    packages: dict[str, str | None] = {}
    for name in pkg_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sys_prefix": str(actual_prefix),
        "lerobot_version": lerobot_ver,
        "lerobot_origin": str(lerobot_origin) if lerobot_origin else None,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_probe_device": str(probe.device),
        "packages": packages,
    }


def apply_tf32_mode(mode: str) -> dict[str, Any]:
    """Configure and report PyTorch CUDA/cuDNN TF32 & float32 matmul precision flags."""
    import torch

    if mode == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    elif mode == "on":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    elif mode != "preserve":
        raise ValueError(f"Unsupported tf32_mode: {mode}")

    matmul_prec = torch.get_float32_matmul_precision() if hasattr(torch, "get_float32_matmul_precision") else None
    return {
        "tf32_mode_requested": mode,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": matmul_prec,
    }


def verify_checkpoint_manifest(checkpoint: Path, baseline_manifest: Path) -> dict[str, Any]:
    """Verify every file listed under baseline BASELINE_MANIFEST.json checkpoint_files."""
    manifest = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    ckpt = checkpoint.resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")
    checked = []
    mismatches = []
    for entry in manifest["checkpoint_files"]:
        target = ckpt / entry["path"]
        if not target.is_file():
            mismatches.append({"path": entry["path"], "reason": "missing"})
            continue
        actual_sha = sha256_file(target)
        actual_bytes = target.stat().st_size
        if actual_sha != entry["sha256"] or actual_bytes != entry["bytes"]:
            mismatches.append({"path": entry["path"], "reason": "checksum_or_size_mismatch", "actual_sha": actual_sha})
        checked.append(entry["path"])
    if mismatches:
        raise RuntimeError(f"Checkpoint manifest verification failed: {mismatches}")
    return {"checkpoint_path": str(ckpt), "verified_files": len(checked), "files": checked}


def load_replay_manifest(
    manifest_path: Path, verify_hashes: bool = True, smolvla_root: Path | None = None
) -> dict[str, Any]:
    """Load the G0 replay manifest and verify file SHA-256 checksums & payload contracts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(smolvla_root or manifest["smolvla_root"]).resolve()
    for win in manifest["windows"]:
        state_path = root / win["state_rel_path"]
        raw_path = root / win["raw_actions_rel_path"]
        if verify_hashes:
            if sha256_file(state_path) != win["state_sha256"]:
                raise RuntimeError(f"State SHA-256 mismatch for {win['window_id']}")
            if sha256_file(raw_path) != win["raw_actions_sha256"]:
                raise RuntimeError(f"Raw actions SHA-256 mismatch for {win['window_id']}")
            for cam_key, img_meta in win["images"].items():
                img_path = root / img_meta["rel_path"]
                if sha256_file(img_path) != img_meta["file_sha256"]:
                    raise RuntimeError(f"Image SHA-256 mismatch for {win['window_id']} {cam_key}")
    return manifest


def clone_observation(val: Any) -> Any:
    """Recursively clone tensor leaves in an observation dict so preprocessing never mutates originals."""
    import torch

    if isinstance(val, torch.Tensor):
        return val.clone()
    if isinstance(val, dict):
        return {k: clone_observation(v) for k, v in val.items()}
    if isinstance(val, list):
        return [clone_observation(v) for v in val]
    return val


def run_timed_chunk(
    policy: Any,
    pre: Any,
    post: Any,
    obs: dict[str, Any],
    noise: Any | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Run synchronized CUDA preprocess -> policy chunk prediction -> postprocess and return (action, ms)."""
    import time
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        batch = pre(clone_observation(obs))
        if noise is not None:
            pred_norm = policy.predict_action_chunk(batch, noise=noise.clone())
        else:
            if seed is not None:
                torch.cuda.manual_seed(int(seed))
            pred_norm = policy.predict_action_chunk(batch)
        act_t = post(pred_norm)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    action = act_t[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
    return action, elapsed_ms


def build_observation_for_window(smolvla_root: Path, window: dict[str, Any]) -> dict[str, Any]:
    """Build exact host-server-equivalent observation dict for a manifest window."""
    import torch

    state_path = smolvla_root / window["state_rel_path"]
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    state = np.asarray(state_data["state"], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f"{window['window_id']}: expected 7 finite floats for state, got {state.shape}")

    obs: dict[str, Any] = {
        "observation.state": torch.from_numpy(state.copy()),
        "task": str(window["task"]),
    }
    for orig_key, mapped_key in CAMERA_MAPPING.items():
        img_meta = window["images"][orig_key]
        img_path = smolvla_root / img_meta["rel_path"]
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if image.width * IMAGE_SIZE[1] != image.height * IMAGE_SIZE[0]:
                raise ValueError(f"{orig_key} aspect mismatch: {image.size}")
            if image.size != IMAGE_SIZE:
                image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.uint8).copy()
        obs[mapped_key] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return obs


def inspect_policy_contract(policy: Any) -> dict[str, Any]:
    """Inspect SmolVLA config camera slots, noise parameter support, and dimensions."""
    cfg = policy.config
    input_feats = {k: list(v.shape) for k, v in cfg.input_features.items()}
    physical_mapped = sorted(CAMERA_MAPPING.values())
    extra_camera_slots = [k for k in input_feats if "images" in k and k not in physical_mapped]
    sig = inspect.signature(policy.predict_action_chunk)
    return {
        "chunk_size": int(cfg.chunk_size),
        "max_action_dim": int(cfg.max_action_dim),
        "max_state_dim": int(cfg.max_state_dim),
        "action_shape": list(cfg.action_feature.shape),
        "input_features": input_feats,
        "physical_camera_keys": physical_mapped,
        "padded_config_camera_slots_not_supplied": extra_camera_slots,
        "num_steps": int(getattr(cfg, "num_steps", -1)),
        "use_amp": bool(getattr(cfg, "use_amp", False)),
        "resize_imgs_with_padding": list(getattr(cfg, "resize_imgs_with_padding", [])),
        "predict_action_chunk_params": list(sig.parameters.keys()),
        "supports_explicit_noise": "noise" in sig.parameters,
        "expected_noise_shape": [1, int(cfg.chunk_size), int(cfg.max_action_dim)],
    }


def load_policy_and_processors(checkpoint: Path):
    """Load SmolVLAPolicy on CUDA and paired pre/post processors."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ckpt = checkpoint.resolve()
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, pre, post


def euler_xyz_to_matrix_batch(rpy_rad: np.ndarray) -> np.ndarray:
    """Convert (..., 3) extrinsic Euler 'xyz' [roll, pitch, yaw] radians to (..., 3, 3) matrices."""
    rpy = np.asarray(rpy_rad, dtype=np.float64)
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    mat = np.empty(rpy.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = cy * cp
    mat[..., 0, 1] = cy * sp * sr - sy * cr
    mat[..., 0, 2] = cy * sp * cr + sy * sr
    mat[..., 1, 0] = sy * cp
    mat[..., 1, 1] = sy * sp * sr + cy * cr
    mat[..., 1, 2] = sy * sp * cr - cy * sr
    mat[..., 2, 0] = -sp
    mat[..., 2, 1] = cp * sr
    mat[..., 2, 2] = cp * cr
    return mat


def geodesic_rotation_error_deg(rpy_a: np.ndarray, rpy_b: np.ndarray) -> np.ndarray:
    """Compute exact SO(3) geodesic angular separation in degrees between two Euler arrays."""
    ra = euler_xyz_to_matrix_batch(rpy_a)
    rb = euler_xyz_to_matrix_batch(rpy_b)
    rel = np.einsum("...ji,...jk->...ik", ra, rb)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def wrapped_euler_delta_deg(rpy_a: np.ndarray, rpy_b: np.ndarray) -> np.ndarray:
    """Compute per-axis wrapped angle difference |wrap(a - b)| in degrees."""
    diff = np.asarray(rpy_a, dtype=np.float64) - np.asarray(rpy_b, dtype=np.float64)
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return np.degrees(np.abs(wrapped))


def summarize_scalar_errors(values: np.ndarray) -> dict[str, float]:
    """Summarize strictly non-negative error array without unit mixing."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }
