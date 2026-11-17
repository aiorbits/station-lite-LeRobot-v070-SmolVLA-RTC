import json
import os
import subprocess
from pathlib import Path
import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "smolvla-storage")
CKPT_RELPATH = os.environ.get(
    "MODAL_CHECKPOINT_RELPATH",
    "checkpoint_010000/010000/pretrained_model",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "lerobot>=0.4.0",
        "transformers",
        "accelerate",
        "num2words",
        "fastapi",
        "uvicorn",
        "pillow",
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/repo")
)

app = modal.App("smolvla-live-server")
storage_volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": storage_volume},
)
@modal.web_server(port=8008, startup_timeout=300)
def serve():
    ckpt_dir = Path("/data") / CKPT_RELPATH

    # Strip pretrained_revision if present
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "pretrained_revision" in cfg:
            del cfg["pretrained_revision"]
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Resolve relative "tokenizer" path in policy_preprocessor.json
    preproc_path = ckpt_dir / "policy_preprocessor.json"
    if preproc_path.exists():
        preproc = json.loads(preproc_path.read_text(encoding="utf-8"))
        local_tok = ckpt_dir / "tokenizer"
        tok_target = (
            str(local_tok)
            if local_tok.exists()
            else "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        for step in preproc.get("steps", []):
            cfg_step = step.get("config", {})
            if cfg_step.get("tokenizer_name") == "tokenizer":
                cfg_step["tokenizer_name"] = tok_target
        preproc_path.write_text(json.dumps(preproc, indent=2), encoding="utf-8")

    subprocess.Popen(
        [
            "python3",
            "/repo/src/station_lite_smolvla_right7_server.py",
            "--checkpoint",
            str(ckpt_dir),
            "--host",
            "0.0.0.0",
            "--port",
            "8008",
            "--seed",
            "42",
            "--seed-mode",
            "fixed",
            "--no-verify-env",
        ],
        cwd="/repo",
    )
