# Station Lite SmolVLA Cloud Deployment Guide (Modal GPU + Local Ethernet Bridge)

This guide documents the end-to-end architecture and step-by-step setup for running the `station-lite-LeRobot-v070-SmolVLA-RTC` policy (`smolvla` right-7 action space) on a **Station Lite dual-arm robot** using a serverless **Modal T4 GPU** bridged through a **Windows workstation** over a local Ethernet connection.

---

## 1. System Architecture

```text
+-----------------------+       Ethernet LAN       +-----------------------+       HTTPS        +-----------------------+
|   Robot Control PC    | <----------------------> |  Windows Bridge Host  | <----------------> |   Modal Cloud GPU     |
|    (192.168.0.101)    |     HTTP :8008           |    (192.168.0.100)    |   .modal.run URL   |     (Tesla T4)        |
+-----------------------+                          +-----------------------+                    +-----------------------+
| - DualTouchRobot SDK  |                          | - modal serve         |                    | - LeRobot v0.4.4+     |
| - teleimager (:60000) |                          | - bridge.py (:8008)   |                    | - SmolVLA Policy      |
| - Guarded Client      |                          |                       |                    | - Modal Volume (/data)|
+-----------------------+                          +-----------------------+                    +-----------------------+
```

### Key Components
- **Modal Cloud GPU (`modal_server.py`)**: Hosts `station_lite_smolvla_right7_server.py` on a Tesla T4 GPU with the checkpoint mounted on a persistent Modal Volume (`act-storage`).
- **Local HTTP Proxy (`bridge.py`)**: Runs on the Windows laptop (`0.0.0.0:8008`), forwarding `GET /health` and `POST /predict` requests from the air-gapped Robot PC (`192.168.0.101`) to the Modal HTTPS endpoint.
- **Robot Control PC (`192.168.0.101`)**: Runs the camera server (`teleimager.image_server` on ZMQ port `60000`) and the guarded execution client (`station_lite_smolvla_right7_guarded.py`).

---

## 2. Cloud Inference Server Setup (Windows Host)

### 2.1 `modal_server.py`
The Modal container mounts the checkpoint volume at `/data`, patches relative tokenizer paths in `policy_preprocessor.json` if needed, and launches the inference server on port `8008`.

> **Note:** Modal resolves `/data` symlinks internally (e.g., `/__modal/volumes/<volume-id>/...`). Check the `checkpoint` field returned by `GET /health` and pass that exact string to `--expected-checkpoint` on the client.

```python
import modal
import subprocess
import json
from pathlib import Path

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
    .add_local_dir("station-lite-LeRobot-v070-SmolVLA-RTC", remote_path="/repo")
)

app = modal.App("smolvla-live-server")
storage_volume = modal.Volume.from_name("act-storage")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": storage_volume},
)
@modal.web_server(port=8008, startup_timeout=300)
def serve():
    ckpt_dir = Path("/data/checkpoint_010000/010000/pretrained_model")

    # Strip pretrained_revision if present
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "pretrained_revision" in cfg:
            del cfg["pretrained_revision"]
            cfg_path.write_text(json.dumps(cfg, indent=2))

    # Resolve relative "tokenizer" path in policy_preprocessor.json
    preproc_path = ckpt_dir / "policy_preprocessor.json"
    if preproc_path.exists():
        preproc = json.loads(preproc_path.read_text())
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
        preproc_path.write_text(json.dumps(preproc, indent=2))

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
```

### 2.2 `bridge.py` (Local Ethernet Proxy)
Run this alongside `modal serve` to bridge LAN HTTP traffic (`0.0.0.0:8008`) to your active `.modal.run` URL.

```python
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
import urllib.error

MODAL_BASE = "https://<your-workspace>--smolvla-live-server-serve-dev.us-east.modal.run"


class ProxyHandler(BaseHTTPRequestHandler):
    def _forward(self, method):
        target_url = MODAL_BASE.rstrip("/") + self.path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None

        headers = {}
        if "Content-Type" in self.headers:
            headers["Content-Type"] = self.headers["Content-Type"]

        req = urllib.request.Request(
            target_url, data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type",
                    resp.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            msg = str(e).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8008), ProxyHandler)
    print("Bridge listening on 0.0.0.0:8008 ->", MODAL_BASE)
    server.serve_forever()
```

### 2.3 Launching the Windows Host Services
0. **Both Terminals**: Navigate to your workspace and activate the virtual environment:
   ```powershell
   .\modal-env\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks `Activate.ps1`, run `.\modal-env\Scripts\modal.exe` directly.)*

1. **Terminal 1**: Start Modal and copy the generated `.modal.run` URL:
   ```powershell
   modal serve modal_server.py
   ```
2. **Terminal 2**: Update `MODAL_BASE` in `bridge.py` and start the proxy:
   ```powershell
   python bridge.py
   ```

---

## 3. Robot Control PC Bring-Up (`192.168.0.101`)

### 3.1 Start the Camera Server (Terminal 1 on Robot PC)
```bash
conda activate lumosteleop
python3 -m teleimager.image_server
```

### 3.2 Verify Health & Checkpoint Path (Terminal 2 on Robot PC)
Wake up the Modal container and retrieve the exact resolved `checkpoint` string:
```bash
IP="192.168.0.100"
curl -s "http://${IP}:8008/health" | python3 -m json.tool
```

### 3.3 Run Guarded Policy Execution
> **Safety Notice:** `DualTouchRobot` energizes and homes both arms on startup. Ensure the workspace is physically clear and keep an operator hand on the hardware E-stop at all times.

```bash
IP="192.168.0.100"
export PYTHONPATH="/home/lumos/lumosteleop/.app/runtime:$PYTHONPATH"

curl -s "http://${IP}:8008/health" > /dev/null && python3 src/station_lite_smolvla_right7_guarded.py   --server "http://${IP}:8008"   --expected-checkpoint "/__modal/volumes/vo-5uH0unrvRZfXunQ57sjRAu/checkpoint_010000/010000/pretrained_model"   --expected-seed-mode fixed   --task "Pick up the red wooden block from the white grid and place it on the brown target area."   --target-color red   --exec-steps 20   --blend-overlap 8   --settled-chunks   --adaptive-refine   --condition-final-grasp   --hold-grasp-pose   --place-after-grasp   --phase-speeds   --smooth-waypoints   --max-observation-age-ms 6000   --hover-z-m 0.200   --fast-grid-approach   --home-takeoff-hover   --direct-place-transport   --trim-idle-tail   --no-async-prefetch   --settle-margin-s 0.10   --settle-drift-sleep-s 0.10   --aerial-approach-speed 0.36   --confirm-startup-home   --confirm-exit-home   --unsafe-exit-action confirm_before_disable   --send-right-chunks   --loops 6
```

---

## 4. Troubleshooting & Cloud Latency Notes

- **`stale_observation` on Chunk 1**: The first `/predict` request after container boot triggers PyTorch/CUDA kernel compilation (~4.4 s). Using `--settled-chunks` with `--max-observation-age-ms 6000` safely accommodates cold-start latency while the arm is stationary.
- **`prefetch_timeout` warnings**: Disable async prefetching (`--no-async-prefetch`) when routing inference over a cloud HTTP bridge so synchronous observation reads occur cleanly after chunk settling.
- **`Adaptive handoff requires --blend-overlap >= 8`**: When `--adaptive-refine` is enabled, `--blend-overlap` must be at least `8` and `--exec-steps - --switch-after` must be `>= --blend-overlap` (e.g., `--exec-steps 20 --blend-overlap 8`).
- **`ModuleNotFoundError: No module named 'xr_teleoperate2'`**: Export the runtime parent directory before launching:
  ```bash
  export PYTHONPATH="/home/lumos/lumosteleop/.app/runtime:$PYTHONPATH"
  ```
