# Modal T4 Cloud Runbook & Field Troubleshooting Guide

This runbook documents the complete multi-terminal workflow and field troubleshooting solutions for running the **Station Lite `right7` SmolVLA** policy and **`v070_demo` Supervised Web Panel** on a **Modal T4 Cloud GPU** bridged over Ethernet to an offline **Robot PC (`<ROBOT_HOST>`)** via a **Windows Laptop (`<WINDOWS_LAPTOP_IP>`)**.

> **Sanitization Notice:** All local file paths, LAN IP addresses, Modal workspace URLs, and volume IDs use placeholders (`<WINDOWS_LAPTOP_IP>`, `<ROBOT_HOST>`, `<RUNTIME_DIR>`, `<YOUR-WORKSPACE>`).

---

## 1. Complete 5-Terminal Architecture & Startup Map

Because the Robot PC has no direct internet access and restricts inbound LAN ports, the deployment uses **5 coordinated terminals** across two machines:

```text
[Windows Laptop: <WINDOWS_LAPTOP_IP>]
  ├── Terminal 1: modal serve deploy/modal/modal_server.py   (Hosts SmolVLA on Tesla T4 GPU)
  ├── Terminal 2: python deploy/modal/bridge.py              (LAN HTTP reverse proxy :8008 -> Modal HTTPS)
  └── Terminal 3: ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>  (Forwards panel UI to browser)

[Robot PC: <ROBOT_HOST>]
  ├── Tab 1: python3 -m teleimager.image_server              (Streams cam_high & cam_right_wrist on :60000)
  └── Tab 2: python3 -B -m v070_demo.run_panel ...           (Runs supervised web panel & robot controller)
```

---

## 2. Step-by-Step Launch Checklist

### Step A — Windows Laptop (`<WINDOWS_LAPTOP_IP>`)

1. **Terminal 1 (Start Modal T4 Server):**
   ```powershell
   \$env:MODAL_VOLUME_NAME = "smolvla-storage"
   \$env:MODAL_CHECKPOINT_RELPATH = "checkpoint_010000/010000/pretrained_model"
   modal serve deploy/modal/modal_server.py
   ```
   Copy the live `.modal.run` URL printed by Modal.

2. **Terminal 2 (Start LAN HTTP Bridge on Port `8008`):**
   ```powershell
   \$env:MODAL_BASE_URL = "https://<YOUR-WORKSPACE>--smolvla-live-server-serve.modal.run"
   python deploy/modal/bridge.py
   ```

3. **Terminal 3 (Open SSH Tunnel for the Web Panel):**
   ```powershell
   ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>
   ```
   *(Leave this window open once authenticated; it sits silently while forwarding `127.0.0.1:8096`.)*

---

### Step B — Robot PC (`<ROBOT_HOST>`)

1. **Tab 1 (Start Camera Server):**
   ```bash
   conda activate lumosteleop
   python3 -m teleimager.image_server
   ```

2. **Tab 2 (Wake Up Modal & Launch Supervised Web Panel):**
   First, inspect `/health` through the bridge to get Modal's resolved volume path:
   ```bash
   curl -s http://<WINDOWS_LAPTOP_IP>:8008/health | python3 -m json.tool
   ```
   Copy the exact `"checkpoint"` string (e.g., `/__modal/volumes/<VOLUME_ID>/checkpoint_010000/010000/pretrained_model`), then start the panel:
   ```bash
   cd ~/station-lite-LeRobot-v070-SmolVLA-RTC
   export PYTHONPATH="./src:<RUNTIME_DIR>:\$PYTHONPATH"

   python3 -B -m v070_demo.run_panel \
     --mode supervised \
     --host 127.0.0.1 \
     --port 8096 \
     --server "http://<WINDOWS_LAPTOP_IP>:8008" \
     --expected-checkpoint "<RESOLVED_CHECKPOINT_PATH_FROM_HEALTH>" \
     --expected-seed-mode fixed \
     --dry-run-report <PATH_TO_PASSED_D3_DRYRUN_REPORT.json> \
     --expected-bundle-sha256 any \
     --admit-hardware \
     --confirm-startup-home \
     --confirm-exit-home \
     --unsafe-exit-action confirm_before_disable \
     --use-clear-obs-retreat
   ```

3. **Open Browser on Windows Laptop:**
   Navigate to `http://127.0.0.1:8096`.

---

## 3. Issues Encountered & Field Solutions

| Issue / Error Message | Root Cause | Exact Fix |
| :--- | :--- | :--- |
| **`OSError: [Errno 2] No such file or directory: 'tokenizer'`** | `policy_preprocessor.json` inside `pretrained_model/` uses a relative `"tokenizer"` string that fails to resolve inside Modal's Linux container. | `deploy/modal/modal_server.py` patches `policy_preprocessor.json` at container startup to point to `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (or `ckpt_dir / "tokenizer"` if present). |
| **`Checkpoint mismatch` on startup** | Modal resolves `/data` volume mounts internally to `/__modal/volumes/<VOLUME_ID>/...`, causing exact string comparison against `/data/...` to fail. | Query `GET /health` first and pass the exact returned `"checkpoint"` path to `--expected-checkpoint`. |
| **`TimeoutError: timed out` on `/health` (5 s)** | Modal scales idle T4 containers down to zero; cold-booting the container and loading weights onto GPU takes ~15–25 seconds. | Run `curl -s http://<WINDOWS_LAPTOP_IP>:8008/health > /dev/null` before launching the client so `curl` absorbs the cold-boot wait. |
| **`ModuleNotFoundError: No module named 'xr_teleoperate2'`** | The vendor robot control package lives outside the repository tree. | Export `PYTHONPATH="./src:<RUNTIME_DIR>:$PYTHONPATH"` before launching `station_lite_smolvla_right7_guarded.py` or `v070_demo.run_panel`. |
| **`Request to 127.0.0.1:60000 timed out` (`ImageClient`)** | The `teleimager` ZMQ camera server was not running on the Robot PC. | Start `python3 -m teleimager.image_server` in a dedicated terminal tab on `<ROBOT_HOST>` before starting the client. *(Note: `Image client has been closed` on client exit only closes the subscriber socket; the server stays alive.)* |
| **`Invalid timing, loop, blend, or margin parameter`** | Enabling `--adaptive-refine` requires `--blend-overlap >= 8`, which fails the default bound `blend_overlap <= exec_steps - switch_after` (`8 <= 15 - 10`). | Pass `--exec-steps 20 --blend-overlap 8` together so `8 <= 20 - 10`. |
| **`reject=stale_observation` (`obs_age > 800 ms` or `2500 ms`)** | Round-trip inference over Ethernet -> Windows Bridge -> Modal T4 takes **~2.3–2.9 s** warmed up (and **~3.4–4.4 s** on the first cold `/predict`), exceeding local-GPU freshness bounds. | For single-block CLI runs, pass `--max-observation-age-ms 6000 --no-async-prefetch`. For `v070_demo`, update the hardcoded `800.0` ms bound in `live_robot_runner.py`, `contracts.py`, and `candidate_bridge.py` to `6000.0` when running over cloud bridges. |
| **`prefetch_timeout` during `--async-prefetch`** | Background prefetch threads use a short internal timeout tuned for local LAN GPUs. | Pass `--no-async-prefetch` when running over a WAN/cloud bridge so the client uses synchronous reads after the arm settles (`state_drift_rad = 0.0000`). |
| **`NEEDS_OPERATOR \| insufficient_clearance:0.025m<0.030m`** | The multi-object planner requires **at least 30 mm (`0.030 m`)** of edge-to-edge spacing around each block on the 3x3 white grid (`R1–R9`) to avoid finger collisions. | Place blocks in **non-adjacent grid cells** (e.g., `R1`, `R3`, `R7`) with **>= 3.5 cm** of empty space between edges. |
| **`Target outside approach candidate envelope` when testing 2 same-color cubes** | Placing two identical-color blocks on the grid splits visual conditioning and causes `cam_high` grid locking to fall back to raw predictions near workspace boundaries. | Place **at most one block per color** (`1 Red`, `1 Yellow`, `1 Blue`) inside separate white grid cells (`R1–R9`). |
| **Browser `Refused to connect` / Port `8090` conflict on Robot PC** | Port `8090` is occupied by an existing station REST service (`{"code": 404, "message": "接口不存在。"}`), and the Robot PC firewall blocks direct LAN access to custom ports (`8095`/`8096`). | Bind `run_panel` to `--host 127.0.0.1 --port 8096` and forward port `8096` via SSH (`ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>`). |

---

## 4. Operational & Hardware Safety Tips

1. **Always Warm Up Before Running a Batch:** Send at least one `/health` and two dummy `/predict` probes after `modal serve` boots so CUDA kernel compilation finishes before any robot motion begins.
2. **Verify Overhead Camera Lock Before Confirming Home:** Watch the startup log for `[HIGH CAM 0] locked target_color='...' at cell=R...` and ensure the robot forearm is parked at Home or `CLEAR_OBS_POSE6` so it does not occlude or shadow the 3x3 white grid (`R1–R9`).
3. **Support the Right Arm Before Releasing Safe-Hold Pauses:** Because `DualTouchRobot` / `ArmController::cleanup()` calls `Startouch::disable_all()` on process exit, holding torque drops immediately when you press `Enter` or `Ctrl+C` at a `CANDIDATE SAFE HOLD PAUSE` prompt. Always support the right arm physically with one hand (or engage the E-stop) before releasing.
4. **Keep `bridge.py` Running Across Restarts:** If you restart the robot script or web panel on `<ROBOT_HOST>`, you do **not** need to restart `modal serve` or `bridge.py` on Windows—only `ImageClient` inside the robot process closes and reconnects automatically on the next run.
