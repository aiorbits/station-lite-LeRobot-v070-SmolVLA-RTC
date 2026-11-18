# Modal T4 Cloud GPU Runbook & Field Troubleshooting Guide

This guide covers the complete end-to-end setup, directory paths, environment activation, 5-terminal workflow, and troubleshooting solutions for running **Station Lite `right7` SmolVLA** and the **`v070_demo` Supervised Web Panel** on a **Modal T4 Cloud GPU** bridged over Ethernet to an offline **Robot PC (`<ROBOT_HOST>`)** via a **Windows Laptop (`<WINDOWS_LAPTOP_IP>`)**.

> **Sanitization Policy:** Local workspace paths, LAN IP addresses, Modal dev URLs, and volume IDs use generic placeholders (`<LOCAL_REPO_DIR>`, `<WINDOWS_LAPTOP_IP>`, `<ROBOT_HOST>`, `<RUNTIME_DIR>`, `<YOUR-WORKSPACE>`, `<VOLUME_ID>`).

---

## 1. End-to-End 5-Terminal Architecture

Because the Robot PC has no direct internet access and blocks inbound LAN connections on custom ports, the system uses **5 coordinated terminals**:

```text
[Windows Laptop: <WINDOWS_LAPTOP_IP>]
  ├── Terminal 1: modal serve deploy/modal/modal_server.py               (Tesla T4 GPU inference)
  ├── Terminal 2: python deploy/modal/bridge.py                          (LAN HTTP proxy :8008 -> Modal HTTPS)
  └── Terminal 3: ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>      (SSH tunnel for browser UI)

[Robot PC: <ROBOT_HOST>]
  ├── Tab 1: python3 -m teleimager.image_server                          (ZMQ camera server on :60000)
  └── Tab 2: station_lite_smolvla_right7_guarded.py OR v070_demo.run_panel
```

---

## 2. Step-by-Step Launch Guide (With Exact `cd` & Env Steps)

### Phase A — Windows Laptop (`<WINDOWS_LAPTOP_IP>`)

1. **Terminal 1 (Start Modal T4 GPU Server):**
   Open PowerShell, navigate into the repository root, set your Modal volume/checkpoint variables, and start Modal:
   ```powershell
   cd <LOCAL_REPO_DIR>\station-lite-LeRobot-v070-SmolVLA-RTC
   $env:MODAL_VOLUME_NAME = "smolvla-storage"
   $env:MODAL_CHECKPOINT_RELPATH = "checkpoint_010000/010000/pretrained_model"
   modal serve deploy/modal/modal_server.py
   ```
   Copy the live `.modal.run` URL printed in the terminal output.

2. **Terminal 2 (Start Local HTTP Reverse Proxy on Port `8008`):**
   Open a second PowerShell window in the repository root, point `MODAL_BASE_URL` to your `.modal.run` URL, and start the bridge:
   ```powershell
   cd <LOCAL_REPO_DIR>\station-lite-LeRobot-v070-SmolVLA-RTC
   $env:MODAL_BASE_URL = "https://<YOUR-WORKSPACE>--smolvla-live-server-serve.modal.run"
   python deploy/modal/bridge.py
   ```
   Ensure Windows Defender Firewall allows inbound TCP connections on port `8008` over your Ethernet adapter.

3. **Terminal 3 (Open SSH Port-Forwarding Tunnel for the Web Panel):**
   Open a third PowerShell window and forward local port `8096` to the Robot PC:
   ```powershell
   ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>
   ```
   Enter your `<USER>` password and leave this window open (it sits silently with a blinking cursor while active).

---

### Phase B — Robot PC (`<ROBOT_HOST>`)

1. **Tab 1 (Start ZMQ Camera Server):**
   Connect via SSH, activate the `lumosteleop` conda environment, and launch `teleimager`:
   ```bash
   conda activate lumosteleop
   python3 -m teleimager.image_server
   ```
   Leave this tab running on port `60000`.

2. **Tab 2 (Navigate to Repo, Wake Modal, & Inspect `/health`):**
   Open a second SSH tab on `<ROBOT_HOST>`, activate `lumosteleop`, `cd` into the repository folder, and query `/health` through the Windows bridge:
   ```bash
   conda activate lumosteleop
   cd ~/station-lite-LeRobot-v070-SmolVLA-RTC
   IP="<WINDOWS_LAPTOP_IP>"

   curl -s "http://${IP}:8008/health" | python3 -m json.tool
   ```
   Copy the exact `"checkpoint"` string returned by `/health` (for example, `/__modal/volumes/<VOLUME_ID>/checkpoint_010000/010000/pretrained_model`).

---

## 3. Running the Robot (Choose One Mode in Robot PC Tab 2)

### Option 1: Multi-Object Supervised Web Dashboard (`v070_demo.run_panel`)

1. **Adjust Cloud Observation Age Limit (One-Time Setup):**
   `v070_demo` defaults to an `800.0` ms local-GPU observation freshness limit, whereas cloud inference over Modal T4 takes ~2.3–2.9 s warmed up (~3.4–4.4 s cold). Raise the limit to `6000.0` ms before launching:
   ```bash
   cd ~/station-lite-LeRobot-v070-SmolVLA-RTC
   python3 - <<'PATCH'
   import pathlib
   for f in [
       "src/v070_demo/live_robot_runner.py",
       "src/v070_demo/contracts.py",
       "src/v070_demo/candidate_bridge.py",
   ]:
       p = pathlib.Path(f)
       p.write_text(p.read_text().replace("800.0", "6000.0"))
   PATCH
   ```

2. **Launch the Web Panel on `127.0.0.1:8096`:**
   ```bash
   pkill -f "v070_demo.run_panel" || true
   sleep 1
   cd ~/station-lite-LeRobot-v070-SmolVLA-RTC
   IP="<WINDOWS_LAPTOP_IP>"
   export PYTHONPATH="./src:<RUNTIME_DIR>:$PYTHONPATH"

   curl -s "http://${IP}:8008/health" > /dev/null && \
   python3 -B -m v070_demo.run_panel \
     --mode supervised \
     --host 127.0.0.1 \
     --port 8096 \
     --server "http://${IP}:8008" \
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

3. **Open in Windows Browser:**
   Navigate to `http://127.0.0.1:8096`, verify all detected blocks on the 3x3 `R1–R9` grid show `QUEUED`, and click **Start**.

---

### Option 2: Single-Block Guarded CLI Execution

To run a single target color directly from the terminal without the web UI:
```bash
cd ~/station-lite-LeRobot-v070-SmolVLA-RTC
IP="<WINDOWS_LAPTOP_IP>"
export PYTHONPATH="<RUNTIME_DIR>:$PYTHONPATH"

curl -s "http://${IP}:8008/health" > /dev/null && \
python3 src/station_lite_smolvla_right7_guarded.py \
  --server "http://${IP}:8008" \
  --expected-checkpoint "<RESOLVED_CHECKPOINT_PATH_FROM_HEALTH>" \
  --expected-seed-mode fixed \
  --task "Pick up the yellow wooden block from the white grid and place it on the brown target area." \
  --target-color yellow \
  --exec-steps 20 \
  --blend-overlap 8 \
  --settled-chunks \
  --adaptive-refine \
  --condition-final-grasp \
  --hold-grasp-pose \
  --place-after-grasp \
  --phase-speeds \
  --smooth-waypoints \
  --max-observation-age-ms 6000 \
  --hover-z-m 0.200 \
  --fast-grid-approach \
  --home-takeoff-hover \
  --direct-place-transport \
  --trim-idle-tail \
  --no-async-prefetch \
  --settle-margin-s 0.10 \
  --settle-drift-sleep-s 0.10 \
  --aerial-approach-speed 0.36 \
  --confirm-startup-home \
  --confirm-exit-home \
  --unsafe-exit-action confirm_before_disable \
  --send-right-chunks \
  --loops 6
```

---

## 4. Field Troubleshooting Reference

| Error / Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| **`can't open file '/home/lumos/src/...'`** | Command was executed from `/home/lumos` instead of inside the repository folder. | Always run `cd ~/station-lite-LeRobot-v070-SmolVLA-RTC` first. |
| **`OSError: [Errno 2] No such file or directory: 'tokenizer'`** | Relative `"tokenizer"` path in `policy_preprocessor.json` fails inside Modal's container. | `deploy/modal/modal_server.py` automatically patches this at container startup to `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. |
| **`Checkpoint mismatch` on startup** | Modal mounts `/data` internally to `/__modal/volumes/<VOLUME_ID>/...`. | Query `GET /health` first and pass the exact returned `"checkpoint"` string to `--expected-checkpoint`. |
| **`TimeoutError: timed out` on `/health` (5 s)** | Idle Modal T4 containers scale to zero and take ~15–25 s to cold-boot. | Prepend `curl -s "http://${IP}:8008/health" > /dev/null &&` so `curl` wakes the container before Python starts. |
| **`ModuleNotFoundError: No module named 'xr_teleoperate2'`** | Vendor robot control SDK lives outside the repo tree. | Export `PYTHONPATH="./src:<RUNTIME_DIR>:$PYTHONPATH"` before launching. |
| **`Request to 127.0.0.1:60000 timed out` (`ImageClient`)** | `teleimager.image_server` is not running on `<ROBOT_HOST>`. | Start `python3 -m teleimager.image_server` in Robot PC Tab 1. *(Note: `Image client has been closed` on client exit only closes the subscriber socket; the server stays alive.)* |
| **`Invalid timing, loop, blend, or margin parameter`** | `--adaptive-refine` requires `--blend-overlap >= 8`, which violates `blend_overlap <= exec_steps - switch_after` (`8 <= 15 - 10`). | Pass `--exec-steps 20 --blend-overlap 8` together. |
| **`reject=stale_observation` (`obs_age > 800 ms / 2500 ms`)** | Cloud round-trip over Modal T4 takes ~2.3–2.9 s warmed up (~3.4–4.4 s cold). | Use `--max-observation-age-ms 6000 --no-async-prefetch` on CLI, or patch `800.0` -> `6000.0` in `src/v070_demo/` for the web panel. |
| **`NEEDS_OPERATOR \| insufficient_clearance:0.025m<0.030m`** | Planner requires **>= 30 mm (`0.030 m`)** edge-to-edge clearance around every block on `R1–R9`. | Place blocks in **non-adjacent grid cells** (`R1`, `R3`, `R7`, `R9`) with **>= 3.5 cm** spacing. |
| **`Target outside approach candidate envelope` (2 same-color cubes)** | Two identical-color blocks split visual conditioning and break `cam_high` single-cell locking. | Place **at most one block per color** (`1 Red`, `1 Yellow`, `1 Blue`) on the grid. |
