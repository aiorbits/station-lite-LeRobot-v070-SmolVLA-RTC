# Cloud GPU Runbook & Field Troubleshooting Guide (Modal T4 or Google Cloud L4)

This runbook covers the day-to-day workflow and field fixes for running the **Station Lite `right7` SmolVLA** policy (single-block client and the **`v070_demo` Supervised Web Panel**) with the model served from a **cloud GPU**. Two server setups have been tested; the robot side is the same for both.

| | **Option A: Modal T4 + laptop bridge** | **Option B: Google Cloud L4 VM (direct)** |
|---|---|---|
| Use when | The Robot PC has **no internet** | The Robot PC **has internet** (e.g. router with SIM) |
| Server runs on | Modal serverless T4 (`modal serve` on the Windows laptop) | Compute Engine VM with 1x NVIDIA L4, server in `tmux` |
| Robot talks to | `http://<WINDOWS_LAPTOP_IP>:8008` (bridge -> Modal HTTPS) | `http://<VM_IP>:8011` (straight to the VM) |
| `--expected-checkpoint` | The `/__modal/volumes/...` path from `/health` | The VM path, e.g. `<VM_HOME>/smolvla/checkpoints/010000/pretrained_model` |
| Measured speed | ~2.3–2.9 s per `/predict` round trip warm (~3.4–4.4 s cold) | ~360 ms model time warm (~2 s first call); robot -> VM `/health` ~0.04–0.06 s |
| One-time setup | `docs/MODAL_DEPLOYMENT_GUIDE.md` | `docs/lego_sorting/google_cloud_setup.md` |

> **Sanitization notice:** LAN IPs, cloud IPs, workspace URLs, volume IDs and home paths are placeholders: `<ROBOT_HOST>`, `<WINDOWS_LAPTOP_IP>`, `<VM_IP>`, `<VM_HOME>`, `<RUNTIME_DIR>`, `<YOUR-WORKSPACE>`. On the robot used here, `<RUNTIME_DIR>` is the lumosteleop runtime folder (`.../lumosteleop/.app/runtime`).

---

## 1. Architecture & Terminal Map

### Option A: Modal T4 + laptop bridge

```text
[Windows Laptop: <WINDOWS_LAPTOP_IP>]
  ├── Terminal 1: modal serve deploy/modal/modal_server.py   (Hosts SmolVLA on Tesla T4 GPU)
  ├── Terminal 2: python deploy/modal/bridge.py              (LAN HTTP reverse proxy :8008 -> Modal HTTPS)
  └── Terminal 3: ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>  (Forwards panel UI to browser)

[Robot PC: <ROBOT_HOST>]
  ├── Tab 1: camera server                                   (Streams cam_high & cam_right_wrist on :60000)
  └── Tab 2: single-block client OR v070_demo.run_panel      (Robot controller)
```

### Option B: Google Cloud L4 VM (direct)

```text
[Google Cloud VM: <VM_IP>]      SmolVLA server on :8011, in tmux session "server"
        ^
        |  robot's own internet (router / SIM)
[Robot PC: <ROBOT_HOST>]
  ├── tmux window 1: camera server
  └── tmux window 0: single-block client OR v070_demo.run_panel

[Windows Laptop]  (Ethernet to the robot only)
  ├── Terminal 1: ssh <USER>@<ROBOT_HOST>                    (Robot shell)
  └── Terminal 2: ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>  (Panel UI, only for run_panel)
```

**Which prompt is which machine (Option B):** `...@<VM name>` = the VM, use it **only** for the server. `lumos@<robot>` = the robot (cameras, runs, panel). `PS C:\...>` = the laptop (`ssh`, the 8096 tunnel, `scp`).

---

## 2. Step-by-Step Launch Checklist

### Step A — Start the model server (pick one option)

#### Option A: Modal T4 (Windows laptop)

1. **Terminal 1 (start the Modal T4 server):**
   ```powershell
   $env:MODAL_VOLUME_NAME = "smolvla-storage"
   $env:MODAL_CHECKPOINT_RELPATH = "checkpoint_010000/010000/pretrained_model"
   modal serve deploy/modal/modal_server.py
   ```
   Copy the live `.modal.run` URL printed by Modal.

2. **Terminal 2 (start the LAN HTTP bridge on port `8008`):**
   ```powershell
   $env:MODAL_BASE_URL = "https://<YOUR-WORKSPACE>--smolvla-live-server-serve.modal.run"
   python deploy/modal/bridge.py
   ```

3. On the robot, get the checkpoint string Modal reports and set the server address:
   ```bash
   curl -s http://<WINDOWS_LAPTOP_IP>:8008/health | python3 -m json.tool
   export SERVER="http://<WINDOWS_LAPTOP_IP>:8008"
   export SMOLVLA_CHECKPOINT="<RESOLVED_CHECKPOINT_PATH_FROM_HEALTH>"   # /__modal/volumes/<VOLUME_ID>/...
   ```

#### Option B: Google Cloud L4 VM

1. Console -> Compute Engine -> VM instances -> tick the VM -> **START**, wait ~2 min.
2. If the VM has the `startup-script` metadata (see `docs/lego_sorting/google_cloud_setup.md`, section 7), the server starts by itself; skip to 3. Otherwise open the VM's **SSH** window and run:
   ```bash
   tmux new -A -s server
   ```
   ```bash
   conda activate smolvla && cd ~/smolvla && export HF_HOME=~/smolvla/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && python src/station_lite_smolvla_right7_server.py --checkpoint ~/smolvla/checkpoints/010000/pretrained_model --host 0.0.0.0 --port 8011 --seed 42 --seed-mode fixed --offline --no-verify-env
   ```
   Wait for `[ready] ... gpu=NVIDIA L4 bind=0.0.0.0:8011`, then **Ctrl+B, D** to leave it running. Never press Ctrl+C there.
3. On the robot, check the server and set the address. You want `200`:
   ```bash
   curl -s -o /dev/null --max-time 5 -w "%{http_code} %{time_total}s\n" http://<VM_IP>:8011/health
   export SERVER="http://<VM_IP>:8011"
   export SMOLVLA_CHECKPOINT="<VM_HOME>/smolvla/checkpoints/010000/pretrained_model"   # must equal /health "checkpoint"
   ```

---

### Step B — Robot PC (`<ROBOT_HOST>`), same for both options

Use `tmux new -A -s work` on the robot so an SSH drop loses nothing (`Ctrl+B, 1` / `Ctrl+B, 0` to switch windows, `Ctrl+B, C` to make one).

1. **Camera server (tab / tmux window 1).** Check first; start only if this prints `0`:
   ```bash
   pgrep -fc start_xr_img_server
   ```
   ```bash
   conda activate lumosteleop
   python3 -m teleimager.image_server
   ```
   (The lumosteleop launcher `cd ~/lumosteleop; ./lumosteleop image-server` also works.) Never Ctrl+C this window during a run.

2. **Robot controller (tab / tmux window 0).**
   ```bash
   cd ~/smolvla
   export PYTHONPATH="./src:$HOME/startouch_sdk/interface_py:<RUNTIME_DIR>:$PYTHONPATH"
   ```

   **2a. Single-block run** (one block in a white grid cell, hand on the E-stop). Paste as **one line**:
   ```bash
   curl -s "$SERVER/health" > /dev/null && python3 src/station_lite_smolvla_right7_guarded.py --server "$SERVER" --expected-checkpoint "$SMOLVLA_CHECKPOINT" --expected-seed-mode fixed --task "Pick up the red wooden block from the white grid and place it on the brown target area." --target-color red --exec-steps 20 --blend-overlap 8 --settled-chunks --adaptive-refine --condition-final-grasp --hold-grasp-pose --place-after-grasp --phase-speeds --smooth-waypoints --max-observation-age-ms 6000 --hover-z-m 0.200 --fast-grid-approach --home-takeoff-hover --direct-place-transport --trim-idle-tail --no-async-prefetch --settle-margin-s 0.10 --settle-drift-sleep-s 0.10 --aerial-approach-speed 0.36 --confirm-startup-home --confirm-exit-home --unsafe-exit-action confirm_before_disable --send-right-chunks --loops 6 --snapshot-dir ./logs/run_$(date +%H%M%S)
   ```
   - Other colours: change the colour in `--task` **and** `--target-color` (`red` / `yellow` / `blue`).
   - Good sign near the start: `[HIGH CAM 0] locked target_color='red' at cell=R…`.

   **2b. Supervised web panel** (instead of 2a). First, on the **laptop**, open the tunnel and leave it open (it stays silent):
   ```powershell
   ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>
   ```
   Then on the robot:
   ```bash
   python3 -B -m v070_demo.run_panel \
     --mode supervised \
     --host 127.0.0.1 \
     --port 8096 \
     --server "$SERVER" \
     --expected-checkpoint "$SMOLVLA_CHECKPOINT" \
     --expected-seed-mode fixed \
     --dry-run-report <PATH_TO_PASSED_D3_DRYRUN_REPORT.json> \
     --expected-bundle-sha256 any \
     --admit-hardware \
     --confirm-startup-home \
     --confirm-exit-home \
     --unsafe-exit-action confirm_before_disable \
     --use-clear-obs-retreat \
     --run-dir ./logs/panel_$(date +%H%M%S)
   ```
   Wait for `[D3 ADMITTED] ...`, then open `http://127.0.0.1:8096` in the laptop browser. Use `--mode mock` to practise with no robot motion. Over a cloud link the panel also needs the `800.0` -> `6000.0` change (see the `stale_observation` row in section 3).

### Step C — End of session
1. Ctrl+C the run / panel in robot window 0 (support the arm first if it asks).
2. **Option A:** stop `modal serve` and `bridge.py`. **Option B:** Console -> tick the VM -> **STOP** (a running VM bills by the hour; the disk and static IP still cost a little while stopped).

---

## 3. Issues Encountered & Field Solutions

### Robot / client (both options)

| Issue / Error Message | Root Cause | Exact Fix |
| :--- | :--- | :--- |
| **`ModuleNotFoundError: No module named 'xr_teleoperate2'`** | The vendor robot control package lives outside the repository tree. | Export `PYTHONPATH="./src:$HOME/startouch_sdk/interface_py:<RUNTIME_DIR>:$PYTHONPATH"` before launching `station_lite_smolvla_right7_guarded.py` or `v070_demo.run_panel`. |
| **`Request to 127.0.0.1:60000 timed out` (`ImageClient`)** | The camera server was not running on the Robot PC. | Start the camera server in its own tab / tmux window before the client (`pgrep -fc start_xr_img_server` = `1` means it is up). *(`Image client has been closed` on client exit only closes the subscriber socket; the server stays alive.)* |
| **`Invalid timing, loop, blend, or margin parameter`** | `--adaptive-refine` requires `--blend-overlap >= 8`, which fails the default bound `blend_overlap <= exec_steps - switch_after` (`8 <= 15 - 10`). | Pass `--exec-steps 20 --blend-overlap 8` together so `8 <= 20 - 10`. |
| **`reject=stale_observation` (`obs_age > 800 ms` or `2500 ms`)** | A cloud round trip is slower than the local-GPU freshness bounds. | Single-block runs: `--max-observation-age-ms 6000 --no-async-prefetch`. `v070_demo`: change the hardcoded `800.0` ms bound in `live_robot_runner.py`, `contracts.py` and `candidate_bridge.py` to `6000.0`. |
| **`prefetch_timeout` during `--async-prefetch`** | Background prefetch threads use a short internal timeout tuned for local LAN GPUs. | Pass `--no-async-prefetch` over a cloud link so the client reads synchronously after the arm settles (`state_drift_rad = 0.0000`). |
| **Motion is stop-go between chunks** | With prefetch off, the arm waits for each cloud reply. | Option B (fast L4 reply) can try `--async-prefetch --max-observation-age-ms 800`; otherwise a faster GPU. |
| **`[APPROACH GUARD] ... outside approach candidate envelope` / no `[HIGH CAM 0] locked`** | The overhead camera did not lock onto the block. | Block inside a white cell, arm at Home or `CLEAR_OBS_POSE6` (not blocking the view), one block per colour. |
| **`Target outside approach candidate envelope` with 2 same-colour cubes** | Two identical-colour blocks split visual conditioning, and `cam_high` grid locking falls back to raw predictions near workspace boundaries. | Place **at most one block per colour** (`1 Red`, `1 Yellow`, `1 Blue`) in separate white grid cells (`R1–R9`). |
| **`NEEDS_OPERATOR \| insufficient_clearance:0.025m<0.030m`** | The multi-object planner needs **at least 30 mm** edge-to-edge spacing around each block on the 3x3 grid. | Use **non-adjacent cells** (e.g. `R1`, `R3`, `R7`) with **>= 3.5 cm** between edges. |
| **`--ik-preflight-only` rejects all 50 targets** | Normal from the folded rest pose (joint idx 2 at its `0.000` limit). | Skip the preflight; the full run moves Home -> hover itself. |
| **Long command arrives mangled (e.g. `--tar>` mid-line)** | Multi-line paste into SSH broke up. | Paste the run as **one line**. |

### Panel / SSH / laptop (both options)

| Issue / Error Message | Root Cause | Exact Fix |
| :--- | :--- | :--- |
| **Browser `Refused to connect` / port `8090` conflict on Robot PC** | Port `8090` is taken by the station's own REST service (`{"code": 404, "message": "接口不存在。"}`), and the robot firewall blocks direct LAN access to `8095`/`8096`. | Bind `run_panel` to `--host 127.0.0.1 --port 8096` and forward it with `ssh -N -L 8096:127.0.0.1:8096 <USER>@<ROBOT_HOST>`. |
| **`Address already in use` (port 8096)** | The tunnel command was run **inside the robot** instead of on the laptop. | On the robot: `ss -tlnp \| grep 8096`, then `kill <pid of ssh>`. Run the tunnel from the laptop's PowerShell. |
| **SSH to the robot keeps dropping** | Laptop Wi-Fi joined the robot router's Wi-Fi as well as the cable (two links into one subnet). | Forget that Wi-Fi on the laptop; cable for the robot only. Always work inside `tmux`. |

### Option A only (Modal)

| Issue / Error Message | Root Cause | Exact Fix |
| :--- | :--- | :--- |
| **`OSError: [Errno 2] No such file or directory: 'tokenizer'`** | `policy_preprocessor.json` in `pretrained_model/` uses a relative `"tokenizer"` string that fails inside Modal's Linux container. | `deploy/modal/modal_server.py` patches `policy_preprocessor.json` at container start to point to `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (or `ckpt_dir / "tokenizer"` if present). |
| **`Checkpoint mismatch` on startup** | Modal resolves `/data` volume mounts to `/__modal/volumes/<VOLUME_ID>/...`, so comparison against `/data/...` fails. | Query `GET /health` first and pass the exact `"checkpoint"` path to `--expected-checkpoint`. |
| **`TimeoutError: timed out` on `/health` (5 s)** | Modal scales idle T4 containers to zero; cold boot + loading weights takes ~15–25 s. | Run `curl -s http://<WINDOWS_LAPTOP_IP>:8008/health > /dev/null` before the client so `curl` absorbs the cold boot. |

### Option B only (Google Cloud)

| Issue / Error Message | Root Cause | Exact Fix |
| :--- | :--- | :--- |
| **`curl .../health` prints nothing or `000`** | VM stopped, server not running, or the robot's public IP moved outside the firewall rule. | Check the VM is RUNNING; on the robot run `curl -s ifconfig.me` and make sure it is inside the firewall rule's source range (widen it if not); otherwise start the server (Step A, Option B). |
| **`Could not find conda environment: smolvla`** | The server command was typed on the robot. | Run it in the VM's SSH window. |
| **Browser SSH "Connection via Cloud Identity-Aware Proxy failed" / refused** | Usually the VM is stopped. | START it (`default-allow-ssh` already allows tcp:22). |
| **Server dies when the browser SSH window closes** | It was started outside `tmux`, or Ctrl+C was pressed. | Start it in `tmux new -A -s server` and detach with Ctrl+B, D, or use the `startup-script`. |
| **`We couldn't connect to 'https://huggingface.co'` (offline start)** | Offline loading asks for revision `main`, which the pinned cache lacks. | Add `refs/main` pointing at the pinned SmolVLM2 revision (see `google_cloud_setup.md`, section 3). |

---

## 4. Operational & Hardware Safety Tips

1. **Always warm up before a batch:** send at least one `/health` and two dummy `/predict` probes after the server starts so CUDA kernel compilation finishes before any robot motion.
2. **Verify the overhead camera lock before confirming Home:** watch for `[HIGH CAM 0] locked target_color='...' at cell=R...` and keep the forearm at Home or `CLEAR_OBS_POSE6` so it does not occlude or shadow the 3x3 grid (`R1–R9`).
3. **Support the right arm before releasing safe-hold pauses:** `DualTouchRobot` / `ArmController::cleanup()` calls `Startouch::disable_all()` on exit, so holding torque drops immediately when you press `Enter` or `Ctrl+C` at a `CANDIDATE SAFE HOLD PAUSE` prompt. Support the arm with one hand (or engage the E-stop) first.
4. **Keep the server side running across robot restarts:** restarting the robot script or panel does **not** need a restart of `modal serve` / `bridge.py` (Option A) or the VM server (Option B); `ImageClient` reconnects on the next run.
5. **Option B: stop the VM when you are done.** The GPU bills only while the VM is running.
