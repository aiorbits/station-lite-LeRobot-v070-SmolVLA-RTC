# Deployment & Verification Guide

All paths, hostnames, checkpoints, and SDK directories are parameterized through environment variables and placeholders. Never hardcode LAN IPs, user home paths, or robot SSH credentials into tracked files.

---

## 1. Environment Variables

| Variable | Side | Purpose | Example |
| --- | --- | --- | --- |
| `SMOLVLA_CHECKPOINT` | Inference Server & Guarded Client | Path to local `pretrained_model` directory | `/path/to/checkpoints/right7_pretrained_model` |
| `HF_HOME` / `HF_HUB_OFFLINE` | Inference Server | Local cached HuggingFace tokenizer/VLM directory with offline lock (`1`) | `./cache/hf`, `1` |
| `EXPECTED_CONDA_PREFIX` | Inference Server (`--verify-env`) | Optional strict virtual env verification prefix | `/opt/conda/envs/station_lerobot_v070` |
| `STARTOUCH_SDK_PY_DIR` | Robot Client / G4 Dry-Run | Directory containing proprietary vendor Python bindings | `/opt/robot_sdk/interface_py` |
| `STATION_RUNTIME_DIR` | Robot Client | Directory containing station camera/robot wrapper (`DualTouchRobot`, `ImageClient`) | `/opt/station_runtime` |
| `ROBOT_PYTHON` | Robot Client / G4 Dry-Run | Absolute Python executable in the validated vendor environment | `/path/to/vendor_env/bin/python3` |
| `SDK_BUNDLE` | G4 Dry-Run | Approved, separately supplied action bundle for the exact candidate | `/path/to/g4_fast_cycle_sdk_bundle.json` |

---

## 2. Three-Terminal Bring-Up (Hardware Session)

### Terminal 1 — Isolated Inference Workstation (`<POLICY_HOST>`, CUDA GPU Required)

```bash
export SMOLVLA_CHECKPOINT=/path/to/checkpoints/right7_pretrained_model
export HF_HOME=/path/to/offline_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 src/station_lite_smolvla_right7_server.py \
  --checkpoint "$SMOLVLA_CHECKPOINT" \
  --host 0.0.0.0 \
  --port 8011 \
  --seed 42 \
  --seed-mode fixed \
  --offline \
  --no-verify-env
```

> [!WARNING]
> Port `8010` is blocked (`Refusing port 8010`) by default inside `parse_server_args` to protect any parallel production `0.6.2` baseline server.

### Inference Readiness Before Robot Motion

On 16 November, `/health` succeeded before the first real prediction was ready: cold inference took 1529.7 ms and the unchanged 800 ms observation-age guard rejected it. Before connecting/moving the robot, issue two synthetic `/predict` requests to the isolated CUDA service, inspect their latency, and confirm camera readiness separately. The following sends no robot commands and does not establish IK or motion admission:

```bash
export POLICY_SERVER_URL=http://127.0.0.1:8011
python3 - <<'WARMUP'
import base64, io, json, os, time, urllib.request
from PIL import Image
buf = io.BytesIO()
Image.new("RGB", (640, 480), (20, 20, 20)).save(buf, format="JPEG")
image = base64.b64encode(buf.getvalue()).decode("ascii")
for seq in (0, 1):
    payload = {
        "schema": "station_lite_smolvla_right7_request_v1",
        "task": "Pick up the yellow wooden block from the white grid and place it on the brown target area.",
        "session_id": "warmup_probe", "seq": seq,
        "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07],
        "images_jpeg_base64": {
            "observation.images.cam_high": image,
            "observation.images.cam_right_wrist": image,
        },
    }
    req = urllib.request.Request(
        os.environ["POLICY_SERVER_URL"].rstrip("/") + "/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=10) as resp:
        out = json.load(resp)
    print(seq, "rtt_ms=", (time.monotonic()-started)*1000,
          "inference_ms=", out["inference_ms"])
WARMUP
```

The observed warmed-up inference times were 115.8–131.9 ms; two calls are a readiness check, not a universal latency guarantee. Do not relax freshness limits to accommodate slow inference. Robot-side endpoints must point to the intended policy host; use a distinct session for each subsequent run.

### Terminal 2 — Pre-Run Hardware Verification (`<ROBOT_HOST>`, CAN-Isolated Dry-Run)

Always validate updated action bundles against the real vendor kinematics planner inside an isolated network namespace (no CAN socket access, no motor actuation):

**Required external input:** `g4_fast_cycle_sdk_bundle.json` is not included in this source-only export because it contains recording-derived robot states and trajectories. The authoring utility `src/g4_execution_adapter_gate.py` generates it from audited replay fixtures, baseline identity, the candidate source, and the CUDA checkpoint; those private inputs must be prepared separately. This checkout alone cannot reproduce the archived 13-case evaluation. Obtain or generate a reviewed bundle for the intended scene, calibration, source, and runtime profile, retain its SHA-256, then transfer it separately to the robot host. Do not replace it with synthetic test data for motion admission.

```bash
export STARTOUCH_SDK_PY_DIR=/path/to/startouch_sdk/interface_py
export ROBOT_PYTHON=/path/to/vendor_env/bin/python3
export SDK_BUNDLE=/path/to/g4_fast_cycle_sdk_bundle.json

sudo unshare --net "$ROBOT_PYTHON" \
  src/g4_sdk_dryrun_gate.py \
  --sdk-py-dir "$STARTOUCH_SDK_PY_DIR" \
  --input "$SDK_BUNDLE" \
  --output g4_fast_cycle_sdk_dryrun_robot.json
```

Confirm `"all_passed": true` and `"cleanup_returned": true` before energizing hardware for live motion.

### Terminal 3 — Guarded Fast-Cycle Hybrid Execution (`<ROBOT_HOST>`)

> [!IMPORTANT]
> **Reference-station pilot completed on 16 November: 13/13 isolated planning cases and 4/4 warmed-up single-block trials.** The flags below select Scheme 1 direct transport, Scheme 3 release-tail compression, and aerial speed `0.36`. These results do not validate other scenes or stations. Complete inference warmup, camera readiness, matched planning/lifecycle checks and on-site authorization for each intended configuration. This is not an unattended launch recipe.

> [!CAUTION]
> Instantiating `DualTouchRobot` may energize motors on startup, and process teardown (`ArmController::cleanup()`) drops holding torque via `Startouch::disable_all()`. Ensure workspace is physically clear before running `--confirm-startup-home` / `--confirm-exit-home`, keep an operator hand on the hardware E-stop, and stay within authorized station bounds.

```bash
export STARTOUCH_SDK_PY_DIR=/path/to/startouch_sdk/interface_py
export STATION_RUNTIME_DIR=/path/to/station_runtime
export SMOLVLA_CHECKPOINT=/path/to/checkpoints/right7_pretrained_model
export ROBOT_PYTHON=/path/to/vendor_env/bin/python3

"$ROBOT_PYTHON" src/station_lite_smolvla_right7_guarded.py \
  --server "http://<POLICY_HOST>:8011" \
  --expected-checkpoint "$SMOLVLA_CHECKPOINT" \
  --expected-seed-mode fixed \
  --task "Pick up the yellow wooden block from the white grid and place it on the brown target area." \
  --target-color yellow \
  --settled-chunks \
  --adaptive-refine \
  --condition-final-grasp \
  --hold-grasp-pose \
  --place-after-grasp \
  --phase-speeds \
  --smooth-waypoints \
  --max-observation-age-ms 800 \
  --hover-z-m 0.200 \
  --fast-grid-approach \
  --home-takeoff-hover \
  --direct-place-transport \
  --trim-idle-tail \
  --async-prefetch \
  --settle-margin-s 0.10 \
  --settle-drift-sleep-s 0.10 \
  --aerial-approach-speed 0.36 \
  --confirm-startup-home \
  --confirm-exit-home \
  --unsafe-exit-action confirm_before_disable \
  --send-right-chunks \
  --loops 6
```

Disable optional optimizations individually with `--no-direct-place-transport`, `--no-home-takeoff-hover`, `--no-fast-grid-approach`, `--no-trim-idle-tail`, or `--no-async-prefetch` to revert smoothly toward the legacy settled baseline.

---

## 3. Multi-Object Supervised Panel (`v070_demo.run_panel`, v1.0)

After completing inference warmup (Section 2) and verifying Stage D3 CAN-isolated SDK dry-run admission (`--input "$D3_SDK_BUNDLE"` -> `"all_passed": true, "cleanup_returned": true`), launch the single-owner supervised web panel on `<ROBOT_HOST>`:

```bash
PYTHONPATH="./src" python3 -B -m v070_demo.run_panel \
  --mode supervised \
  --host 127.0.0.1 \
  --port 8095 \
  --server "http://<POLICY_HOST>:8011" \
  --img-host 127.0.0.1 \
  --dry-run-report /path/to/g4_demo_d3_sdk_dryrun_robot.json \
  --expected-bundle-sha256 a7aa4f3d97a69207d51be8df4c29794f2cbe33e26fbd3e3fff55f0bc6d71f478 \
  --expected-checkpoint "$SMOLVLA_CHECKPOINT" \
  --expected-seed-mode fixed \
  --use-clear-obs-retreat \
  --batch-deadline-s 600 \
  --admit-hardware \
  --confirm-startup-home \
  --confirm-exit-home \
  --unsafe-exit-action confirm_before_disable \
  --run-dir ./logs/demo_d3_run \
  --snapshot-dir ./logs/demo_d3_run/snapshots \
  --lifecycle-audit-json ./logs/demo_d3_run/lifecycle_audit.json
```

Use `--mode mock --host 127.0.0.1 --port 8095` to exercise the full panel UI and session state machine offline with zero robot or GPU dependencies.


