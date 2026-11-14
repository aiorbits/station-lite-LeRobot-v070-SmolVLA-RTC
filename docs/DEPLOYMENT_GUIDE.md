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
> **Candidate command; latest Hybrid live validation is `NOT RUN`.** The flags below enable Scheme 1 direct transport, Scheme 3 release-tail compression, and aerial speed `0.36`. The earlier Fast-Cycle v1 `4/4` result does not validate this combination. A matched bundle dry-run is necessary but not sufficient: complete the applicable lifecycle/hold checks and obtain on-site authorization before a supervised pilot. This is not an unattended launch recipe.

> [!CAUTION]
> Instantiating `DualTouchRobot` may energize motors on startup, and process teardown (`ArmController::cleanup()`) drops holding torque via `Startouch::disable_all()`. Ensure workspace is physically clear before running `--confirm-startup-home` / `--confirm-exit-home`, keep an operator hand on the hardware E-stop, and stay within authorized station bounds.

```bash
export STARTOUCH_SDK_PY_DIR=/path/to/startouch_sdk/interface_py
export STATION_RUNTIME_DIR=/path/to/station_runtime
export SMOLVLA_CHECKPOINT=/path/to/checkpoints/right7_pretrained_model
export ROBOT_PYTHON=/path/to/vendor_env/bin/python3

"$ROBOT_PYTHON" src/station_lite_smolvla_right7_guarded.py \
  --server http://<POLICY_HOST>:8011 \
  --expected-checkpoint "$SMOLVLA_CHECKPOINT" \
  --expected-seed-mode fixed \
  --task "Pick up the yellow wooden block from the white grid and place it on the brown target area." \
  --target-color yellow \
  --settled-chunks \
  --condition-final-grasp \
  --hold-grasp-pose \
  --place-after-grasp \
  --phase-speeds \
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
  --loops 8
```

Disable optional optimizations individually with `--no-direct-place-transport`, `--no-home-takeoff-hover`, `--no-fast-grid-approach`, `--no-trim-idle-tail`, or `--no-async-prefetch` to revert smoothly toward the legacy settled baseline.

