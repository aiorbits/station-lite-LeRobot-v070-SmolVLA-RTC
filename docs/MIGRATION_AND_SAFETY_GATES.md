# LeRobot `v0.7.0` Migration, `bfloat16` Root Cause, & Hardware Safety Gates

Moving a physical robot stack from `lerobot==0.6.2` (`torch==2.11.0+cu130`, `transformers==5.5.4`) to an isolated `lerobot==0.7.0` (`torch==2.15.0+cu132`, `transformers==5.22.2`) environment requires proving that model weights, processors, HTTP transport, physical IK plans, and hardware exit teardown behave safely without silent regression.

---

## 1. Staged Verification Gate Architecture (`G0–G4`)

| Gate | Scope | Isolation Contract | Verified Result |
| --- | --- | --- | --- |
| **`G0` Replay Manifest** | Fixed `12`-window corpus spanning all `8` baseline `new9` runs and `3` colors (`yellow`, `red`, `cyan`) | Read-only hash verification (`sha256_file`, `sha256_array`); zero hardware imports | `12/12` windows valid (`7D` state, `50x7` action, `camera1/2` remapping) |
| **`G1` Offline Model Gate** (`src/v070_common.py`) | Cold load + CUDA warmup/benchmark + 5 negative tests (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) | CUDA-only; fails if offline cache revision `7b375e1b...` missing or CPU requested | **PASSED**: cold load `47.258 s`, `917.0 MB` peak VRAM, `0.0` fixed-noise repeat diff, `12/12` finite `50x7` |
| **`G2` Paired Numerical Parity** | `12 windows x 3 seeds x 3 repeats = 108 chunks/env` across `0.6.2` and `0.7.0` (`TF32=False`) | Identical pre-generated noise tensors and `uint8` image frames | `XYZ max = 0.7908 mm` (`<= 1.0`), `SO(3) max = 0.2530 deg` (`> 0.20`), `Gripper max = 4.2325 mm` (`> 0.5`) -> Triggered bisection |
| **`G3` HTTP Service Contract** (`src/g3_http_compatibility_gate.py`) | Port separation (`8011`, refuses `8010`), `400` malformed/NaN/aspect checks, stale `seq/session/seed` client rejection | Runs with `hardware_connected=false`; never touches robot ports | **PASSED**: all malformed inputs rejected `400`, all stale responses rejected, reconnect RTT `233.672 ms` |
| **`G4` Execution & Dry-Run Gate** (`src/g4_execution_adapter_gate.py`, `src/g4_sdk_dryrun_gate.py`) | Host adapter conditioning, fault-injection lifecycle tests, and CAN-isolated (`sudo unshare --net`) vendor SDK IK/400 Hz planner check | Blocks all CAN sockets inside network namespace during robot workstation dry-run | Host gates **PASSED**; `11/11` v1 fast-cycle robot dry-run **PASSED**; latest 13-case hybrid robot dry-run **`NOT RUN`** |

---

## 2. Root Cause of `G2` Numerical Drift: 3-Stage Sublayer Bisection

Because `SO(3)` (`0.2530°`) and `Gripper` (`4.2325 mm`) exceeded strict FP32 parity thresholds (`0.2°` / `0.5 mm`), we executed a 3-stage sublayer bisection comparing identical inputs and weights across `v0.6.2` and `v0.7.0`:

| Stage / Probe | Max Abs Diff | Rel L2 Diff | Finding |
| --- | ---: | ---: | --- |
| Checkpoint Weight SHA-256 (`patch_emb`, `q_proj`, `post_ln`, `connector`) | — | `0.0` (exact match) | Weights are bit-for-bit identical across environments |
| Preprocessed Pixel Input (`pv`) | `0.0` | `0.0` | Image preprocessing / resize / padding is identical |
| Vision Embeddings (`emb_def`, `bfloat16`) | `0.001953` | `1.78e-5` | Minimal initial drift |
| Vision Encoder Layer 0 (`l0_def`, `bfloat16`) | `0.015625` | `0.00108` | Accumulates across 12 SigLIP SDPA/MLP layers |
| Full Vision Tower Output (`vis_def`, default SDPA `bfloat16`) | `0.750` | **`0.02634`** | Significant `bfloat16` kernel divergence |
| Connector Projection Output (`conn_def`, `bfloat16`) | **`1.000`** | `0.01220` | Amplifies slightly differently through `torch 2.15 + cu132` vs `2.11 + cu130` |
| Vision Tower (`vis_eager_fp32`, eager `float32`) | **`0.04089`** | **`0.00101`** | **~26x reduction** in relative L2 error when cast to `fp32` |
| Connector Projection (`conn_eager_fp32`, eager `float32`) | **`0.03862`** | **`0.000466`** | **~26x reduction** in relative L2 error |

**Resolution (`Option 1` Native `bfloat16` Sign-Off)**:
- The divergence comes strictly from reduced-precision (`bfloat16` 7-bit mantissa) rounding differences between PyTorch/Transformers CUDA kernels across minor versions—not from weight corruption or processor bugs.
- Replaying all 12 `G0` windows through `station_lite_smolvla_right7_guarded.py` confirmed: `12/12` phase agreements, `12/12` deferral agreements, max conditioned XYZ diff `0.541 mm`, max rotation diff `0.148°`, max gripper diff `0.111 mm`, and zero host guard violations. Native `bfloat16` was signed off for `v0.7.0`.

---

## 3. Hardware Lifecycle Safety Hardening

Code & disassembly auditing of the third-party vendor C++ SDK revealed a critical mechanical risk: calling `ArmController::cleanup()`, `robot.disconnect()`, or exiting the Python process invokes `Startouch::disable_all()`, **immediately dropping holding torque across all joints**. On a gravity-loaded desktop arm carrying a wooden block or positioned near table contact, uncontrolled torque loss creates a drop/collision hazard.

We implemented multi-layer lifecycle protection in `src/station_lite_smolvla_right7_guarded.py` (verified via 14 fault-injection scenarios in `tests/test_g4_adapter_and_lifecycle.py`):

1. **Workspace Clear Gates (`--confirm-startup-home`, `--confirm-exit-home`)**: Explicit operator confirmation flags required before executing dual-arm `go_home()`.
2. **Post-Placement Vertical Retreat Before Homing**: When placement completes at low table height (`ee_z < 0.175 m`), the client first executes a vertical lift (`-> 0.200 m`) so joints clear table obstacles cleanly prior to homing.
3. **Fail-Closed `evaluate_exit_home_safety` Suppression**: Automatically blocks `go_home()` if any of the following hold:
   - Non-normal exit (`KeyboardInterrupt`, exception, guard rejection, `max_grasp_retries`),
   - Unreleased/held object detected (`holding_object=True`),
   - Unknown EE Z or near-table contact (`ee_z_m < min_safe_home_z_m`, default `0.175 m`),
   - `--exit-home-policy never` or missing `--confirm-exit-home`.
4. **Energized Hold Pause (`--unsafe-exit-action confirm_before_disable`)**: When safe exit homing is suppressed, the client skips destructive sleep (`skipped_wait_for_interrupt`), preserves `ArmController` torque, and pauses interactively for physical E-stop / mechanical support confirmation before teardown (or degrades cleanly with an explicit audit flag in non-TTY scripts).
5. **Network-Isolated CAN Dry-Run (`src/g4_sdk_dryrun_gate.py`)**: Executes vendor `solve_ik_euler` and `400 Hz` `plan_joint_waypoints_with_gripper` inside `sudo unshare --net` on the robot workstation—proving zero physical/soft joint violations and clean `cleanup()` return without touching hardware CAN buses.

> [!NOTE]
> While lifecycle fault injection (`14` unit scenarios) passed offline, a live physical hardware emergency-stop `safe_hold` pause test on an energized arm remains **`NOT RUN`**.

