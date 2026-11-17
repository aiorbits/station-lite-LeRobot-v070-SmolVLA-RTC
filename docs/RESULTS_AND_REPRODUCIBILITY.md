# Results, Verification Boundaries, & Reproducibility

This document separates **robot-verified runs**, **offline host/CUDA evaluations**, and **architectural updates whose live hardware tests remain `NOT RUN`**.

---

## 1. Baseline Benchmark: `new9` (`LeRobot 0.6.2`)

- **Configuration**: `right7` (`10,000`-step checkpoint), top-grid hover guidance, calibrated 3D wrist projection & square OBB symmetry, discrete settled chunks (`0.20 s + 0.20 s` settle wait).
- **Live Hardware Result**: **`8/8` documented system pick-and-place trials succeeded**.
- **Archived Controller Dry-Run Audit**: `65` `new9` trajectories (`157` total across `19` snapshot runs) replayed through the `400 Hz` vendor spline planner:
  - Physical/soft joint violations: **`0`**
  - Median planner rate: **`400.0 Hz`**
  - Peak joint velocity / accel: **`0.470 rad/s` / `17.038 rad/s^2`**
  - Max aligned waypoint joint step / XYZ step / gripper step: **`0.0460 rad` / `14.94 mm` / `14.25 mm`**

---

## 2. Live Robot Fast-Cycle v1 (`LeRobot 0.7.0`)

Enabled **Async `/predict` Prefetch** (`AsyncPredictClient`, `tail_lead = 0.10 s`), **Halved Settle Wait** (`0.10 s + 0.10 s`), **Option A Home->Hover Takeoff**, **Far-Grid Shortcut**, and **Idle-Tail Trimming**:

| Test Cell / Scenario | Legacy Settled Duration | Fast-Cycle v1 Live Duration | Live Outcome |
| --- | ---: | ---: | --- |
| Grid Cell `R8` (near front row) | `47.7 s` | **`30.21 s`** (`-36.7%`) | Success (full grasp & place) |
| Grid Cell `R2` (far rear row) | `75.5 s` | **`31.63–32.20 s`** (`-57.4%` to `-58.1%`) | Success (full grasp & place) |
| Prefetch Acceptance Rate | n/a | **`19/20` (95%)** async prefetch hits; `1/20` safe fallback | Zero drift/age violations |
| CAN-Isolated SDK Dry-Run (`unshare --net`) | Passed | **`11/11`** Fast-Cycle v1 bundles passed on robot host | Zero CAN / physical limit faults |

*Total live Fast-Cycle v1 score: **`4/4` full pick-and-place robot trials succeeded**.*

---

## 3. Latest Hybrid Optimizations (`Scheme 1` + `Scheme 3` + `Aerial 1.5x`)

Implemented on top of Fast-Cycle v1 immediately prior to archiving this repository milestone:

1. **Scheme 1 (`condition_direct_transport_actions`)**: Post-grasp 2-stage collision-clearing geometric lift (`z >= 0.208 m`) + cosine lateral/wrist arc directly to calibrated `BROWN_TARGET_HOVER_POSE6`, bypassing Chunk 4 `/predict` with fail-closed fallback (`[DIRECT TRANSPORT FALLBACK]`).
2. **Scheme 3 (`trim_release_tail=True`)**: Compresses excess stationary dwell after brown-pad gripper release (`open_steps + 4`).
3. **Aerial Pre-Grasp `1.5x` (`--aerial-approach-speed 0.36`)**: Accelerates pure aerial chunks above `max(0.176, hover_z_m - 0.012)` while preserving slow precision alignment near tabletop objects.

### Live Pilot — 16 November 2026

The latest configuration was tested on the calibrated reference station with one yellow block at a time. The operator reported all four warmed-up trials successfully grasped and placed the block. Archived text logs and lifecycle JSON agree on six chunks, `normal_completion`, release confirmation, successful post-placement lift, and exit Home in every trial. Physical placement is supported by the operator outcome, not inferred solely from gripper opening.

| Run ID suffix (local time, UTC+08:00) | Cell | Strategy cycle (s) | Including lift + exit Home (s) | Pure commanded motion (s, handoff summary) | Direct transport | Prefetch |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `20261116_154826` | R2 | 27.90 | 34.68 | 22.396 | 21 points / 2.333 s | 4/4 |
| `20261116_154927` | R2 | 27.92 | 34.68 | 22.634 | 21 points / 2.333 s | 4/4 |
| `20261116_155030` | R2 | 26.77 | 33.51 | 21.583 | 21 points / 2.333 s | 4/4 |
| `20261116_155531` | R8 | 28.90 | 35.64 | 23.488 | 31 points / 3.444 s | 4/4 |

**Timing:** strategy cycle starts at completion of startup Home and ends when the six-chunk policy loop completes. It includes camera acquisition, inference/planning and inter-chunk overhead, and excludes the subsequent vertical lift and exit Home. The longer column uses the same start and ends at completion of exit Home; initial startup homing and final process teardown are excluded from both. Times are rounded from the recorded run summary/transcript; pure motion is not a wall-clock task duration.

Relative to the earlier v1 observations, R2 strategy time changed from 31.63–32.20 s to 26.77–27.92 s, and R8 from 30.21 s to 28.90 s. These are historical, unmatched comparisons with several controller changes; they do not isolate a causal LeRobot, asynchronous-prefetch or RTC benefit. The prior **20–24 s projection was not realized**: measured strategy time was 26.77–28.90 s. Four successes are a pilot observation, not a reliability estimate.

### Planning Identity and Local Evidence

The independently archived vendor report records `13/13` cases passed, `cases_failed=0`, `all_passed=true`, `cleanup_returned=true`, SDK `0.1.9`, `hardware_connected=false`, and only `lo` visible in the isolated namespace. Its input bundle SHA-256 is `282d5eec862f314cca9a5d7b108dcf71528f428695a983c658fa83ae1772806c`.

Recorded original candidate source identities are guarded client `cabdc343b44e8768afd05ff75f0d69599f2d269b7dee3d3b629312ec842bf646` and conditioning helper `3b08bdf8aeb0839ef2b0ef7983cf345bf182744075444bc98b2e92dabbb899b3`. Current original local files match these hashes. Public source is a de-privatized edition and has separate hashes in `EXPORT_MANIFEST.json`; the live-source identities are not claims that the public files have identical bytes.

The private archive now contains the vendor JSON, run logs/lifecycle audits and snapshot/trace directories; this is newer evidence than the handoff's earlier note that return transfer was pending. No raw logs, images, state/action arrays or SDK artifacts are published here. The concise [machine-readable summary](../manifests/live_hybrid_20261116_summary.json) records aggregate evidence and source hashes. No independent video scoring was performed in this publication update.

### Startup Failures and Observed Hold Behavior

The archived sequence also contains **three camera-configuration exceptions** (`154207`, `154225`, `154244`, each zero policy chunks), a **health-request timeout** (`154310`, no lifecycle audit in the inspected archive), and a **cold-start guard rejection** (`154544`) before the four warmed-up successes. Thus **4/4 refers only to the warmed-up sequence**, not all launch records that day.

During the cold-start attempt, inference took `1529.7 ms`; observation age reached `1693.4 ms`, exceeding the unchanged `800 ms` bound. The stale-observation guard rejected the candidate after one chunk. Exit Home was suppressed and the process entered its operator hold pause. The audit records `hold_pause_requested=true`, `hold_pause_kept_motors_enabled=true`, `home_called=false`, then an interrupted pause and motor disable during disconnect.

This observes the software hold-pause path on an energized arm. It **does not certify a vendor stop/hold API, a hardware emergency-stop response, loaded-object hold, or arbitrary interruption recovery**: `safe_hold_verified` remains `false`, and the SDK has no exposed hold/stop method in the inspected report. Do not change that flag based on the visible pause alone.

Two synthetic `/predict` warmups subsequently returned inference times `131.876 ms` and `115.782 ms` (round trips 157.0 ms and 127.7 ms), followed by the four successful runs. This supports a startup-readiness procedure, not bypassing freshness checks.

### Remaining Validation

- Matched same-session G5 A/B repetitions, rotated blocks and distractor scenes: **NOT RUN** for this Hybrid batch.
- T3 continuous mid-flight splicing and live Guided RTC: **NOT RUN**; the measured prefetch mechanism remains settled HTTP prefetch.
- Comprehensive hardware emergency-stop/loaded hold and failure recovery: **not established** by the single startup hold observation.

---

## 4. Multi-Object Closed-Loop Supervised Orchestrator (`src/v070_demo/`, Stage D0–D3, v1.0–v1.1)

In v1.0–v1.1, `src/v070_demo/` connects the single-owner supervised web panel (`run_panel.py`, `web_panel.py`, `session_service.py`) to live robot hardware (`live_robot_runner.py`, `trajectory_planner.py`, `evidence_verifier.py`, `perception_adapter.py`):

- **CAN-Isolated Dry-Run Admission**: D2 bundle (`10/10`, `input_sha256="bf0bdc76..."`) and D3 bundle (`7/7`, `input_sha256="a7aa4f3d..."`) both passed on the robot host with `all_passed=true`, `cleanup_returned=true`.
- **v1.0 Live Closed-Loop Trials (16 November 2026)**:
  - **`r9` (`batch_8f2c1664`)**: First autonomous **`3/3` (`108.218 s`, `0` interventions)** across Red (`R9`), Yellow (`R2`), and Blue (`R7`).
  - **`r12` (`batch_6f649714`)**: Autonomous **`3/3` (`110.644 s`, `0` interventions)** across Red (`R3`, `0.0264 m`), Blue (`R8`, `0.0351 m`), and edge-cell Yellow (`R1`, `0.0280 m`), followed by safe verified return to Home (`RETURNED_HOME_ON_COMPLETION`).
- **v1.1 Live Verification & Panel Upgrades (17 November 2026, `r13` & `r14`, [live_multiblock_20261117_summary.json](../manifests/live_multiblock_20261117_summary.json))**:
  - **`r14` `Scene 1` Strict Order (`batch_32f4b3aa`)**: First live confirmation of default **`Red -> Yellow -> Blue`** sequence (`obj_red_R4_1` -> `obj_yellow_R8_1` -> `obj_blue_R3_1`), **`3/3` in `118.744 s`, `0` interventions**, `RETURNED_HOME_ON_COMPLETION`.
  - **`r13` & `r14` In-Flight Priority Override (`batch_da8e4b5c` & `batch_a9f597a2`)**: Mid-batch clicks to prioritize `Blue` during `PLACING` (`HOLDING_QUEUED_NEXT`) switched cleanly into `Gen 2` after empty-hand checkpoint settle (`5.50 s` and `9.83 s` boundary latencies) and completed **`Red -> Blue -> Yellow` (`3/3` in `113.603 s` and `118.582 s`, `0` interventions)** with `RETURNED_HOME_ON_COMPLETION`.
  - **`r14` `Clear the table` (`batch_e88cd039`)**: Completed continuous table replenishment (`obj_yellow_R6_1` -> `TABLE_REENQUEUED` -> `obj_yellow_R6_1_c2` -> `obj_red_R1_1`), **`3/3` in `120.938 s`, `0` interventions**, `RETURNED_HOME_ON_COMPLETION`.
  - **`r13` Initial Batch Safe Halt (`batch_d0c27c84`)**: Placed Red (`R6`) and Yellow (`R2`), then safely paused (`2/3`, `99.979 s`) when final Blue (`R7`) triggered `placement_unconfirmed` without unsafe auto-Home, prompting the guarded **`Return Home (Empty Gripper)`** operator recovery control shipped in v1.1.
- **Diagnostic Fixes & v1.1 Additions**:
  1. *Brown-Pad Slot Separation & Target-Color Release Check (`r8`)*: Unified approach transport to calibrated slot 0, fanned release targets across a 26 mm pad-interior triangle, and enforced `require_target_color=True` in `verify_live_placement_outcome`.
  2. *Lift/Transport Preload Tightening & Slip Recovery (`r10`)*: Increased `grasp_duration_s` (`2.0 -> 2.5 s`) and active clamping preload (`0.008 -> 0.0125 m`, ~5.3% tighter over the 85 mm stroke) applied immediately inside `_run_lifting_check()`, plus `CLEAR_OBS_POSE6` open retreat recovery (`block_slipped_recovered_empty`) when wrist vision confirms an empty drop.
  3. *Aligned Hover Descent Fallback & Hand-Occlusion Retry (`r11`)*: Added `_plan_aligned_hover_descent` + wrist-OBB closure completion when an out-of-distribution non-red block stays strictly aligned (`|du|, |dv| <= 15 px`) at `z = 0.200 m` without policy `< 0.04` output, plus `2 x 0.6 s` release-check retries and `place_0` `fallback_target_obb` checks against transient operator hand skin HSV contour merges (`> 45,000 px`).
  4. *1-Click Priority UI, Pre-Chunk-3 Switch Boundary, 3×3 Grid Canvas, & Guarded `Return Home` (v1.1)*:
     - Replaced text-box priority inputs with 1-click **`Red` / `Yellow` / `Blue`** buttons and a live **Chunk-3 (`GRASPING`) Switch Window** pill (`PRE_GRASP_IMMEDIATE` lifts open gripper safely to hover and switches before `GRASPING`; `HOLDING_QUEUED_NEXT` queues cleanly until post-placement settle once Chunk 3+ begins). Note: `PREEMPTED_BEFORE_GRASP` is verified in offline unit test `test_26`, while live `r13`/`r14` priority clicks occurred post-grasp (`PLACING`).
     - Rewrote `renderCanvas()` with explicit **`White Workspace Grid (R1–R9)`** and **`Brown Target Zone (Slot 0–2)`** mapping, fixed-width table wrapping, and snapshot event deduplication (`seenEventSeqs`).
     - Added **`CommandType.RETURN_HOME`** (`Return Home (Empty Gripper)` button), verified in offline `test_27` to reject closed grippers (`< 45 mm`), lift low Z before `go_home()`, and reset `PAUSED -> READY`.
- **Remaining Stage D4–D5 & Live Preemption Scope**: Live mid-approach pre-Chunk-3 priority preemption trial, full Stage D4 mid-trajectory switching, Stage D5 stress sweeps, external VLM planning, and Lego sorting remain **`NOT RUN`**. Concise evidence fields are summarized in [live_multiblock_20261116_summary.json](../manifests/live_multiblock_20261116_summary.json) and [live_multiblock_20261117_summary.json](../manifests/live_multiblock_20261117_summary.json).

---

## 5. Reproducing the Offline Benchmarks

### 5.1 Guided RTC & NFE Perturbation Sweep (Requires CUDA + Local Checkpoints/Snapshots)

```bash
python3 src/rtc_analysis/compare_guided_rtc.py \
  --checkpoint /path/to/checkpoint \
  --snapshots /path/to/handoff_snapshot_dir \
  --output ./outputs/rtc_comparison \
  --device cuda

python3 src/rtc_analysis/probe_rtc_perturb_nfe.py \
  --checkpoint /path/to/checkpoint \
  --snapshots /path/to/handoff_snapshot_dir \
  --output ./outputs/rtc_perturb_nfe
```

### 5.2 Right-Arm Dataset Conversion & Training (Requires CUDA + Raw Episodes)

1. Convert approved dual-arm episodes (`station_lite_pose14_v1`, 35 train / 10 holdout split) to single-right-arm `right7` LeRobot dataset format:

```bash
python3 src/training/convert_right_only_dataset.py \
  --manifest /path/to/woodblock_manifest.csv \
  --raw /path/to/raw_episodes \
  --split train \
  --out ./data/woodblock_right7_v1/train \
  --repo-id local/station_lite_woodblock_right7_train35_v1
```

2. Train `SmolVLA` (CUDA required; checkpoint output must reside on POSIX `ext4`, not exFAT):

```bash
SMOLVLA_PROJECT_ROOT=. \
SMOLVLA_DATASET_ROOT=./data/woodblock_right7_v1/train \
SMOLVLA_BASE_SNAPSHOT=./models/smolvla_base \
LEROBOT_TRAIN_BIN=lerobot-train \
python3 src/training/train_right_only_woodblock.py \
  --steps 10000 --batch-size 8 --save-freq 2000 --output-dir ./outputs/smolvla_right7_10k
```

