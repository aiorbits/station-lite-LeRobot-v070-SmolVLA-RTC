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

### Current Verification Scope & Honest Boundaries

- **Verified**:
  - Offline host guard & optimization unit tests (`test_right7_guarded_contract.py`, `test_v070_optimizations.py`, `test_g3_contract.py`, `test_g4_adapter_and_lifecycle.py`, `test_joint_trace.py`).
  - `g4_execution_adapter_gate.py` `raw_actions` bundle key bug fix (discovered during robot dry-run prep) verified against synthetic bundle schema in `TestG4SdkDryRunGate`.
- **`NOT RUN` (Explicitly Deferred Pending Next Lab Session)**:
  - **13-Case CAN-Isolated Robot SDK Dry-Run (`unshare --net`)**: Needs re-execution on the robot workstation with the regenerated 13-case bundle.
  - **Live Robot Pick-and-Place under Scheme 1 + Scheme 3 + Aerial `1.5x`**: Projected analytically to save an additional `~8–11 s` per cycle (`~20–24 s` estimated total vs. `30.2–32.2 s` Fast-Cycle v1), **not yet measured on hardware**.
  - **Physical Hardware `safe_hold` E-Stop Interrupt Check**: Lifecycle logic verified across `14` mock fault-injection scenarios offline; energized physical hold pause **not yet exercised on robot**.
  - **T3 Continuous Mid-Flight Splicing & Closed-Loop Multi-Object Sorting**: Unit-tested in offline suites (`evaluate_async_splice_candidate`), **not yet run on robot**.

---

## 4. Reproducing the Offline Benchmarks

### 4.1 Guided RTC & NFE Perturbation Sweep (Requires CUDA + Local Checkpoints/Snapshots)

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

### 4.2 Right-Arm Dataset Conversion & Training (Requires CUDA + Raw Episodes)

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

