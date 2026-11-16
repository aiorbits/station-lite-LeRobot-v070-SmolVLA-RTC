"""Single helper locating and importing v070_candidate modules safely."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any


def resolve_candidate_dir() -> Path:
    """Resolve v070_candidate directory across staging and SmolVLA_v070 layouts."""
    env_dir = os.environ.get("SMOLVLA_V070_CANDIDATE_DIR", "").strip()
    if env_dir:
        cand = Path(env_dir).expanduser().resolve()
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"SMOLVLA_V070_CANDIDATE_DIR not found: {cand}")

    pkg_root = Path(__file__).resolve().parent
    sibling_cand = pkg_root.parent / "v070_candidate"
    if sibling_cand.is_dir():
        return sibling_cand

    if (pkg_root.parent / "station_lite_smolvla_right7_guarded.py").is_file():
        return pkg_root.parent

    staged_cand = pkg_root.parent.parent.parent / "SmolVLA_v070" / "src" / "v070_candidate"
    if staged_cand.is_dir():
        return staged_cand

    raise FileNotFoundError(
        "Unable to locate v070_candidate; set SMOLVLA_V070_CANDIDATE_DIR or sync into SmolVLA_v070/src/v070_demo."
    )


def ensure_candidate_on_path() -> Path:
    cand_dir = resolve_candidate_dir()
    cand_str = str(cand_dir)
    if cand_str not in sys.path:
        sys.path.insert(0, cand_str)

    # Also ensure station_lite_smolvla_single_chunk (in baseline SmolVLA/ or shared src/) is discoverable.
    single_chunk_candidates = [
        os.environ.get("SMOLVLA_BASELINE_DIR", "").strip(),
        str(cand_dir),
        str(Path(__file__).resolve().parent.parent.parent),
        str(cand_dir.parent.parent.parent / "SmolVLA"),
        str(Path(__file__).resolve().parent.parent.parent / "exports" / "station-lite-LeRobot-v070-SmolVLA-RTC" / "src"),
    ]
    for raw_dir in single_chunk_candidates:
        if not raw_dir:
            continue
        p = Path(raw_dir)
        if (p / "station_lite_smolvla_single_chunk.py").is_file():
            p_str = str(p.resolve())
            if p_str not in sys.path:
                sys.path.append(p_str)
            break
    return cand_dir


def import_build_grasp_replay_cases() -> Any:
    ensure_candidate_on_path()
    return importlib.import_module("build_grasp_replay_cases")


def import_right7_guarded() -> Any:
    ensure_candidate_on_path()
    return importlib.import_module("station_lite_smolvla_right7_guarded")
