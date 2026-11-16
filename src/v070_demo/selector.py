"""Explainable heuristic selector and feasibility checks matching Section 5.2-5.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import ObjectRecord, ObjectState, SafetyBudgets


@dataclass(frozen=True)
class SelectionOutcome:
    selected: ObjectRecord | None
    ordered_feasible: list[ObjectRecord]
    rejected_details: dict[str, str]
    stale_observation_count: int
    reason: str


def check_object_feasibility(
    obj: ObjectRecord,
    *,
    now_mono: float,
    budgets: SafetyBudgets,
) -> tuple[bool, str]:
    """Validate identity, freshness, clearance, grasp pose, and guard checks."""
    if obj.status in (
        ObjectState.PLACED_CONFIRMED,
        ObjectState.FAILED,
        ObjectState.NEEDS_OPERATOR,
    ):
        return False, f"terminal_state:{obj.status.value}"

    if obj.attempt_count >= budgets.max_attempts_per_object:
        return False, f"attempt_budget_exhausted:{obj.attempt_count}/{budgets.max_attempts_per_object}"

    obs_age_ms = (now_mono - obj.observation_ts_mono) * 1000.0
    if obs_age_ms < 0.0 or obs_age_ms > budgets.max_observation_age_ms:
        return False, f"stale_observation:{obs_age_ms:.1f}ms>{budgets.max_observation_age_ms:.1f}ms"

    if obj.confidence < budgets.min_confidence:
        return False, f"low_confidence:{obj.confidence:.2f}<{budgets.min_confidence:.2f}"

    if obj.occlusion_score > budgets.max_occlusion_score:
        return False, f"high_occlusion:{obj.occlusion_score:.2f}>{budgets.max_occlusion_score:.2f}"

    if obj.clearance_m < budgets.min_clearance_m:
        return False, f"insufficient_clearance:{obj.clearance_m:.3f}m<{budgets.min_clearance_m:.3f}m"

    if not obj.grasp_pose_valid:
        return False, "invalid_grasp_pose"

    if not obj.path_checked:
        return False, "candidate_path_check_failed"

    return True, "feasible"


def score_object_ease(
    obj: ObjectRecord,
    *,
    priority_color: str | None,
) -> tuple[float, str]:
    """Compute explainable heuristic ease ranking (Section 5.2)."""
    priority_bonus = 1000.0 if (priority_color is not None and obj.color == priority_color) else 0.0
    score = (
        priority_bonus
        + (obj.confidence * 25.0)
        + ((1.0 - obj.occlusion_score) * 20.0)
        + (min(obj.clearance_m, 0.12) * 150.0)
        - (obj.estimated_duration_s * 0.4)
        - (obj.attempt_count * 3.0)
    )
    reason_parts = [
        f"conf={obj.confidence:.2f}",
        f"occ={obj.occlusion_score:.2f}",
        f"clear={obj.clearance_m * 1000:.0f}mm",
        f"est={obj.estimated_duration_s:.1f}s",
    ]
    if priority_bonus > 0.0:
        reason_parts.insert(0, f"priority_override={obj.color}")
    return score, "heuristic_ease(" + ", ".join(reason_parts) + ")"


def select_next_object(
    objects: Iterable[ObjectRecord],
    *,
    now_mono: float,
    budgets: SafetyBudgets,
    priority_color: str | None = None,
    sort_mode: str = "heuristic_ease",
    ordered_colors: tuple[str, ...] = (),
) -> SelectionOutcome:
    """Filter infeasible candidates first, then rank by strict order or heuristic ease rules."""
    feasible: list[tuple[tuple[float, float], ObjectRecord, str]] = []
    rejected: dict[str, str] = {}
    stale_count = 0
    color_rank = {c.lower(): idx for idx, c in enumerate(ordered_colors)}
    use_strict = sort_mode == "strict_order" and bool(color_rank)

    for obj in objects:
        ok, reason = check_object_feasibility(obj, now_mono=now_mono, budgets=budgets)
        if not ok:
            rejected[obj.object_id] = reason
            if reason.startswith("stale_observation:"):
                stale_count += 1
            obj.rejection_reason = reason
            continue

        obj.rejection_reason = ""
        ease_score, ease_detail = score_object_ease(obj, priority_color=priority_color)
        if use_strict:
            is_prio = priority_color is not None and obj.color.lower() == priority_color.lower()
            if is_prio:
                primary_key = 1000.0
                detail = f"strict_order(priority_override={obj.color}, ease={ease_score:.1f})"
            else:
                idx = color_rank.get(obj.color.lower(), len(color_rank))
                primary_key = float(len(color_rank) - idx)
                detail = f"strict_order(pos={idx + 1}/{len(color_rank)}, color={obj.color}, clear={obj.clearance_m * 1000:.0f}mm)"
            rank_key = (primary_key, ease_score)
        else:
            detail = ease_detail
            rank_key = (ease_score, 0.0)
        obj.selection_reason = detail
        feasible.append((rank_key, obj, detail))

    feasible.sort(key=lambda entry: (-entry[0][0], -entry[0][1], entry[1].object_id))
    ordered = [item[1] for item in feasible]
    if not ordered:
        return SelectionOutcome(
            selected=None,
            ordered_feasible=[],
            rejected_details=rejected,
            stale_observation_count=stale_count,
            reason="no_feasible_candidates",
        )

    winner = ordered[0]
    return SelectionOutcome(
        selected=winner,
        ordered_feasible=ordered,
        rejected_details=rejected,
        stale_observation_count=stale_count,
        reason=f"selected={winner.object_id}:{winner.selection_reason}",
    )

