"""Deterministic task and live-command parser for Phase-1 sorting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .contracts import ObjectRecord

COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("红", "红色", "red"),
    "yellow": ("黄", "黄色", "yellow"),
    "blue": ("蓝", "蓝色", "blue"),
}

UNSUPPORTED_COLOR_TOKENS = ("绿", "绿色", "green", "青", "cyan", "紫", "purple")

FORBIDDEN_BATCH_PATTERNS = (
    (re.compile(r"双臂|左手|dual.?arm|left.?arm", re.I), "unsupported_dual_arm"),
    (re.compile(r"堆叠|拆分|拼装|拆卸|stack|lego|assemble", re.I), "unsupported_stacking_or_lego"),
    (re.compile(r"/reset|disable_all|go_home|subprocess|import\s+os|sudo\s", re.I), "forbidden_unsafe_control_token"),
)

NON_BROWN_DEST_PATTERN = re.compile(
    r"(放到|放进|移到|move to|place in|into).{0,10}(红框|绿区|蓝箱|白区|多箱|other box|green zone|blue bin)",
    re.I,
)


@dataclass(frozen=True)
class ParsedBatchRequest:
    valid: bool
    requested_colors: tuple[str, ...]
    expected_count: int | None
    destination_id: str
    max_attempts_per_object: int
    sort_mode: str
    clear_table: bool = False
    error_code: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class ParsedPriorityUpdate:
    valid: bool
    priority_color: str | None
    scope: str
    error_code: str | None = None
    reason: str = ""


def _extract_colors_in_order(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    lower = text.lower()
    for canonical, aliases in COLOR_ALIASES.items():
        best_idx: int | None = None
        for alias in aliases:
            idx = lower.find(alias)
            if idx != -1 and (best_idx is None or idx < best_idx):
                best_idx = idx
        if best_idx is not None:
            found.append((best_idx, canonical))
    found.sort(key=lambda item: item[0])
    return [color for _, color in found]


def parse_batch_instruction(text: str, *, scene_id: str | None = None) -> ParsedBatchRequest:
    """Parse initial batch instruction (fixed 1-3 targets or continuous clear-table loop)."""
    clean = text.strip()
    if not clean:
        return ParsedBatchRequest(
            valid=False,
            requested_colors=(),
            expected_count=None,
            destination_id="brown_target",
            max_attempts_per_object=2,
            sort_mode="heuristic_ease",
            error_code="empty_instruction",
            reason="Batch instruction cannot be empty.",
        )

    for pattern, code in FORBIDDEN_BATCH_PATTERNS:
        if pattern.search(clean):
            return ParsedBatchRequest(
                valid=False,
                requested_colors=(),
                expected_count=None,
                destination_id="brown_target",
                max_attempts_per_object=2,
                sort_mode="heuristic_ease",
                error_code=code,
                reason=f"Instruction violates Phase-1 scope constraint: {code}.",
            )

    if NON_BROWN_DEST_PATTERN.search(clean):
        return ParsedBatchRequest(
            valid=False,
            requested_colors=(),
            expected_count=None,
            destination_id="brown_target",
            max_attempts_per_object=2,
            sort_mode="heuristic_ease",
            error_code="unsupported_destination",
            reason="Phase-1 supports only the calibrated brown placement area.",
        )

    lower = clean.lower()
    for token in UNSUPPORTED_COLOR_TOKENS:
        if token in lower:
            return ParsedBatchRequest(
                valid=False,
                requested_colors=(),
                expected_count=None,
                destination_id="brown_target",
                max_attempts_per_object=2,
                sort_mode="heuristic_ease",
                error_code="unsupported_color_in_phase1",
                reason=f"Unsupported Phase-1 woodblock color token detected: {token}.",
            )

    is_clear_table = (scene_id == "scene_clear_table") or bool(
        re.search(
            r"clear\s*(?:the\s*)?table|until\s*(?:the\s*)?(?:white\s*)?(?:grid|table|workspace)\s*(?:is\s*)?empty|"
            r"清空桌面|清理桌面|循环清理|循环放置|直到桌面",
            clean,
            re.I,
        )
    )
    colors = _extract_colors_in_order(clean)
    if not colors and is_clear_table:
        colors = ["red", "yellow", "blue"]
    if not colors:
        return ParsedBatchRequest(
            valid=False,
            requested_colors=(),
            expected_count=None,
            destination_id="brown_target",
            max_attempts_per_object=2,
            sort_mode="heuristic_ease",
            error_code="clarification_needed_no_target_color",
            reason="Could not resolve target colors (expected red, yellow, blue).",
        )

    expected_count: int | None = None if is_clear_table else len(colors)
    if not is_clear_table:
        if re.search(r"三个|3\s*个|three", clean, re.I):
            expected_count = 3
        elif re.search(r"两个|2\s*个|two", clean, re.I):
            expected_count = 2
        elif re.search(r"一个|1\s*个|one", clean, re.I):
            expected_count = 1

        if expected_count != len(colors):
            return ParsedBatchRequest(
                valid=False,
                requested_colors=tuple(colors),
                expected_count=expected_count,
                destination_id="brown_target",
                max_attempts_per_object=2,
                sort_mode="heuristic_ease",
                error_code="clarification_needed_count_vs_colors",
                reason=(
                    f"Expected count {expected_count} does not match explicit color list {colors}; "
                    "requesting operator clarification instead of guessing."
                ),
            )

    has_brown = bool(re.search(r"棕色|brown", clean, re.I))
    if not has_brown:
        return ParsedBatchRequest(
            valid=False,
            requested_colors=tuple(colors),
            expected_count=expected_count,
            destination_id="brown_target",
            max_attempts_per_object=2,
            sort_mode="heuristic_ease",
            clear_table=is_clear_table,
            error_code="clarification_needed_destination",
            reason="Destination must explicitly reference the calibrated brown area.",
        )

    explicit_ease = bool(
        re.search(r"easiest\s*first|closest\s*first|优先抓容易|容易的优先|最容易", clean, re.I)
    )
    explicit_strict = bool(
        re.search(r"in\s+exact\s+order|in\s+order|strict\s+order|严格按顺序|严格顺序|按顺序", clean, re.I)
    )
    if explicit_strict:
        sort_mode = "strict_order"
    elif explicit_ease or is_clear_table:
        sort_mode = "heuristic_ease"
    elif scene_id == "scene_1_standard":
        sort_mode = "strict_order"
    else:
        sort_mode = "heuristic_ease"

    return ParsedBatchRequest(
        valid=True,
        requested_colors=tuple(colors),
        expected_count=expected_count,
        destination_id="brown_target",
        max_attempts_per_object=2,
        sort_mode=sort_mode,
        clear_table=is_clear_table,
        reason=(
            "Parsed continuous clear-table sorting loop until empty."
            if is_clear_table
            else f"Parsed validated Phase-1 sorting batch ({sort_mode})."
        ),
    )


def bind_scene_objects(
    parsed: ParsedBatchRequest,
    observed_objects: Sequence[ObjectRecord],
) -> tuple[list[ObjectRecord], str | None]:
    """Resolve requested targets deterministically against observed instances."""
    if not parsed.valid:
        return [], parsed.error_code or "invalid_batch"

    by_color: dict[str, list[ObjectRecord]] = {}
    for obj in observed_objects:
        by_color.setdefault(obj.color.lower(), []).append(obj)

    bound: list[ObjectRecord] = []
    for color in parsed.requested_colors:
        matches = by_color.get(color, [])
        if len(matches) > 1:
            ids = [item.object_id for item in matches]
            return [], f"ambiguous_multiple_candidates:{color}:{','.join(ids)}"
        if not matches:
            if parsed.clear_table:
                continue
            return [], f"missing_target_color:{color}"
        bound.append(matches[0])

    if not parsed.clear_table and parsed.expected_count is not None and len(bound) != parsed.expected_count:
        return [], f"count_mismatch:expected={parsed.expected_count},bound={len(bound)}"

    return bound, None


def parse_priority_command(
    text: str,
    *,
    scope: str = "remaining_objects",
    remaining_colors: set[str] | None = None,
) -> ParsedPriorityUpdate:
    """Parse an in-flight priority update command cleanly without arbitrary actions."""
    clean = text.strip()
    if not clean:
        return ParsedPriorityUpdate(
            valid=False,
            priority_color=None,
            scope=scope,
            error_code="empty_command",
            reason="Command text cannot be empty.",
        )

    if scope not in ("remaining_objects", "current_approach"):
        return ParsedPriorityUpdate(
            valid=False,
            priority_color=None,
            scope=scope,
            error_code="invalid_scope",
            reason=f"Unsupported update scope: {scope}.",
        )

    for pattern, code in FORBIDDEN_BATCH_PATTERNS:
        if pattern.search(clean):
            return ParsedPriorityUpdate(
                valid=False,
                priority_color=None,
                scope=scope,
                error_code=code,
                reason=f"Unsupported or unsafe control command text: {code}.",
            )

    if NON_BROWN_DEST_PATTERN.search(clean) or re.search(r"改放|换目标区|destination|drop zone", clean, re.I):
        return ParsedPriorityUpdate(
            valid=False,
            priority_color=None,
            scope=scope,
            error_code="unsupported_destination_change",
            reason="Changing placement destination mid-batch requires unverified paths and is blocked.",
        )

    colors = _extract_colors_in_order(clean)
    if len(colors) != 1:
        code = "clarification_needed_no_color" if not colors else "clarification_needed_multiple_colors"
        return ParsedPriorityUpdate(
            valid=False,
            priority_color=None,
            scope=scope,
            error_code=code,
            reason=f"Priority update must name exactly one target color, got {colors}.",
        )

    target_color = colors[0]
    if remaining_colors is not None and target_color not in remaining_colors:
        return ParsedPriorityUpdate(
            valid=False,
            priority_color=None,
            scope=scope,
            error_code="target_not_remaining",
            reason=f"Color '{target_color}' is not among remaining active batch objects {sorted(remaining_colors)}.",
        )

    return ParsedPriorityUpdate(
        valid=True,
        priority_color=target_color,
        scope=scope,
        reason=f"Prioritize remaining object with color '{target_color}'.",
    )


def format_single_target_task(obj: ObjectRecord) -> str:
    """Format an unambiguous single-target prompt matching right7 training style."""
    return (
        f"Pick up the {obj.color} wooden block ({obj.object_id}) from the white grid "
        f"and place it on the brown target area."
    )

