"""Stage D0 frozen contracts, state enums, budgets, and records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SessionState(str, Enum):
    PREPARING = "PREPARING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSE_PENDING = "PAUSE_PENDING"
    PAUSED = "PAUSED"
    SWITCH_PENDING = "SWITCH_PENDING"
    STOPPING = "STOPPING"
    CLOSED = "CLOSED"
    FAULTED = "FAULTED"


class HardwareAdmissionState(str, Enum):
    SIMULATION_MOCK = "SIMULATION_MOCK"
    UNADMITTED = "UNADMITTED"
    ADMITTED_SUPERVISED = "ADMITTED_SUPERVISED"


class ContactHeldState(str, Enum):
    EMPTY_CONFIRMED = "EMPTY_CONFIRMED"
    HELD_CONFIRMED = "HELD_CONFIRMED"
    HELD_UNCERTAIN = "HELD_UNCERTAIN"
    UNKNOWN = "UNKNOWN"


class ObjectState(str, Enum):
    PENDING = "PENDING"
    ASSESSING = "ASSESSING"
    EXECUTING = "EXECUTING"
    HELD_CONFIRMED = "HELD_CONFIRMED"
    PLACED_CONFIRMED = "PLACED_CONFIRMED"
    EMPTY_GRASP_CONFIRMED = "EMPTY_GRASP_CONFIRMED"
    RECOVERING = "RECOVERING"
    DEFERRED = "DEFERRED"
    FAILED = "FAILED"
    NEEDS_OPERATOR = "NEEDS_OPERATOR"


TERMINAL_OBJECT_STATES = frozenset({
    ObjectState.PLACED_CONFIRMED,
    ObjectState.FAILED,
    ObjectState.NEEDS_OPERATOR,
})


class CommandStatus(str, Enum):
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    ACCEPTED_PENDING = "ACCEPTED_PENDING"
    APPLYING = "APPLYING"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


TERMINAL_COMMAND_STATUSES = frozenset({
    CommandStatus.APPLIED,
    CommandStatus.REJECTED,
    CommandStatus.SUPERSEDED,
    CommandStatus.CANCELLED,
    CommandStatus.EXPIRED,
    CommandStatus.FAILED,
})


class CommandType(str, Enum):
    START_BATCH = "START_BATCH"
    UPDATE_PRIORITY = "UPDATE_PRIORITY"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL_BATCH = "CANCEL_BATCH"
    RETURN_HOME = "RETURN_HOME"
    CLOSE_SESSION = "CLOSE_SESSION"


class ExecutionPhase(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    APPROACHING_EMPTY = "APPROACHING_EMPTY"
    DESCENDING_CONTACT = "DESCENDING_CONTACT"
    GRASPING = "GRASPING"
    LIFTING_CHECK = "LIFTING_CHECK"
    TRANSPORTING = "TRANSPORTING"
    PLACING = "PLACING"
    RELEASING_CHECK = "RELEASING_CHECK"
    RECOVERING = "RECOVERING"
    HOLDING = "HOLDING"
    CHECKPOINT_EMPTY_SETTLED = "CHECKPOINT_EMPTY_SETTLED"


# Phases where Phase-1 command switching MUST wait (cannot break holding/contact).
HOLD_OR_CONTACT_PHASES = frozenset({
    ExecutionPhase.DESCENDING_CONTACT,
    ExecutionPhase.GRASPING,
    ExecutionPhase.LIFTING_CHECK,
    ExecutionPhase.TRANSPORTING,
    ExecutionPhase.PLACING,
    ExecutionPhase.RELEASING_CHECK,
    ExecutionPhase.RECOVERING,
})


@dataclass(frozen=True)
class SafetyBudgets:
    """Frozen D0 budget and timing boundaries."""

    max_attempts_per_object: int = 2
    max_planning_cycles_without_progress: int = 2
    batch_deadline_s: float = 180.0
    max_observation_age_ms: float = 800.0
    heartbeat_interval_s: float = 1.0
    lease_ttl_s: float = 3.0
    command_ttl_s: float = 60.0
    min_confidence: float = 0.70
    max_occlusion_score: float = 0.45
    min_clearance_m: float = 0.030

    def validate(self) -> None:
        if self.max_attempts_per_object < 1 or self.max_attempts_per_object > 2:
            raise ValueError("Phase-1 max_attempts_per_object must be in [1, 2]")
        if self.max_planning_cycles_without_progress < 1:
            raise ValueError("max_planning_cycles_without_progress must be >= 1")
        if self.batch_deadline_s <= 0.0:
            raise ValueError("batch_deadline_s must be finite positive (required before D3)")
        if self.max_observation_age_ms <= 0.0 or self.max_observation_age_ms > 800.0:
            raise ValueError("max_observation_age_ms must preserve <= 800 ms bound")
        if self.lease_ttl_s <= self.heartbeat_interval_s:
            raise ValueError("lease_ttl_s must exceed heartbeat_interval_s")


@dataclass
class ObjectRecord:
    """Tracked instance record matching Section 5.1."""

    object_id: str
    category: str
    color: str
    pose_estimate: dict[str, float]
    confidence: float
    observation_id: str
    observation_ts_mono: float
    destination_id: str = "brown_target"
    status: ObjectState = ObjectState.PENDING
    attempt_count: int = 0
    occlusion_score: float = 0.0
    clearance_m: float = 0.080
    grasp_pose_valid: bool = True
    path_checked: bool = True
    estimated_duration_s: float = 22.0
    grid_cell: str | None = None
    destination_slot: int | None = None
    selection_reason: str = ""
    rejection_reason: str = ""
    grasp_evidence: dict[str, Any] = field(default_factory=dict)
    placement_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class CommandRecord:
    """Audited command envelope and receipt status matching Section 8.2."""

    session_id: str
    batch_id: str | None
    command_id: str
    owner_client_id: str
    lease_token_id: str
    lease_epoch: int
    expected_active_generation: int
    command_type: CommandType
    scope: str
    raw_text: str
    parsed_result: dict[str, Any]
    status: CommandStatus
    created_ts_mono: float
    expires_at_mono: float | None = None
    accepted_ts_mono: float | None = None
    applied_ts_mono: float | None = None
    applied_generation: int | None = None
    switch_boundary: dict[str, Any] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command_type"] = self.command_type.value
        data["status"] = self.status.value
        return data


@dataclass
class CapabilitiesMatrix:
    """Explicit backend verification flags controlling UI button activation."""

    pause_verified: bool = True
    resume_verified: bool = True
    cancel_batch_verified: bool = True
    update_remaining_priority_verified: bool = True
    mid_approach_switch_verified: bool = False  # Gated behind D4
    external_vlm_verified: bool = False         # Optional T2 after baseline
    rtc_live_verified: bool = False             # Optional T3 after parity
    return_home_verified: bool = False          # No implicit /reset or Home in Phase 1
    hardware_execution_verified: bool = False   # Mock-only in D1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

