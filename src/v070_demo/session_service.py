"""Single-owner session & command service with checkpoint generation switching."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import Any, Callable

from .contracts import (
    TERMINAL_OBJECT_STATES,
    CapabilitiesMatrix,
    CommandRecord,
    CommandStatus,
    CommandType,
    ContactHeldState,
    ExecutionPhase,
    ObjectRecord,
    ObjectState,
    SafetyBudgets,
    SessionState,
)
from .event_logger import DemoRunLogger, utc_now_iso
from .executor_bridge import (
    ExecutionBridge,
    PhaseStepResult,
    evaluate_grasp_evidence,
    evaluate_placement_evidence,
)
from .selector import SelectionOutcome, select_next_object
from .task_parser import (
    parse_batch_instruction,
    parse_priority_command,
    bind_scene_objects,
)


OBJECT_EXEC_PHASES: tuple[ExecutionPhase, ...] = (
    ExecutionPhase.APPROACHING_EMPTY,
    ExecutionPhase.DESCENDING_CONTACT,
    ExecutionPhase.GRASPING,
    ExecutionPhase.LIFTING_CHECK,
    ExecutionPhase.TRANSPORTING,
    ExecutionPhase.PLACING,
    ExecutionPhase.RELEASING_CHECK,
)


class SessionService:
    """Authoritative backend owner for sessions, commands, and closed-loop scheduling."""

    def __init__(
        self,
        bridge: ExecutionBridge,
        *,
        budgets: SafetyBudgets | None = None,
        logger: DemoRunLogger | None = None,
        clock_fn: Callable[[], float] | None = None,
        watchdog_poll_s: float = 0.05,
    ) -> None:
        self._lock = threading.RLock()
        self._wake_event = threading.Event()
        self._clock = clock_fn or time.monotonic
        self.budgets = budgets or SafetyBudgets()
        self.budgets.validate()
        self.bridge = bridge
        self.logger = logger or DemoRunLogger(None)
        self._watchdog_poll_s = watchdog_poll_s

        self.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self.session_state = SessionState.PREPARING
        self.contact_state = ContactHeldState.EMPTY_CONFIRMED
        self.execution_phase = ExecutionPhase.IDLE
        self.capabilities = CapabilitiesMatrix(
            hardware_execution_verified=False,
            mid_approach_switch_verified=False,
        )

        # Single-owner lease state
        self._lease_epoch = 0
        self._owner_client_id: str | None = None
        self._lease_token_id: str = ""
        self._last_heartbeat_mono: float = 0.0
        self._lease_expired = False

        # Batch / task state
        self.scene_id = "scene_1_standard"
        self.batch_id: str | None = None
        self.active_generation = 0
        self.active_instruction = ""
        self.priority_color: str | None = None
        self._clear_table_mode: bool = False
        self._sort_mode: str = "heuristic_ease"
        self._allowed_batch_colors: tuple[str, ...] = ()
        self._cycle_seq: int = 1
        self.objects: list[ObjectRecord] = []
        self.bound_object_ids: set[str] = set()
        self.current_object_id: str | None = None
        self.last_completed_object_id: str | None = None
        self.latest_obs_id: str = ""
        self.latest_obs_ts_mono: float = 0.0
        self.latest_selection_reason: str = ""

        # Budgets & counters
        self.batch_start_mono: float | None = None
        self.batch_end_mono: float | None = None
        self.planning_cycles_without_progress = 0
        self.operator_interventions = 0
        self.intervention_reasons: list[str] = []
        self.empty_grasp_count = 0
        self.recovery_success_count = 0

        # Control flags & worker
        self._pause_requested = False
        self._cancel_requested = False
        self._close_requested = False
        self._batch_starting = False
        self._worker: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._closed_event = threading.Event()

        # Command & event logs
        self._commands_by_id: dict[str, CommandRecord] = {}
        self._command_order: list[str] = []
        self._pending_priority_cmd_id: str | None = None
        self._events: list[dict[str, Any]] = []
        self._event_seq = 0

    def _emit_event(self, stream: str, kind: str, detail: dict[str, Any]) -> None:
        with self._lock:
            self._event_seq += 1
            entry = {
                "seq": self._event_seq,
                "ts_utc": utc_now_iso(),
                "ts_mono": round(self._clock(), 6),
                "session_id": self.session_id,
                "batch_id": self.batch_id,
                "generation": self.active_generation,
                "kind": kind,
                **detail,
            }
            self._events.append(entry)
            self.logger.append_jsonl(stream, entry)

    def start_session(self, client_id: str, *, scene_id: str = "scene_1_standard") -> dict[str, Any]:
        """Initialize software session and claim lease without touching robot hardware."""
        with self._lock:
            if self.session_state == SessionState.CLOSED:
                raise RuntimeError("Closed session cannot be restarted; instantiate a fresh SessionService.")
            now = self._clock()
            if self._owner_client_id is None or self._owner_client_id == client_id:
                self._grant_lease(client_id, now)
            self.scene_id = scene_id
            if self.session_state == SessionState.PREPARING:
                self.session_state = SessionState.READY
                self.execution_phase = ExecutionPhase.IDLE
                self._start_watchdog_locked()
            self.logger.write_json(
                "run_manifest.json",
                {
                    "session_id": self.session_id,
                    "created_utc": utc_now_iso(),
                    "admission_state": self.bridge.admission_state.value,
                    "budgets": {
                        "max_attempts_per_object": self.budgets.max_attempts_per_object,
                        "max_planning_cycles_without_progress": self.budgets.max_planning_cycles_without_progress,
                        "batch_deadline_s": self.budgets.batch_deadline_s,
                        "max_observation_age_ms": self.budgets.max_observation_age_ms,
                        "lease_ttl_s": self.budgets.lease_ttl_s,
                    },
                    "capabilities": self.capabilities.to_dict(),
                },
            )
            self._emit_event(
                "session_events.jsonl",
                "SESSION_READY",
                {
                    "owner_client_id": self._owner_client_id,
                    "lease_epoch": self._lease_epoch,
                    "admission_state": self.bridge.admission_state.value,
                },
            )
            return self.get_snapshot(client_id=client_id)

    def _start_watchdog_locked(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"v070-watchdog-{self.session_id}",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self._watchdog_poll_s):
            self.check_lease_watchdog()

    def _grant_lease(self, client_id: str, now_mono: float) -> None:
        self._lease_epoch += 1
        self._owner_client_id = client_id
        self._lease_token_id = f"lease_{self._lease_epoch:04d}_{uuid.uuid4().hex[:6]}"
        self._last_heartbeat_mono = now_mono
        self._lease_expired = False

    def claim_control(self, client_id: str) -> dict[str, Any]:
        """Explicit ownership takeover or re-claim after disconnect."""
        with self._lock:
            now = self._clock()
            prev = self._owner_client_id
            self._grant_lease(client_id, now)
            self._emit_event(
                "session_events.jsonl",
                "CONTROL_CLAIMED",
                {
                    "previous_owner": prev,
                    "new_owner": client_id,
                    "lease_epoch": self._lease_epoch,
                    "lease_token_id": self._lease_token_id,
                },
            )
            return self.get_snapshot(client_id=client_id)

    def heartbeat(self, client_id: str, lease_epoch: int | None = None) -> tuple[bool, str]:
        with self._lock:
            if client_id != self._owner_client_id:
                return False, "not_active_owner"
            if lease_epoch is not None and lease_epoch != self._lease_epoch:
                return False, "stale_lease_epoch"
            if self._lease_expired:
                return False, "lease_already_expired_requires_reclaim"
            self._last_heartbeat_mono = self._clock()
            return True, "ok"

    def check_lease_watchdog(self) -> bool:
        """Expire missed heartbeats and trigger phase-appropriate hold."""
        with self._lock:
            if self._owner_client_id is None or self._lease_expired:
                return False
            if self.session_state in (SessionState.CLOSED, SessionState.PREPARING) or self._batch_starting:
                return False
            elapsed = self._clock() - self._last_heartbeat_mono
            if elapsed <= self.budgets.lease_ttl_s:
                return False

            self._lease_expired = True
            self._emit_event(
                "session_events.jsonl",
                "CONTROLLER_LEASE_EXPIRED",
                {
                    "expired_owner": self._owner_client_id,
                    "lease_epoch": self._lease_epoch,
                    "elapsed_s": round(elapsed, 3),
                },
            )
            if self.session_state in (SessionState.RUNNING, SessionState.SWITCH_PENDING):
                self._pause_requested = True
                self.session_state = SessionState.PAUSE_PENDING
                held = self.bridge.request_phase_hold(
                    current_phase=self.execution_phase,
                    reason="controller_disconnect_timeout",
                )
                self.session_state = SessionState.PAUSED if held else SessionState.FAULTED
                self._emit_event(
                    "session_events.jsonl",
                    "WATCHDOG_HOLD_RESULT",
                    {"hold_confirmed": held, "session_state": self.session_state.value},
                )
                self._wake_event.set()
            return True

    def _validate_owner_access(self, client_id: str, lease_epoch: int | None) -> str | None:
        if self._owner_client_id is None:
            return "no_active_session_owner"
        if client_id != self._owner_client_id:
            return f"read_only_client:owner_is_{self._owner_client_id}"
        if self._lease_expired:
            return "lease_expired_reclaim_required"
        if lease_epoch is not None and lease_epoch != self._lease_epoch:
            return f"stale_lease_epoch:{lease_epoch}!={self._lease_epoch}"
        return None

    def _record_command(self, cmd: CommandRecord, *, stream_kind: str) -> CommandRecord:
        self._commands_by_id[cmd.command_id] = cmd
        if cmd.command_id not in self._command_order:
            self._command_order.append(cmd.command_id)
        self._emit_event(
            "command_events.jsonl",
            stream_kind,
            {
                "command_id": cmd.command_id,
                "command_type": cmd.command_type.value,
                "status": cmd.status.value,
                "scope": cmd.scope,
                "expected_active_generation": cmd.expected_active_generation,
                "applied_generation": cmd.applied_generation,
                "reason": cmd.reason,
                "switch_boundary": cmd.switch_boundary,
            },
        )
        return copy.deepcopy(cmd)

    def submit_command(
        self,
        *,
        client_id: str,
        command_id: str,
        command_type: CommandType,
        raw_text: str = "",
        scope: str = "remaining_objects",
        scene_id: str | None = None,
        lease_epoch: int | None = None,
        expected_active_generation: int | None = None,
    ) -> CommandRecord:
        """Process session, batch, priority, pause/resume/cancel commands authoritatively."""
        with self._lock:
            # 1. Idempotency check (retried request returns previous receipt).
            if command_id in self._commands_by_id:
                existing = copy.deepcopy(self._commands_by_id[command_id])
                return existing

            now = self._clock()
            exp_gen = self.active_generation if expected_active_generation is None else expected_active_generation
            cmd = CommandRecord(
                session_id=self.session_id,
                batch_id=self.batch_id,
                command_id=command_id,
                owner_client_id=client_id,
                lease_token_id=self._lease_token_id,
                lease_epoch=lease_epoch if lease_epoch is not None else self._lease_epoch,
                expected_active_generation=exp_gen,
                command_type=command_type,
                scope=scope,
                raw_text=raw_text,
                parsed_result={},
                status=CommandStatus.RECEIVED,
                created_ts_mono=now,
                expires_at_mono=now + self.budgets.command_ttl_s,
            )

            cmd.status = CommandStatus.VALIDATING
            access_err = self._validate_owner_access(client_id, lease_epoch)
            if access_err is not None:
                cmd.status = CommandStatus.REJECTED
                cmd.reason = access_err
                return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

            # Refresh heartbeat timestamp when owner sends a valid command.
            self._last_heartbeat_mono = now

            if self.session_state in (SessionState.CLOSED, SessionState.STOPPING):
                cmd.status = CommandStatus.REJECTED
                cmd.reason = f"session_not_accepting_commands:{self.session_state.value}"
                return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

            if command_type == CommandType.START_BATCH:
                return self._handle_start_batch(cmd, scene_id=scene_id, now=now)
            if command_type == CommandType.UPDATE_PRIORITY:
                return self._handle_update_priority(cmd, expected_active_generation=expected_active_generation, now=now)
            if command_type == CommandType.PAUSE:
                return self._handle_pause(cmd, now=now)
            if command_type == CommandType.RESUME:
                return self._handle_resume(cmd, now=now)
            if command_type == CommandType.CANCEL_BATCH:
                return self._handle_cancel_batch(cmd, now=now)
            if command_type == CommandType.CLOSE_SESSION:
                return self._handle_close_session(cmd, now=now)

            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"unsupported_command_type:{command_type}"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

    def _handle_start_batch(self, cmd: CommandRecord, *, scene_id: str | None, now: float) -> CommandRecord:
        if self.session_state != SessionState.READY:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"batch_start_disallowed_in_state:{self.session_state.value}"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        target_scene = scene_id or self.scene_id
        parsed = parse_batch_instruction(cmd.raw_text, scene_id=target_scene)
        cmd.parsed_result = {
            "requested_colors": list(parsed.requested_colors),
            "expected_count": parsed.expected_count,
            "destination_id": parsed.destination_id,
            "sort_mode": parsed.sort_mode,
            "clear_table": parsed.clear_table,
            "error_code": parsed.error_code,
        }
        if not parsed.valid:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = parsed.reason
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        if self._batch_starting:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = "batch_already_starting"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        self._batch_starting = True
        self._lock.release()
        try:
            obs_id, observed = self.bridge.observe_scene(target_scene, now)
        finally:
            self._lock.acquire()
            self._batch_starting = False

        if self._close_requested or self.session_state in (SessionState.CLOSED, SessionState.FAULTED):
            cmd.status = CommandStatus.CANCELLED
            cmd.reason = "session_closed_during_batch_start"
            return self._record_command(cmd, stream_kind="COMMAND_CANCELLED")

        # Hardware startup homing/warmup can exceed lease_ttl_s; keep owner lease live once start completes.
        ready_now = self._clock()
        self._last_heartbeat_mono = ready_now
        self.latest_obs_id = obs_id
        self.latest_obs_ts_mono = max((o.observation_ts_mono for o in observed), default=ready_now)
        bound, bind_err = bind_scene_objects(parsed, observed)
        if bind_err is not None:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"clarification_needed:{bind_err}"
            self.operator_interventions += 1
            self.intervention_reasons.append(f"batch_clarification:{bind_err}")
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        # Initialize new batch atomically
        self._clear_table_mode = bool(parsed.clear_table)
        self._sort_mode = str(parsed.sort_mode)
        self._allowed_batch_colors = tuple(parsed.requested_colors)
        self._cycle_seq = 1
        begin_fn = getattr(self.bridge, "begin_new_batch", None)
        if callable(begin_fn):
            try:
                begin_fn(allow_slot_recycling=self._clear_table_mode)
            except TypeError:
                begin_fn()
        if self._clear_table_mode:
            sync_fn = getattr(self.bridge, "sync_active_grid_colors", None)
            if callable(sync_fn):
                sync_fn({obj.color.lower() for obj in bound})
        self.scene_id = target_scene
        self.batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        cmd.batch_id = self.batch_id
        self.active_generation = 1
        self.active_instruction = cmd.raw_text
        self.priority_color = None
        self.objects = bound
        self.bound_object_ids = {obj.object_id for obj in bound}
        self.current_object_id = None
        self.last_completed_object_id = None
        self.planning_cycles_without_progress = 0
        self.operator_interventions = 0
        self.intervention_reasons = []
        self.empty_grasp_count = 0
        self.recovery_success_count = 0
        self._pause_requested = False
        self._cancel_requested = False
        self._pending_priority_cmd_id = None
        self.batch_start_mono = ready_now
        self.batch_end_mono = None
        self.contact_state = ContactHeldState.EMPTY_CONFIRMED

        self.logger.write_json(
            "scene_manifest.json",
            {
                "batch_id": self.batch_id,
                "scene_id": self.scene_id,
                "destination_id": parsed.destination_id,
                "bound_objects": [obj.to_dict() for obj in self.objects],
            },
        )

        self.session_state = SessionState.RUNNING
        cmd.status = CommandStatus.APPLIED
        cmd.accepted_ts_mono = ready_now
        cmd.applied_ts_mono = ready_now
        cmd.applied_generation = self.active_generation
        cmd.reason = "Batch started."

        self._record_command(cmd, stream_kind="COMMAND_APPLIED")
        self._emit_event(
            "task_events.jsonl",
            "BATCH_STARTED",
            {"scene_id": self.scene_id, "object_ids": [obj.object_id for obj in self.objects]},
        )

        self._worker = threading.Thread(
            target=self._run_batch_loop,
            args=(self.batch_id,),
            name=f"v070-batch-{self.batch_id}",
            daemon=True,
        )
        self._worker.start()
        return copy.deepcopy(cmd)

    def _handle_update_priority(
        self,
        cmd: CommandRecord,
        *,
        expected_active_generation: int | None,
        now: float,
    ) -> CommandRecord:
        if self.session_state not in (SessionState.RUNNING, SessionState.PAUSED, SessionState.SWITCH_PENDING):
            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"update_requires_active_batch:{self.session_state.value}"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        if cmd.scope == "current_approach" and not self.capabilities.mid_approach_switch_verified:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = "mid_approach_switch_blocked_until_D4"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        if expected_active_generation is not None and expected_active_generation != self.active_generation:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"generation_conflict:expected={expected_active_generation},active={self.active_generation}"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        remaining_colors = {
            obj.color for obj in self.objects if obj.status not in TERMINAL_OBJECT_STATES
        }
        parsed = parse_priority_command(
            cmd.raw_text,
            scope=cmd.scope,
            remaining_colors=remaining_colors,
        )
        cmd.parsed_result = {
            "priority_color": parsed.priority_color,
            "scope": parsed.scope,
            "error_code": parsed.error_code,
        }
        if not parsed.valid:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = parsed.reason
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        # Mark any previous pending priority update as SUPERSEDED
        if self._pending_priority_cmd_id is not None:
            prev = self._commands_by_id.get(self._pending_priority_cmd_id)
            if prev is not None and prev.status == CommandStatus.ACCEPTED_PENDING:
                prev.status = CommandStatus.SUPERSEDED
                prev.reason = f"Superseded by newer valid update {cmd.command_id}"
                self._record_command(prev, stream_kind="COMMAND_SUPERSEDED")

        cmd.status = CommandStatus.ACCEPTED_PENDING
        cmd.accepted_ts_mono = now
        cmd.reason = (
            "Accepted; waiting for empty-hand settled checkpoint before applying "
            f"(current_phase={self.execution_phase.value})."
        )
        self._pending_priority_cmd_id = cmd.command_id
        self._wake_event.set()
        return self._record_command(cmd, stream_kind="COMMAND_ACCEPTED_PENDING")

    def _handle_pause(self, cmd: CommandRecord, *, now: float) -> CommandRecord:
        if self.session_state not in (SessionState.RUNNING, SessionState.SWITCH_PENDING):
            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"cannot_pause_from:{self.session_state.value}"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        self._pause_requested = True
        self.session_state = SessionState.PAUSE_PENDING
        cmd.status = CommandStatus.APPLYING
        cmd.accepted_ts_mono = now
        cmd.reason = "Pause requested; awaiting verified executor hold confirmation."
        self._record_command(cmd, stream_kind="COMMAND_APPLYING")

        held = self.bridge.request_phase_hold(
            current_phase=self.execution_phase,
            reason="operator_pause",
        )
        if not held:
            self.session_state = SessionState.FAULTED
            cmd.status = CommandStatus.FAILED
            cmd.reason = "Executor could not verify safe phase hold."
            self._emit_event("session_events.jsonl", "PAUSE_HOLD_FAILED", {"phase": self.execution_phase.value})
            return self._record_command(cmd, stream_kind="COMMAND_FAILED")

        self.session_state = SessionState.PAUSED
        self.execution_phase = ExecutionPhase.HOLDING
        cmd.status = CommandStatus.APPLIED
        cmd.applied_ts_mono = self._clock()
        cmd.applied_generation = self.active_generation
        cmd.reason = "Executor confirmed phase hold; session PAUSED."
        self._wake_event.set()
        return self._record_command(cmd, stream_kind="COMMAND_APPLIED")

    def _handle_resume(self, cmd: CommandRecord, *, now: float) -> CommandRecord:
        if self.session_state != SessionState.PAUSED:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"resume_requires_paused_state:{self.session_state.value}"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        if self.contact_state == ContactHeldState.HELD_UNCERTAIN:
            cmd.status = CommandStatus.REJECTED
            cmd.reason = "resume_blocked_uncertain_held_state"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        # Require fresh observation + re-validation instead of reusing pre-pause actions
        obs_id, _ = self.bridge.observe_scene(self.scene_id, now)
        self.latest_obs_id = obs_id
        self.latest_obs_ts_mono = now
        self.bridge.cancel_future_queue(generation=self.active_generation, reason="resume_revalidate")

        self._pause_requested = False
        self.session_state = SessionState.RUNNING
        self.execution_phase = ExecutionPhase.OBSERVING
        cmd.status = CommandStatus.APPLIED
        cmd.accepted_ts_mono = now
        cmd.applied_ts_mono = now
        cmd.applied_generation = self.active_generation
        cmd.reason = f"Resumed after fresh observation {obs_id} and contact check."
        self._wake_event.set()
        return self._record_command(cmd, stream_kind="COMMAND_APPLIED")

    def _handle_cancel_batch(self, cmd: CommandRecord, *, now: float) -> CommandRecord:
        if self.session_state not in (
            SessionState.RUNNING,
            SessionState.PAUSE_PENDING,
            SessionState.PAUSED,
            SessionState.SWITCH_PENDING,
        ):
            cmd.status = CommandStatus.REJECTED
            cmd.reason = f"no_active_batch_to_cancel:{self.session_state.value}"
            return self._record_command(cmd, stream_kind="COMMAND_REJECTED")

        self._cancel_requested = True
        self._cancel_pending_priority_cmds(reason="batch_cancelled")
        self.bridge.cancel_future_queue(generation=self.active_generation, reason="batch_cancelled")
        self.bridge.request_phase_hold(current_phase=self.execution_phase, reason="cancel_batch")

        cmd.status = CommandStatus.APPLIED
        cmd.accepted_ts_mono = now
        cmd.applied_ts_mono = now
        cmd.applied_generation = self.active_generation
        cmd.reason = "Batch cancellation requested; pending commands cancelled, physical state preserved."
        self._wake_event.set()
        return self._record_command(cmd, stream_kind="COMMAND_APPLIED")

    def _handle_close_session(self, cmd: CommandRecord, *, now: float) -> CommandRecord:
        self._close_requested = True
        self._cancel_requested = True
        self._cancel_pending_priority_cmds(reason="session_closing")
        self.bridge.cancel_future_queue(generation=self.active_generation, reason="close_session")
        held = self.bridge.request_phase_hold(current_phase=self.execution_phase, reason="close_session")
        self.session_state = SessionState.CLOSED if held else SessionState.FAULTED
        self._watchdog_stop.set()
        self._wake_event.set()
        self._closed_event.set()
        cmd.status = CommandStatus.APPLIED if held else CommandStatus.FAILED
        cmd.accepted_ts_mono = now
        cmd.applied_ts_mono = self._clock()
        cmd.reason = "Session closed cleanly without implicit Home." if held else "Session close hold failed."
        return self._record_command(cmd, stream_kind="COMMAND_APPLIED" if held else "COMMAND_FAILED")

    def _cancel_pending_priority_cmds(self, *, reason: str) -> None:
        if self._pending_priority_cmd_id is None:
            return
        pending = self._commands_by_id.get(self._pending_priority_cmd_id)
        if pending is not None and pending.status == CommandStatus.ACCEPTED_PENDING:
            pending.status = CommandStatus.CANCELLED
            pending.reason = reason
            self._record_command(pending, stream_kind="COMMAND_CANCELLED")
        self._pending_priority_cmd_id = None

    def _apply_pending_update_at_checkpoint(self, *, next_observation_id: str) -> None:
        """Atomically bump generation and apply latest valid priority update."""
        with self._lock:
            if self._pending_priority_cmd_id is None:
                return
            pending = self._commands_by_id.get(self._pending_priority_cmd_id)
            if pending is None or pending.status != CommandStatus.ACCEPTED_PENDING:
                self._pending_priority_cmd_id = None
                return

            now = self._clock()
            if pending.expires_at_mono is not None and now > pending.expires_at_mono:
                pending.status = CommandStatus.EXPIRED
                pending.reason = "Expired before reaching safe checkpoint."
                self._record_command(pending, stream_kind="COMMAND_EXPIRED")
                self._pending_priority_cmd_id = None
                return

            # Cannot switch unless arm is confirmed empty (Phase-1 checkpoint safety)
            if self.contact_state != ContactHeldState.EMPTY_CONFIRMED:
                return

            self.session_state = SessionState.SWITCH_PENDING
            pending.status = CommandStatus.APPLYING
            self._record_command(pending, stream_kind="COMMAND_APPLYING")

            old_gen = self.active_generation
            self.bridge.cancel_future_queue(generation=old_gen, reason="checkpoint_generation_switch")
            self.active_generation = old_gen + 1
            new_color = pending.parsed_result.get("priority_color")
            self.priority_color = str(new_color) if new_color else None
            self.active_instruction = pending.raw_text

            # Clear old selection/rejection state while PRESERVING terminal items and contact_state
            for obj in self.objects:
                if obj.status not in TERMINAL_OBJECT_STATES:
                    obj.selection_reason = ""
                    obj.rejection_reason = ""

            pending.status = CommandStatus.APPLIED
            pending.applied_ts_mono = now
            pending.applied_generation = self.active_generation
            pending.switch_boundary = {
                "from_generation": old_gen,
                "to_generation": self.active_generation,
                "last_old_object_id": self.last_completed_object_id,
                "checkpoint_phase": self.execution_phase.value,
                "reobserved_with_id": next_observation_id,
                "applied_latency_ms": round((now - pending.created_ts_mono) * 1000.0, 2),
            }
            pending.reason = f"Applied at empty-hand settled checkpoint into generation {self.active_generation}."
            self._pending_priority_cmd_id = None
            self.session_state = SessionState.RUNNING
            self._record_command(pending, stream_kind="COMMAND_APPLIED")

    def wait_closed(self, timeout: float | None = None) -> bool:
        """Block up to `timeout` seconds until CLOSE_SESSION completes."""
        return self._closed_event.wait(timeout=timeout)

    def _refresh_observations_locked(self, *, allow_reenqueue: bool = False) -> tuple[str, str | None]:
        now = self._clock()
        obs_id, observed = self.bridge.observe_scene(self.scene_id, now)
        self.latest_obs_id = obs_id
        self.latest_obs_ts_mono = max((o.observation_ts_mono for o in observed), default=self._clock())
        obs_by_id = {item.object_id: item for item in observed}
        by_color: dict[str, list[ObjectRecord]] = {}
        for item in observed:
            by_color.setdefault(item.color.lower(), []).append(item)

        # Update bound batch objects (in clear_table mode also track a pending block if nudged to another cell).
        for target in self.objects:
            if target.status in TERMINAL_OBJECT_STATES:
                continue
            fresh = obs_by_id.get(target.object_id)
            if fresh is None and allow_reenqueue and target.status == ObjectState.PENDING:
                color_matches = by_color.get(target.color.lower(), [])
                if len(color_matches) == 1:
                    fresh = color_matches[0]
            if fresh is None:
                target.confidence = 0.0
                target.rejection_reason = "missing_in_latest_observation"
            else:
                target.pose_estimate = fresh.pose_estimate
                target.confidence = fresh.confidence
                target.observation_id = fresh.observation_id
                target.observation_ts_mono = fresh.observation_ts_mono
                target.occlusion_score = fresh.occlusion_score
                target.clearance_m = fresh.clearance_m
                target.grasp_pose_valid = fresh.grasp_pose_valid
                target.path_checked = fresh.path_checked
                target.estimated_duration_s = fresh.estimated_duration_s
                target.grid_cell = fresh.grid_cell or target.grid_cell

        if not allow_reenqueue:
            return obs_id, None

        sync_fn = getattr(self.bridge, "sync_active_grid_colors", None)
        if callable(sync_fn):
            sync_fn({item.color.lower() for item in observed})

        for color in self._allowed_batch_colors:
            matches = by_color.get(color, [])
            if len(matches) > 1:
                ids = [m.object_id for m in matches]
                return obs_id, f"ambiguous_multiple_candidates:{color}:{','.join(ids)}"

        active_nonterminal_colors = {
            o.color.lower() for o in self.objects if o.status not in TERMINAL_OBJECT_STATES
        }
        blocked_colors = {
            o.color.lower()
            for o in self.objects
            if o.status in (ObjectState.FAILED, ObjectState.NEEDS_OPERATOR)
        }
        existing_ids = {o.object_id for o in self.objects}
        reenqueued: list[str] = []

        for color in self._allowed_batch_colors:
            matches = by_color.get(color, [])
            if len(matches) != 1 or color in active_nonterminal_colors or color in blocked_colors:
                continue
            fresh = copy.deepcopy(matches[0])
            if fresh.object_id in existing_ids:
                self._cycle_seq += 1
                fresh.object_id = f"{fresh.object_id}_c{self._cycle_seq}"
            fresh.status = ObjectState.PENDING
            fresh.attempt_count = 0
            fresh.selection_reason = ""
            fresh.rejection_reason = ""
            fresh.destination_slot = None
            fresh.grasp_evidence = {}
            fresh.placement_evidence = {}
            self.objects.append(fresh)
            self.bound_object_ids.add(fresh.object_id)
            existing_ids.add(fresh.object_id)
            active_nonterminal_colors.add(color)
            reenqueued.append(fresh.object_id)

        if reenqueued:
            self._emit_event(
                "task_events.jsonl",
                "TABLE_REENQUEUED",
                {"observation_id": obs_id, "object_ids": reenqueued},
            )
        return obs_id, None

    def _finalize_batch_locked(self, *, end_state: SessionState, reason: str) -> None:
        self.batch_end_mono = self._clock()
        self.current_object_id = None
        if self.session_state not in (SessionState.CLOSED, SessionState.FAULTED):
            self.session_state = end_state
        self.execution_phase = (
            ExecutionPhase.HOLDING if end_state == SessionState.PAUSED else ExecutionPhase.IDLE
        )
        metrics = self._compute_metrics_locked()
        self.logger.write_json("metrics.json", metrics)
        self.logger.write_json(
            "scene_final_manifest.json",
            {
                "batch_id": self.batch_id,
                "scene_id": self.scene_id,
                "end_state": end_state.value,
                "reason": reason,
                "bound_objects": [obj.to_dict() for obj in self.objects],
            },
        )
        self._emit_event(
            "task_events.jsonl",
            "BATCH_FINISHED",
            {"end_state": end_state.value, "reason": reason, "metrics": metrics},
        )

    def _compute_metrics_locked(self) -> dict[str, Any]:
        total = len(self.objects)
        completed = sum(1 for o in self.objects if o.status == ObjectState.PLACED_CONFIRMED)
        failed = sum(1 for o in self.objects if o.status == ObjectState.FAILED)
        needs_op = sum(1 for o in self.objects if o.status == ObjectState.NEEDS_OPERATOR)
        end_t = self.batch_end_mono if self.batch_end_mono is not None else self._clock()
        elapsed_s = round(end_t - self.batch_start_mono, 3) if self.batch_start_mono is not None else 0.0
        apply_latencies = [
            round((c.applied_ts_mono - c.created_ts_mono) * 1000.0, 2)
            for c in self._commands_by_id.values()
            if c.command_type == CommandType.UPDATE_PRIORITY
            and c.status == CommandStatus.APPLIED
            and c.applied_ts_mono is not None
        ]
        return {
            "batch_id": self.batch_id,
            "completed_count": completed,
            "total_count": total,
            "headline_score": f"{completed}/{total}",
            "batch_elapsed_s": elapsed_s,
            "operator_interventions": self.operator_interventions,
            "intervention_reasons": list(self.intervention_reasons),
            "failed_count": failed,
            "needs_operator_count": needs_op,
            "empty_grasp_count": self.empty_grasp_count,
            "recovery_success_count": self.recovery_success_count,
            "priority_apply_latencies_ms": apply_latencies,
            "clock_domain": "host_monotonic",
        }

    def _wait_if_paused_locked(self) -> bool:
        """Wait safely while PAUSED without releasing stale candidate plans."""
        while self._pause_requested or self.session_state in (SessionState.PAUSE_PENDING, SessionState.PAUSED):
            if self._cancel_requested or self._close_requested:
                return False
            self._lock.release()
            try:
                self._wake_event.wait(0.05)
                self._wake_event.clear()
            finally:
                self._lock.acquire()
        return not (self._cancel_requested or self._close_requested)

    def _run_batch_loop(self, expected_batch_id: str) -> None:
        while True:
            with self._lock:
                if self.batch_id != expected_batch_id or self._close_requested:
                    return
                if not self._wait_if_paused_locked():
                    end = SessionState.CLOSED if self._close_requested else SessionState.READY
                    if self.contact_state != ContactHeldState.EMPTY_CONFIRMED:
                        end = SessionState.PAUSED
                        self.operator_interventions += 1
                        self.intervention_reasons.append("cancelled_or_stopped_while_non_empty")
                    self._finalize_batch_locked(end_state=end, reason="cancelled_or_closed")
                    return

                now = self._clock()
                if self.batch_start_mono is not None and (now - self.batch_start_mono) >= self.budgets.batch_deadline_s:
                    self.operator_interventions += 1
                    self.intervention_reasons.append("batch_deadline_exhausted")
                    for obj in self.objects:
                        if obj.status not in TERMINAL_OBJECT_STATES:
                            obj.status = ObjectState.NEEDS_OPERATOR
                            obj.rejection_reason = "batch_deadline_exhausted"
                    self._finalize_batch_locked(end_state=SessionState.READY, reason="batch_deadline_exhausted")
                    return

                remaining = [o for o in self.objects if o.status not in TERMINAL_OBJECT_STATES]
                if not remaining and self._clear_table_mode:
                    self.execution_phase = ExecutionPhase.OBSERVING
                    _obs_chk, amb_err = self._refresh_observations_locked(allow_reenqueue=True)
                    if amb_err is not None:
                        self.operator_interventions += 1
                        self.intervention_reasons.append(amb_err)
                        self._finalize_batch_locked(end_state=SessionState.PAUSED, reason=amb_err)
                        return
                    remaining = [o for o in self.objects if o.status not in TERMINAL_OBJECT_STATES]

                if not remaining:
                    no_failures = not any(
                        o.status in (ObjectState.FAILED, ObjectState.NEEDS_OPERATOR) for o in self.objects
                    )
                    if (
                        self.contact_state == ContactHeldState.EMPTY_CONFIRMED
                        and no_failures
                        and (len(self.objects) > 0 or self._clear_table_mode)
                    ):
                        ret_fn = getattr(self.bridge, "return_home_on_completion", None)
                        if callable(ret_fn):
                            self.execution_phase = ExecutionPhase.IDLE
                            self._lock.release()
                            try:
                                home_ok, home_err_c, home_err_d = ret_fn()
                            finally:
                                self._lock.acquire()
                            if self.batch_id != expected_batch_id or self._close_requested:
                                return
                            if home_ok:
                                self._emit_event(
                                    "task_events.jsonl",
                                    "RETURNED_HOME_ON_COMPLETION",
                                    {"reason": "table_cleared" if self._clear_table_mode else "all_objects_processed"},
                                )
                            elif home_err_c is not None:
                                self.operator_interventions += 1
                                self.intervention_reasons.append(f"{home_err_c}:{home_err_d}")
                    self._finalize_batch_locked(
                        end_state=SessionState.READY,
                        reason=("table_cleared" if self._clear_table_mode else "all_objects_processed"),
                    )
                    return

                self.execution_phase = ExecutionPhase.OBSERVING
                obs_id, amb_err = self._refresh_observations_locked(allow_reenqueue=self._clear_table_mode)
                if amb_err is not None:
                    self.operator_interventions += 1
                    self.intervention_reasons.append(amb_err)
                    self._finalize_batch_locked(end_state=SessionState.PAUSED, reason=amb_err)
                    return
                self._apply_pending_update_at_checkpoint(next_observation_id=obs_id)

                now = self._clock()
                outcome: SelectionOutcome = select_next_object(
                    self.objects,
                    now_mono=now,
                    budgets=self.budgets,
                    priority_color=self.priority_color,
                    sort_mode=self._sort_mode,
                    ordered_colors=self._allowed_batch_colors,
                )
                self.latest_selection_reason = outcome.reason

                if outcome.selected is None:
                    self.planning_cycles_without_progress += 1
                    for obj in self.objects:
                        if obj.status not in TERMINAL_OBJECT_STATES:
                            obj.status = ObjectState.DEFERRED
                    self._emit_event(
                        "task_events.jsonl",
                        "NO_FEASIBLE_CANDIDATE_CYCLE",
                        {
                            "planning_cycles_without_progress": self.planning_cycles_without_progress,
                            "rejected": outcome.rejected_details,
                        },
                    )
                    if self.planning_cycles_without_progress >= self.budgets.max_planning_cycles_without_progress:
                        self.operator_interventions += 1
                        self.intervention_reasons.append("no_progress_after_consecutive_planning_cycles")
                        for obj in self.objects:
                            if obj.status not in TERMINAL_OBJECT_STATES:
                                obj.status = ObjectState.NEEDS_OPERATOR
                        self._finalize_batch_locked(
                            end_state=SessionState.READY,
                            reason="max_planning_cycles_without_progress",
                        )
                        return
                    continue

                selected = outcome.selected
                self.planning_cycles_without_progress = 0
                selected.status = ObjectState.EXECUTING
                selected.attempt_count += 1
                self.current_object_id = selected.object_id
                run_gen = self.active_generation
                attempt_num = selected.attempt_count
                self._emit_event(
                    "task_events.jsonl",
                    "OBJECT_SELECTED",
                    {
                        "object_id": selected.object_id,
                        "attempt_count": attempt_num,
                        "selection_reason": selected.selection_reason,
                    },
                )

            # Execute single-target phases outside the lock
            stop_batch = self._execute_single_object_cycle(selected, run_gen=run_gen, attempt_num=attempt_num)
            if stop_batch:
                return

    def _execute_single_object_cycle(
        self,
        selected: ObjectRecord,
        *,
        run_gen: int,
        attempt_num: int,
    ) -> bool:
        """Run approach -> grasp -> lift check -> transport -> place -> release check. Returns True if batch ended."""
        for phase in OBJECT_EXEC_PHASES:
            with self._lock:
                if not self._wait_if_paused_locked():
                    end = SessionState.CLOSED if self._close_requested else SessionState.READY
                    if self.contact_state != ContactHeldState.EMPTY_CONFIRMED:
                        end = SessionState.PAUSED
                        self.operator_interventions += 1
                        self.intervention_reasons.append("interrupted_mid_object_preserved_contact")
                    self._finalize_batch_locked(end_state=end, reason="interrupted_during_phase")
                    return True
                self.execution_phase = phase
                now = self._clock()
                obj_snapshot = copy.deepcopy(selected)

            res: PhaseStepResult = self.bridge.run_phase_step(
                obj=obj_snapshot,
                phase=phase,
                task_generation=run_gen,
                attempt_number=attempt_num,
                now_mono=now,
            )

            with self._lock:
                if res.task_generation != self.active_generation:
                    self._emit_event(
                        "task_events.jsonl",
                        "STALE_GENERATION_RESULT_DISCARDED",
                        {
                            "object_id": selected.object_id,
                            "result_generation": res.task_generation,
                            "active_generation": self.active_generation,
                            "phase": phase.value,
                        },
                    )
                    selected.status = ObjectState.NEEDS_OPERATOR
                    selected.rejection_reason = "stale_generation_result_discarded"
                    self.operator_interventions += 1
                    self.intervention_reasons.append("stale_generation_result_discarded")
                    self._finalize_batch_locked(end_state=SessionState.FAULTED, reason="stale_generation_result")
                    return True

                if not res.ok:
                    if res.error_code == "block_slipped_recovered_empty":
                        self.empty_grasp_count += 1
                        self.contact_state = ContactHeldState.EMPTY_CONFIRMED
                        selected.grasp_evidence = {
                            **selected.grasp_evidence,
                            "slip_recovery": res.error_detail or "retracted_to_clear_obs",
                            "gripper_width_m": res.gripper_width_m,
                        }
                        self._emit_event(
                            "task_events.jsonl",
                            "EMPTY_GRASP_CONFIRMED",
                            {
                                "object_id": selected.object_id,
                                "attempt_count": selected.attempt_count,
                                "phase": phase.value,
                                "evidence": selected.grasp_evidence,
                            },
                        )
                        if selected.attempt_count >= self.budgets.max_attempts_per_object:
                            selected.status = ObjectState.FAILED
                            selected.rejection_reason = "max_grasp_attempts_reached"
                            if not self._clear_table_mode:
                                self.operator_interventions += 1
                                self.intervention_reasons.append(f"failed_after_retry:{selected.object_id}")
                            self.execution_phase = ExecutionPhase.CHECKPOINT_EMPTY_SETTLED
                            return False
                        selected.status = ObjectState.PENDING
                        self.execution_phase = ExecutionPhase.CHECKPOINT_EMPTY_SETTLED
                        self._emit_event(
                            "task_events.jsonl",
                            "EMPTY_GRASP_RECOVERED_FOR_RETRY",
                            {
                                "object_id": selected.object_id,
                                "next_attempt": selected.attempt_count + 1,
                                "recovered_from_phase": phase.value,
                            },
                        )
                        return False

                    selected.status = ObjectState.NEEDS_OPERATOR
                    selected.rejection_reason = res.error_code or "execution_fault"
                    self.operator_interventions += 1
                    self.intervention_reasons.append(f"{res.error_code}:{selected.object_id}")
                    self.bridge.request_phase_hold(current_phase=phase, reason=res.error_code or "phase_error")
                    self._finalize_batch_locked(
                        end_state=SessionState.FAULTED if res.error_code == "inference_timeout" else SessionState.READY,
                        reason=res.error_detail or (res.error_code or "phase_error"),
                    )
                    return True

                if res.destination_slot is not None:
                    selected.destination_slot = res.destination_slot

                if phase == ExecutionPhase.LIFTING_CHECK:
                    verdict, contact_state, evidence = evaluate_grasp_evidence(
                        gripper_width_m=res.gripper_width_m,
                        visual_block_on_table=res.visual_block_on_table,
                    )
                    selected.grasp_evidence = evidence
                    self.contact_state = contact_state
                    if verdict == "HELD_UNCERTAIN":
                        selected.status = ObjectState.NEEDS_OPERATOR
                        selected.rejection_reason = "uncertain_held_state_evidence_conflict"
                        self.operator_interventions += 1
                        self.intervention_reasons.append(f"uncertain_held:{selected.object_id}")
                        self.bridge.request_phase_hold(current_phase=phase, reason="uncertain_held_state")
                        self._finalize_batch_locked(
                            end_state=SessionState.PAUSED,
                            reason="uncertain_held_state_stops_safely",
                        )
                        return True
                    if verdict == "EMPTY_GRASP_CONFIRMED":
                        self.empty_grasp_count += 1
                        selected.status = ObjectState.EMPTY_GRASP_CONFIRMED
                        return self._handle_empty_grasp_recovery_locked(selected, run_gen=run_gen)
                    selected.status = ObjectState.HELD_CONFIRMED
                    self._emit_event(
                        "task_events.jsonl",
                        "HELD_CONFIRMED",
                        {"object_id": selected.object_id, "evidence": evidence},
                    )

                if phase == ExecutionPhase.RELEASING_CHECK:
                    placed_ok, post_contact, evidence = evaluate_placement_evidence(
                        gripper_width_m=res.gripper_width_m,
                        visual_block_in_brown_zone=res.visual_block_in_brown_zone,
                        visual_block_at_gripper=res.visual_block_at_gripper,
                    )
                    selected.placement_evidence = evidence
                    self.contact_state = post_contact
                    if not placed_ok:
                        selected.status = ObjectState.NEEDS_OPERATOR
                        selected.rejection_reason = "placement_unconfirmed"
                        self.operator_interventions += 1
                        self.intervention_reasons.append(f"placement_unconfirmed:{selected.object_id}")
                        self.bridge.request_phase_hold(current_phase=phase, reason="placement_unconfirmed")
                        self._finalize_batch_locked(
                            end_state=SessionState.PAUSED,
                            reason="placement_unconfirmed",
                        )
                        return True
                    selected.status = ObjectState.PLACED_CONFIRMED
                    if attempt_num > 1:
                        self.recovery_success_count += 1
                    self.last_completed_object_id = selected.object_id
                    self.execution_phase = ExecutionPhase.CHECKPOINT_EMPTY_SETTLED
                    self._emit_event(
                        "task_events.jsonl",
                        "PLACED_CONFIRMED",
                        {
                            "object_id": selected.object_id,
                            "attempt_count": attempt_num,
                            "evidence": evidence,
                        },
                    )
        return False

    def _handle_empty_grasp_recovery_locked(self, selected: ObjectRecord, *, run_gen: int) -> bool:
        """Execute verified retract/re-observe on attempt 1; stop retries after attempt 2."""
        self._emit_event(
            "task_events.jsonl",
            "EMPTY_GRASP_CONFIRMED",
            {
                "object_id": selected.object_id,
                "attempt_count": selected.attempt_count,
                "evidence": selected.grasp_evidence,
            },
        )
        if selected.attempt_count >= self.budgets.max_attempts_per_object:
            selected.status = ObjectState.FAILED
            selected.rejection_reason = "max_grasp_attempts_reached"
            self.operator_interventions += 1
            self.intervention_reasons.append(f"failed_after_retry:{selected.object_id}")
            self.execution_phase = ExecutionPhase.CHECKPOINT_EMPTY_SETTLED
            return False

        selected.status = ObjectState.RECOVERING
        self.execution_phase = ExecutionPhase.RECOVERING
        obj_snapshot = copy.deepcopy(selected)
        now = self._clock()
        self._lock.release()
        try:
            rec_res = self.bridge.run_empty_grasp_recovery(
                obj=obj_snapshot,
                task_generation=run_gen,
                now_mono=now,
            )
        finally:
            self._lock.acquire()

        if not rec_res.ok:
            selected.status = ObjectState.NEEDS_OPERATOR
            selected.rejection_reason = "recovery_path_failed"
            self.operator_interventions += 1
            self.intervention_reasons.append(f"recovery_failed:{selected.object_id}")
            self._finalize_batch_locked(end_state=SessionState.FAULTED, reason="recovery_path_failed")
            return True

        selected.status = ObjectState.PENDING
        self.contact_state = ContactHeldState.EMPTY_CONFIRMED
        self.execution_phase = ExecutionPhase.CHECKPOINT_EMPTY_SETTLED
        self._emit_event(
            "task_events.jsonl",
            "EMPTY_GRASP_RECOVERED_FOR_RETRY",
            {"object_id": selected.object_id, "next_attempt": selected.attempt_count + 1},
        )
        return False

    def get_snapshot(self, *, client_id: str | None = None, since_event_seq: int = 0) -> dict[str, Any]:
        """Return authoritative UI snapshot and incremental events without side effects."""
        with self._lock:
            now = self._clock()
            obs_age_ms = (
                round((now - self.latest_obs_ts_mono) * 1000.0, 1)
                if self.latest_obs_ts_mono > 0.0
                else None
            )
            pending_cmd = (
                self._commands_by_id[self._pending_priority_cmd_id].to_dict()
                if self._pending_priority_cmd_id in self._commands_by_id
                else None
            )
            commands = [self._commands_by_id[cid].to_dict() for cid in self._command_order[-20:]]
            events = [e for e in self._events if e["seq"] > since_event_seq]
            lease_remaining_s = (
                max(0.0, round(self.budgets.lease_ttl_s - (now - self._last_heartbeat_mono), 2))
                if self._owner_client_id is not None and not self._lease_expired
                else 0.0
            )
            return {
                "session_id": self.session_id,
                "batch_id": self.batch_id,
                "scene_id": self.scene_id,
                "session_state": self.session_state.value,
                "hardware_admission": self.bridge.admission_state.value,
                "contact_held_state": self.contact_state.value,
                "execution_phase": self.execution_phase.value,
                "active_generation": self.active_generation,
                "active_instruction": self.active_instruction,
                "pending_instruction": pending_cmd,
                "priority_color": self.priority_color,
                "current_object_id": self.current_object_id,
                "latest_observation_id": self.latest_obs_id,
                "latest_observation_age_ms": obs_age_ms,
                "observation_fresh": obs_age_ms is not None and obs_age_ms <= self.budgets.max_observation_age_ms,
                "latest_selection_reason": self.latest_selection_reason,
                "publisher_count": self.bridge.publisher_count,
                "lease": {
                    "owner_client_id": self._owner_client_id,
                    "viewer_client_id": client_id,
                    "is_owner": client_id is not None and client_id == self._owner_client_id and not self._lease_expired,
                    "lease_epoch": self._lease_epoch,
                    "lease_token_id": self._lease_token_id,
                    "expired": self._lease_expired,
                    "remaining_ttl_s": lease_remaining_s,
                },
                "capabilities": self.capabilities.to_dict(),
                "objects": [obj.to_dict() for obj in self.objects],
                "commands": commands,
                "metrics": self._compute_metrics_locked(),
                "runtime_status": self.bridge.runtime_status() if hasattr(self.bridge, "runtime_status") else None,
                "latest_event_seq": self._event_seq,
                "events": events,
            }
