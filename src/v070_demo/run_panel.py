"""CLI launcher for the Stage D1 mock panel and Stage D3 supervised live panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Callable

from .candidate_bridge import D2_BUNDLE_SHA256
from .candidate_import import import_right7_guarded
from .contracts import SafetyBudgets
from .event_logger import DemoRunLogger
from .executor_bridge import MockRobotPublisher
from .live_robot_runner import LiveHardwareConfig, LiveRight7HardwareBridge
from .session_service import SessionService
from .web_panel import DemoPanelServer

_guarded = import_right7_guarded()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Start the LeRobot v0.7.0 closed-loop sorting panel.")
    ap.add_argument("--mode", choices=("mock", "supervised"), default="mock", help="Execution bridge mode")
    ap.add_argument("--host", default="127.0.0.1", help="Panel bind host (default loopback 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8090, help="Panel HTTP port (default 8090)")
    ap.add_argument("--allow-lan", action="store_true", help="Allow binding non-loopback panel host")
    ap.add_argument("--step-delay-s", type=float, default=0.35, help="Mock mode simulated per-phase duration (s)")
    ap.add_argument("--run-dir", type=Path, default=None, help="Optional directory to write run logs/metrics")

    # Stage D3 supervised hardware flags (ignored in --mode mock)
    ap.add_argument("--server", default="http://127.0.0.1:8011", help="Isolated right7 v0.7.0 /predict server URL")
    ap.add_argument("--img-host", default="127.0.0.1", help="TeleImager ZMQ host on robot control PC")
    ap.add_argument("--dry-run-report", type=Path, default=None, help="Passed g4_sdk_dryrun_gate JSON report path")
    ap.add_argument(
        "--expected-bundle-sha256",
        default=D2_BUNDLE_SHA256,
        help="Expected input_sha256 in SDK dry-run report (pass 'any' to match any SHA256 if D3 bundle)",
    )
    ap.add_argument("--expected-checkpoint", default=str(_guarded.EXPECTED_CHECKPOINT), help="Expected checkpoint path")
    ap.add_argument("--expected-seed-mode", default="fixed", help="Expected inference seed_mode")
    ap.add_argument("--admit-hardware", action="store_true", help="Explicit operator admission for supervised D3 hardware")
    ap.add_argument(
        "--confirm-startup-home",
        action="store_true",
        help="Confirm both arm workspaces are clear before DualTouchRobot() homes at first batch start",
    )
    ap.add_argument(
        "--confirm-exit-home",
        action="store_true",
        help="Confirm workspaces are clear to allow go_home() after safe normal completion",
    )
    ap.add_argument(
        "--no-return-home-on-batch-complete",
        action="store_true",
        help="Stay at elevated READY pose after batch completion instead of returning Home immediately",
    )
    ap.add_argument(
        "--unsafe-exit-action",
        choices=("confirm_before_disable", "immediate_disable"),
        default="confirm_before_disable",
        help="Action when exit go_home() is suppressed",
    )
    ap.add_argument("--min-safe-home-z-m", type=float, default=0.175, help="Minimum safe right-arm Z before exit Home")
    ap.add_argument("--hover-z-m", type=float, default=0.200, help="Hover height over grid cells (m)")
    ap.add_argument(
        "--use-clear-obs-retreat",
        action="store_true",
        help="Retract to CLEAR_OBS_POSE6 after placement/empty-check to eliminate cam_high forearm occlusion",
    )
    ap.add_argument("--snapshot-dir", type=Path, default=None, help="Optional directory for raw live observation captures")
    ap.add_argument("--lifecycle-audit-json", type=Path, default=None, help="Optional JSON path for teardown safety audit")
    ap.add_argument("--batch-deadline-s", type=float, default=180.0, help="Whole-batch deadline budget in seconds")
    return ap.parse_args(argv)


def _safe_normal_completion(service: SessionService, *, interrupted: bool) -> bool:
    if interrupted:
        return False
    snap = service.get_snapshot()
    objs = snap.get("objects") or []
    return bool(
        snap.get("session_state") in ("CLOSED", "READY")
        and snap.get("contact_held_state") == "EMPTY_CONFIRMED"
        and (len(objs) >= 1 or snap.get("scene_id") == "scene_clear_table")
        and all(o.get("status") == "PLACED_CONFIRMED" for o in objs)
    )


def main(
    argv: list[str] | None = None,
    *,
    robot_cls: Callable[[], Any] | None = None,
    image_client_cls: Callable[..., Any] | None = None,
    hold_confirm_fn: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    args = parse_args(argv)
    logger = DemoRunLogger(args.run_dir)
    budgets = SafetyBudgets(batch_deadline_s=float(args.batch_deadline_s))

    supervised_bridge: LiveRight7HardwareBridge | None = None
    if args.mode == "supervised":
        if (
            args.unsafe_exit_action == "confirm_before_disable"
            and hold_confirm_fn is None
            and not (sys.stdin is not None and sys.stdin.isatty())
        ):
            raise SystemExit(
                "[D3 LIFECYCLE] Refusing supervised live startup in non-TTY environment with "
                "--unsafe-exit-action confirm_before_disable. Run in an interactive TTY or pass "
                "--unsafe-exit-action immediate_disable."
            )
        exp_sha = None if str(args.expected_bundle_sha256).lower() == "any" else str(args.expected_bundle_sha256)
        cfg = LiveHardwareConfig(
            server_url=args.server,
            img_host=args.img_host,
            dry_run_report_path=args.dry_run_report,
            expected_bundle_sha256=exp_sha,
            expected_checkpoint=args.expected_checkpoint,
            expected_seed_mode=args.expected_seed_mode,
            operator_admitted=bool(args.admit_hardware),
            confirm_startup_home=bool(args.confirm_startup_home),
            confirm_exit_home=bool(args.confirm_exit_home),
            unsafe_exit_action=args.unsafe_exit_action,
            min_safe_home_z_m=float(args.min_safe_home_z_m),
            hover_z_m=float(args.hover_z_m),
            use_clear_obs_retreat=bool(args.use_clear_obs_retreat),
            return_home_on_batch_complete=bool(args.confirm_exit_home and not args.no_return_home_on_batch_complete),
            snapshot_dir=args.snapshot_dir,
        )
        supervised_bridge = LiveRight7HardwareBridge(
            config=cfg,
            robot_cls=robot_cls,
            image_client_cls=image_client_cls,
            hold_confirm_fn=hold_confirm_fn,
        )
        admitted = supervised_bridge.admit_hardware()
        bridge: Any = supervised_bridge
        print(
            f"[D3 ADMITTED] server={args.server} cases={admitted['dryrun']['cases_passed']}/{admitted['dryrun']['cases_total']} "
            f"sha256={admitted['dryrun']['input_sha256']} seed_base={admitted['health'].get('seed_base')} hardware_attached=False",
            flush=True,
        )
    else:
        bridge = MockRobotPublisher(step_delay_s=args.step_delay_s)

    service = SessionService(bridge=bridge, budgets=budgets, logger=logger)
    server = DemoPanelServer(service, host=args.host, port=args.port, allow_lan=args.allow_lan)
    server.start()
    tag = "D3 SUPERVISED PANEL" if args.mode == "supervised" else "D1 MOCK PANEL"
    print(
        f"[{tag}] Listening on http://{server.host}:{server.port} "
        f"(admission={bridge.admission_state.value}, hardware_attached={getattr(bridge, 'hardware_attached', False)})",
        flush=True,
    )

    interrupted = False
    try:
        while not service.wait_closed(timeout=0.35):
            pass
        # Allow any in-flight CLOSE_SESSION HTTP JSON response to finish flushing cleanly.
        time.sleep(0.10)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[{tag}] Interrupt received; holding/tearing down cleanly without blind go_home()...", flush=True)
    finally:
        server.stop()
        if supervised_bridge is not None:
            safe_exit = _safe_normal_completion(service, interrupted=interrupted)
            supervised_bridge.shutdown_hardware(
                safe_normal_completion=safe_exit,
                audit_json_path=args.lifecycle_audit_json,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
