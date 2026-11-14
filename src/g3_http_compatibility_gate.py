#!/usr/bin/env python3
"""G3 gate: isolated HTTP compatibility service test suite on 127.0.0.1:8011 without hardware."""
from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
CANDIDATE_SERVER_DEFAULT = THIS_DIR / "station_lite_smolvla_right7_server.py"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from v070_common import (  # noqa: E402
    load_replay_manifest,
    sha256_file,
    summarize_scalar_errors,
    verify_checkpoint_manifest,
    verify_env_identity,
)

SCHEMA = "station_lite_smolvla_right7_v1"
SOURCE_SCHEMA = "station_lite_smolvla_right7_request_v1"
RESERVED_PRODUCTION_PORT = 8010


def validate_right7_health(
    health: dict[str, Any],
    expected_checkpoint: str,
    expected_seed_mode: str | None = None,
    expected_lerobot: str | None = "0.7.0",
) -> None:
    """Validate /health payload using guarded client contract plus candidate provenance."""
    if (
        health.get("schema") != SCHEMA
        or health.get("checkpoint") != expected_checkpoint
        or health.get("chunk_size") != 50
        or health.get("state_dim") != 7
        or health.get("action_dim") != 7
        or health.get("source_schema") != SOURCE_SCHEMA
        or not isinstance(health.get("task"), str)
        or not health.get("task")
        or health.get("device") != "cuda"
        or health.get("hardware_connected") is not False
    ):
        raise ValueError(f"Unexpected right7 health contract: {health}")
    if expected_seed_mode is not None and health.get("seed_mode") != expected_seed_mode:
        raise ValueError(f"Seed mode mismatch: actual={health.get('seed_mode')}, expected={expected_seed_mode}")
    runtime = health.get("candidate_runtime")
    if expected_lerobot is not None:
        if not isinstance(runtime, dict) or runtime.get("lerobot_version") != expected_lerobot:
            raise ValueError(f"Candidate runtime lerobot mismatch: {runtime}")


def validate_right7_predict_response(
    result: dict[str, Any],
    *,
    expected_session_id: str,
    expected_seq: int,
    expected_seed_mode: str | None = None,
    expected_seed_base: int | None = None,
) -> np.ndarray:
    """Validate /predict JSON response and reject stale, mismatched, or non-finite chunks."""
    if (
        result.get("schema") != SCHEMA
        or result.get("source_schema") != SOURCE_SCHEMA
        or result.get("session_id") != expected_session_id
        or result.get("seq") != expected_seq
        or result.get("action_dim") != 7
        or result.get("hardware_connected") is not False
    ):
        raise ValueError(
            f"Response identity/contract mismatch: got session={result.get('session_id')} "
            f"seq={result.get('seq')} hardware={result.get('hardware_connected')}"
        )
    if expected_seed_mode == "fixed" and expected_seed_base is not None:
        if result.get("sampling_seed") != expected_seed_base:
            raise ValueError(
                f"Fixed-seed mismatch: got={result.get('sampling_seed')}, expected={expected_seed_base}"
            )
    elif expected_seed_mode == "sequence" and expected_seed_base is not None:
        if result.get("sampling_seed") != expected_seed_base + expected_seq:
            raise ValueError(
                f"Sequence-seed mismatch: got={result.get('sampling_seed')}, "
                f"expected={expected_seed_base + expected_seq}"
            )

    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.shape != (50, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid right7 actions array: shape={actions.shape}, finite={np.isfinite(actions).all()}")
    return actions


def build_http_payload_for_window(
    smolvla_root: Path,
    window: dict[str, Any],
    *,
    session_id: str,
    seq: int,
    task_override: str | None = None,
) -> dict[str, Any]:
    """Construct exact right7 HTTP request payload from archived G0 window assets."""
    state_data = json.loads((smolvla_root / window["state_rel_path"]).read_text(encoding="utf-8"))
    images_b64: dict[str, str] = {}
    for cam_key, cam_meta in window["images"].items():
        img_bytes = (smolvla_root / cam_meta["rel_path"]).read_bytes()
        images_b64[cam_key] = base64.b64encode(img_bytes).decode("ascii")
    return {
        "schema": SOURCE_SCHEMA,
        "task": task_override if task_override is not None else str(window["task"]),
        "session_id": session_id,
        "seq": seq,
        "state": state_data["state"],
        "images_jpeg_base64": images_b64,
    }


def _encode_synthetic_jpeg_b64(size_wh: tuple[int, int], color: tuple[int, int, int] = (80, 120, 160)) -> str:
    img = Image.new("RGB", size_wh, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _http_request_raw(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            data = json.loads(resp.read().decode("utf-8"))
            return int(resp.status), resp_headers, data
    except urllib.error.HTTPError as err:
        resp_headers = {k.lower(): v for k, v in err.headers.items()} if err.headers else {}
        raw = err.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw}
        return int(err.code), resp_headers, data


def _http_post_json(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], float]:
    raw = json.dumps(payload, allow_nan=False).encode("utf-8")
    t0 = time.perf_counter()
    status, _headers, data = _http_request_raw(
        base_url + "/predict",
        method="POST",
        body=raw,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    return status, data, rtt_ms


def _check_port_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(f"Refusing to start G3: {host}:{port} is already bound")


def _isolated_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TQDM_DISABLE"] = "1"
    return env


def _drain_stream_to_file(stream: Any, log_path: Path) -> threading.Thread:
    """Drain subprocess stdout/stderr pipe cleanly onto disk without holding raw exFAT unbuffered fds."""
    def _worker() -> None:
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            for line in stream:
                handle.write(line)
                handle.flush()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def _start_candidate_server(
    server_script: Path,
    checkpoint: Path,
    host: str,
    port: int,
    seed: int,
    seed_mode: str,
    log_path: Path,
    startup_timeout_s: float = 180.0,
) -> tuple[subprocess.Popen[str], threading.Thread, dict[str, Any], float]:
    _check_port_free(host, port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        str(server_script),
        "--checkpoint",
        str(checkpoint),
        "--host",
        host,
        "--port",
        str(port),
        "--seed",
        str(seed),
        "--seed-mode",
        seed_mode,
        "--offline",
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_isolated_subprocess_env(),
    )
    assert proc.stdout is not None
    drain_thread = _drain_stream_to_file(proc.stdout, log_path)

    base_url = f"http://{host}:{port}"
    deadline = time.monotonic() + startup_timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            drain_thread.join(timeout=2.0)
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"G3 server exited early rc={proc.returncode}:\n{tail}")
        try:
            status, headers, data = _http_request_raw(base_url + "/health", timeout=2.0)
            if status == 200:
                if "no-store" not in headers.get("cache-control", ""):
                    raise RuntimeError(f"Expected Cache-Control: no-store on /health, got {headers}")
                ready_s = time.perf_counter() - t0
                return proc, drain_thread, data, ready_s
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    drain_thread.join(timeout=2.0)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
    raise RuntimeError(f"Timed out waiting for G3 server readiness ({last_err}):\n{tail}")


def _stop_server(proc: subprocess.Popen[str], drain_thread: threading.Thread, host: str, port: int) -> int:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    drain_thread.join(timeout=2.0)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return int(proc.returncode or 0)
        time.sleep(0.1)
    raise RuntimeError(f"Port {host}:{port} stayed open after terminating G3 server")


def _run_sequential_replay_pass(
    base_url: str,
    smolvla_root: Path,
    windows: list[dict[str, Any]],
    *,
    session_id: str,
    seed_mode: str,
    seed_base: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Replay all G0 windows sequentially over HTTP and persist exact requests & raw outputs."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    exact_requests: list[dict[str, Any]] = []
    exact_responses: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    action_stack: list[np.ndarray] = []
    rtts: list[float] = []
    infers: list[float] = []

    for seq, win in enumerate(windows):
        payload = build_http_payload_for_window(smolvla_root, win, session_id=session_id, seq=seq)
        status, resp_json, rtt_ms = _http_post_json(base_url, payload, timeout=30.0)
        if status != 200:
            raise RuntimeError(f"Sequential replay {win['window_id']} failed HTTP {status}: {resp_json}")
        actions = validate_right7_predict_response(
            resp_json,
            expected_session_id=session_id,
            expected_seq=seq,
            expected_seed_mode=seed_mode,
            expected_seed_base=seed_base,
        )
        infer_ms = float(resp_json["inference_ms"])
        rtts.append(rtt_ms)
        infers.append(infer_ms)
        action_stack.append(actions)
        exact_requests.append(payload)
        exact_responses.append(resp_json)
        per_window.append({
            "window_id": win["window_id"],
            "phase_category": win["phase_category"],
            "seq": seq,
            "task": win["task"],
            "sampling_seed": int(resp_json["sampling_seed"]),
            "rtt_ms": round(rtt_ms, 3),
            "inference_ms": round(infer_ms, 3),
            "http_overhead_ms": round(rtt_ms - infer_ms, 3),
            "action_shape": list(actions.shape),
            "all_finite": True,
        })

    (artifact_dir / "exact_requests.json").write_text(json.dumps(exact_requests, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "exact_raw_responses.json").write_text(json.dumps(exact_responses, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        artifact_dir / "replay_arrays.npz",
        actions=np.stack(action_stack, axis=0),
        rtt_ms=np.asarray(rtts, dtype=np.float64),
        inference_ms=np.asarray(infers, dtype=np.float64),
        window_ids=np.asarray([w["window_id"] for w in windows]),
    )
    overheads = [r - i for r, i in zip(rtts, infers)]
    return {
        "session_id": session_id,
        "seed_mode": seed_mode,
        "window_count": len(windows),
        "exact_requests_path": str(artifact_dir / "exact_requests.json"),
        "exact_raw_responses_path": str(artifact_dir / "exact_raw_responses.json"),
        "replay_arrays_npz": str(artifact_dir / "replay_arrays.npz"),
        "rtt_ms_summary": summarize_scalar_errors(np.asarray(rtts)),
        "inference_ms_summary": summarize_scalar_errors(np.asarray(infers)),
        "http_overhead_ms_summary": summarize_scalar_errors(np.asarray(overheads)),
        "per_window": per_window,
    }


def _check_input_and_seed_sensitivities(
    base_url: str,
    smolvla_root: Path,
    w0: dict[str, Any],
    w_alt: dict[str, Any],
    *,
    seed_mode: str,
    seed_base: int,
) -> dict[str, Any]:
    """Verify fixed-seed repeatability, camera field sensitivity, task sensitivity, and 4:3 resize."""
    base_payload = build_http_payload_for_window(smolvla_root, w0, session_id="g3-sens", seq=0)
    s0, r0, _ = _http_post_json(base_url, base_payload)
    assert s0 == 200
    a0 = validate_right7_predict_response(
        r0, expected_session_id="g3-sens", expected_seq=0,
        expected_seed_mode=seed_mode, expected_seed_base=seed_base,
    )

    # Repeat exact seq=0 -> must be bit/float identical
    s0_rep, r0_rep, _ = _http_post_json(base_url, base_payload)
    assert s0_rep == 200
    a0_rep = validate_right7_predict_response(
        r0_rep, expected_session_id="g3-sens", expected_seq=0,
        expected_seed_mode=seed_mode, expected_seed_base=seed_base,
    )
    repeat_seq0_max_diff = float(np.max(np.abs(a0 - a0_rep)))

    # Same observation with seq=1 -> identical under fixed mode, different noise sample under sequence mode
    seq1_payload = dict(base_payload, seq=1)
    s1, r1, _ = _http_post_json(base_url, seq1_payload)
    assert s1 == 200
    a1 = validate_right7_predict_response(
        r1, expected_session_id="g3-sens", expected_seq=1,
        expected_seed_mode=seed_mode, expected_seed_base=seed_base,
    )
    seq1_same_obs_max_diff = float(np.max(np.abs(a0 - a1)))

    # Swap cam_high only (using same seq=0)
    alt_payload = build_http_payload_for_window(smolvla_root, w_alt, session_id="g3-sens", seq=0)
    p_hi = dict(base_payload, images_jpeg_base64={
        "observation.images.cam_high": alt_payload["images_jpeg_base64"]["observation.images.cam_high"],
        "observation.images.cam_right_wrist": base_payload["images_jpeg_base64"]["observation.images.cam_right_wrist"],
    })
    _, r_hi, _ = _http_post_json(base_url, p_hi)
    a_hi = validate_right7_predict_response(
        r_hi, expected_session_id="g3-sens", expected_seq=0,
        expected_seed_mode=seed_mode, expected_seed_base=seed_base,
    )
    cam_high_sensitivity = float(np.max(np.abs(a0 - a_hi)))

    # Swap cam_right_wrist only (using same seq=0)
    p_wr = dict(base_payload, images_jpeg_base64={
        "observation.images.cam_high": base_payload["images_jpeg_base64"]["observation.images.cam_high"],
        "observation.images.cam_right_wrist": alt_payload["images_jpeg_base64"]["observation.images.cam_right_wrist"],
    })
    _, r_wr, _ = _http_post_json(base_url, p_wr)
    a_wr = validate_right7_predict_response(
        r_wr, expected_session_id="g3-sens", expected_seq=0,
        expected_seed_mode=seed_mode, expected_seed_base=seed_base,
    )
    cam_wrist_sensitivity = float(np.max(np.abs(a0 - a_wr)))

    # Change task prompt only
    p_task = dict(base_payload, task="Pick up the blue wooden block from the white grid and place it on the brown target area.")
    _, r_task, _ = _http_post_json(base_url, p_task)
    a_task = validate_right7_predict_response(
        r_task, expected_session_id="g3-sens", expected_seq=0,
        expected_seed_mode=seed_mode, expected_seed_base=seed_base,
    )
    task_sensitivity = float(np.max(np.abs(a0 - a_task)))

    # Valid non-640x480 4:3 image (320x240) must pass BILINEAR resize
    p_320 = dict(base_payload, images_jpeg_base64={
        "observation.images.cam_high": _encode_synthetic_jpeg_b64((320, 240), (180, 180, 180)),
        "observation.images.cam_right_wrist": _encode_synthetic_jpeg_b64((320, 240), (90, 90, 90)),
    })
    s_320, r_320, _ = _http_post_json(base_url, p_320)
    assert s_320 == 200
    _ = validate_right7_predict_response(
        r_320, expected_session_id="g3-sens", expected_seq=0,
        expected_seed_mode=seed_mode, expected_seed_base=seed_base,
    )

    if seed_mode == "fixed":
        seed_behavior_ok = (repeat_seq0_max_diff == 0.0 and seq1_same_obs_max_diff == 0.0)
    else:
        seed_behavior_ok = (repeat_seq0_max_diff == 0.0 and seq1_same_obs_max_diff > 1e-5)

    passed = bool(
        seed_behavior_ok
        and cam_high_sensitivity > 1e-4
        and cam_wrist_sensitivity > 1e-4
        and task_sensitivity > 1e-6
        and s_320 == 200
    )
    return {
        "seed_mode": seed_mode,
        "repeat_seq0_max_abs_diff": repeat_seq0_max_diff,
        "seq1_same_obs_max_abs_diff": seq1_same_obs_max_diff,
        "seed_behavior_ok": seed_behavior_ok,
        "cam_high_changed_max_abs_diff": cam_high_sensitivity,
        "cam_right_wrist_changed_max_abs_diff": cam_wrist_sensitivity,
        "task_changed_max_abs_diff": task_sensitivity,
        "resize_320x240_4_3_status": s_320,
        "passed": passed,
    }


def _check_validation_and_malformed_requests(base_url: str, valid_payload: dict[str, Any]) -> dict[str, Any]:
    """Run negative validation matrix across routes, schema, task, seq, state, cameras, and malformed JSON."""
    results: list[dict[str, Any]] = []

    def check_case(name: str, expected_status: int, method: str, path: str, raw_body: bytes | None, hdrs: dict[str, str]):
        st, _, body = _http_request_raw(base_url + path, method=method, body=raw_body, headers=hdrs, timeout=10.0)
        ok = st == expected_status
        results.append({"case": name, "expected_status": expected_status, "actual_status": st, "passed": ok, "error": body.get("error")})
        if not ok:
            raise RuntimeError(f"Validation case {name} returned {st} != {expected_status}: {body}")

    def json_bytes(obj: Any) -> bytes:
        return json.dumps(obj).encode("utf-8")

    check_case("unknown_get_route", 404, "GET", "/unknown", None, {})
    check_case("unknown_post_route", 404, "POST", "/unknown", b"{}", {"Content-Type": "application/json"})
    check_case("wrong_schema", 400, "POST", "/predict", json_bytes({**valid_payload, "schema": "wrong_v0"}), {"Content-Type": "application/json"})
    check_case("empty_task", 400, "POST", "/predict", json_bytes({**valid_payload, "task": "   "}), {"Content-Type": "application/json"})
    check_case("non_string_task", 400, "POST", "/predict", json_bytes({**valid_payload, "task": 123}), {"Content-Type": "application/json"})
    check_case("negative_seq", 400, "POST", "/predict", json_bytes({**valid_payload, "seq": -1}), {"Content-Type": "application/json"})
    check_case("bool_seq", 400, "POST", "/predict", json_bytes({**valid_payload, "seq": True}), {"Content-Type": "application/json"})
    check_case("wrong_state_dim_6", 400, "POST", "/predict", json_bytes({**valid_payload, "state": valid_payload["state"][:6]}), {"Content-Type": "application/json"})
    check_case("dict_state", 400, "POST", "/predict", json_bytes({**valid_payload, "state": {"bad": 1}}), {"Content-Type": "application/json"})
    check_case("bool_in_state", 400, "POST", "/predict", json_bytes({**valid_payload, "state": [True] * 7}), {"Content-Type": "application/json"})
    check_case("json_nan_in_state", 400, "POST", "/predict", json_bytes({**valid_payload, "state": [float("nan")] + [0.0] * 6}), {"Content-Type": "application/json"})

    cams = valid_payload["images_jpeg_base64"]
    check_case("missing_cam_high", 400, "POST", "/predict", json_bytes({**valid_payload, "images_jpeg_base64": {"observation.images.cam_right_wrist": cams["observation.images.cam_right_wrist"]}}), {"Content-Type": "application/json"})
    check_case("missing_cam_right_wrist", 400, "POST", "/predict", json_bytes({**valid_payload, "images_jpeg_base64": {"observation.images.cam_high": cams["observation.images.cam_high"]}}), {"Content-Type": "application/json"})
    check_case("wrong_aspect_ratio_100x100", 400, "POST", "/predict", json_bytes({**valid_payload, "images_jpeg_base64": {**cams, "observation.images.cam_high": _encode_synthetic_jpeg_b64((100, 100))}}), {"Content-Type": "application/json"})
    check_case("invalid_base64", 400, "POST", "/predict", json_bytes({**valid_payload, "images_jpeg_base64": {**cams, "observation.images.cam_high": "!!!bad_b64!!!"}}), {"Content-Type": "application/json"})
    check_case("valid_b64_non_image", 400, "POST", "/predict", json_bytes({**valid_payload, "images_jpeg_base64": {**cams, "observation.images.cam_right_wrist": base64.b64encode(b"not a jpeg").decode("ascii")}}), {"Content-Type": "application/json"})
    check_case("malformed_json_syntax", 400, "POST", "/predict", b"{bad_json", {"Content-Type": "application/json"})
    check_case("invalid_utf8_bytes", 400, "POST", "/predict", b"\xff\xfe\xfd", {"Content-Type": "application/json"})
    check_case("json_array_not_object", 400, "POST", "/predict", b"[1, 2, 3]", {"Content-Type": "application/json"})
    check_case("zero_content_length", 400, "POST", "/predict", b"", {"Content-Type": "application/json", "Content-Length": "0"})

    return {"passed": all(r["passed"] for r in results), "cases": results}


def _check_timeout_reconnect_and_stale(
    host: str,
    port: int,
    valid_payload: dict[str, Any],
    *,
    seed_mode: str,
    seed_base: int,
) -> dict[str, Any]:
    """Verify client read timeout during GPU inference, raw disconnect recovery, reconnects, and stale guards."""
    base_url = f"http://{host}:{port}"

    # 1. Force client read timeout in-flight (15 ms << ~150 ms inference)
    timeout_triggered = False
    timeout_exc_type = None
    p_timeout = dict(valid_payload, session_id="g3-timeout-abort", seq=0)
    try:
        _http_post_json(base_url, p_timeout, timeout=0.015)
    except Exception as exc:
        timeout_triggered = True
        timeout_exc_type = type(exc).__name__

    # Give server thread time to finish CUDA forward pass and handle disconnected client socket
    time.sleep(0.45)

    # 2. Reconnect cleanly immediately with a brand-new session_id
    p_reconn = dict(valid_payload, session_id="g3-reconnect-ok", seq=0)
    st_reconn, r_reconn, rtt_reconn = _http_post_json(base_url, p_reconn, timeout=10.0)
    assert st_reconn == 200
    _ = validate_right7_predict_response(
        r_reconn,
        expected_session_id="g3-reconnect-ok",
        expected_seq=0,
        expected_seed_mode=seed_mode,
        expected_seed_base=seed_base,
    )

    # 3. Truncated body on raw TCP socket -> server must return 400 without hanging
    chunks: list[bytes] = []
    with socket.create_connection((host, port), timeout=5.0) as sock:
        req_hdr = (
            "POST /predict HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 100\r\n"
            "Connection: close\r\n\r\n"
            '{"partial": 1'
        ).encode("utf-8")
        sock.sendall(req_hdr)
        sock.shutdown(socket.SHUT_WR)
        while True:
            part = sock.recv(4096)
            if not part:
                break
            chunks.append(part)
    raw_resp = b"".join(chunks).decode("utf-8", errors="replace")
    truncated_handled = ("400 Bad Request" in raw_resp) and ("incomplete request body" in raw_resp)

    # 4. Early client TCP drop after sending full request, followed by sequential HTTPConnection requests
    full_body = json.dumps(dict(valid_payload, session_id="g3-early-drop", seq=7)).encode("utf-8")
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall(
            (
                "POST /predict HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(full_body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("utf-8") + full_body
        )
    time.sleep(0.45)

    conn = http.client.HTTPConnection(host, port, timeout=10.0)
    try:
        for seq_num in (0, 1):
            b = json.dumps(dict(valid_payload, session_id="g3-reconn-seq", seq=seq_num)).encode("utf-8")
            conn.request("POST", "/predict", body=b, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"Sequential post seq={seq_num} failed: {resp.status}")
            parsed = json.loads(resp.read().decode("utf-8"))
            validate_right7_predict_response(
                parsed,
                expected_session_id="g3-reconn-seq",
                expected_seq=seq_num,
                expected_seed_mode=seed_mode,
                expected_seed_base=seed_base,
            )
    finally:
        conn.close()

    # 5. Stale / mismatched response guard negative tests on real returned payload
    stale_checks: list[dict[str, Any]] = []
    stale_variants = [
        ("stale_seq_mismatch", {**r_reconn, "seq": 0}, "g3-reconnect-ok", 1),
        ("stale_session_mismatch", {**r_reconn, "session_id": "g3-timeout-abort"}, "g3-reconnect-ok", 0),
        ("wrong_schema_rejected", {**r_reconn, "schema": "wrong_v0"}, "g3-reconnect-ok", 0),
        ("hardware_connected_true_rejected", {**r_reconn, "hardware_connected": True}, "g3-reconnect-ok", 0),
        ("wrong_seed_rejected", {**r_reconn, "sampling_seed": seed_base + 999}, "g3-reconnect-ok", 0),
    ]
    for name, mutated, exp_sess, exp_seq in stale_variants:
        rejected = False
        try:
            validate_right7_predict_response(
                mutated,
                expected_session_id=exp_sess,
                expected_seq=exp_seq,
                expected_seed_mode=seed_mode,
                expected_seed_base=seed_base,
            )
        except ValueError:
            rejected = True
        stale_checks.append({"case": name, "rejected": rejected})

    all_stale_rejected = all(c["rejected"] for c in stale_checks)
    passed = bool(timeout_triggered and truncated_handled and all_stale_rejected and st_reconn == 200)
    return {
        "timeout_triggered": timeout_triggered,
        "timeout_exception": timeout_exc_type,
        "reconnect_status": st_reconn,
        "reconnect_rtt_ms": round(rtt_reconn, 3),
        "truncated_body_handled_400": truncated_handled,
        "sequential_connection_recovery_ok": True,
        "stale_response_rejection_checks": stale_checks,
        "all_stale_rejected": all_stale_rejected,
        "passed": passed,
    }


def _check_port_8010_refusal(server_script: Path, checkpoint: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(server_script), "--checkpoint", str(checkpoint), "--port", str(RESERVED_PRODUCTION_PORT)],
        capture_output=True,
        text=True,
        timeout=10,
        env=_isolated_subprocess_env(),
        check=False,
    )
    msg = (proc.stderr or "") + (proc.stdout or "")
    refused = proc.returncode != 0 and str(RESERVED_PRODUCTION_PORT) in msg
    return {"returncode": int(proc.returncode), "refused_port_8010": refused, "message": msg.strip()[:300]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full G3 isolated HTTP server & client compatibility gate.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--smolvla-root", default=None)
    parser.add_argument("--server-script", default=str(CANDIDATE_SERVER_DEFAULT))
    parser.add_argument("--baseline-server-script", default=os.environ.get("SMOLVLA_BASELINE_SERVER_SCRIPT", ""))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8011, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--expected-prefix", default=os.environ.get("EXPECTED_CONDA_PREFIX", ""))
    parser.add_argument("--expected-lerobot", default="0.7.0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.port == RESERVED_PRODUCTION_PORT:
        raise SystemExit(f"Refusing to run G3 against production port {RESERVED_PRODUCTION_PORT}")

    manifest_path = Path(args.manifest).resolve()
    smolvla_root = Path(args.smolvla_root).resolve() if args.smolvla_root else None
    manifest = load_replay_manifest(manifest_path, verify_hashes=True, smolvla_root=smolvla_root)
    smolvla_root = Path(args.smolvla_root or manifest["smolvla_root"]).resolve()
    checkpoint = Path(args.checkpoint or manifest["checkpoint_reference_path"]).resolve()
    ckpt_info = verify_checkpoint_manifest(checkpoint, Path(args.baseline_manifest).resolve())

    server_script = Path(args.server_script).resolve()
    baseline_server_script = Path(args.baseline_server_script).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"

    port_8010_guard = _check_port_8010_refusal(server_script, checkpoint)
    if not port_8010_guard["refused_port_8010"]:
        raise RuntimeError(f"Candidate server did not refuse production port 8010: {port_8010_guard}")

    w0 = manifest["windows"][0]
    w_alt = manifest["windows"][7] if len(manifest["windows"]) > 7 else manifest["windows"][1]
    valid_payload = build_http_payload_for_window(smolvla_root, w0, session_id="g3-valid", seq=0)

    # Phase A: --seed-mode fixed
    fixed_proc, fixed_drain, fixed_health, fixed_ready_s = _start_candidate_server(
        server_script, checkpoint, args.host, args.port, args.seed, "fixed",
        out_dir / "server_fixed.log",
    )
    try:
        validate_right7_health(
            fixed_health, str(checkpoint), expected_seed_mode="fixed", expected_lerobot=args.expected_lerobot
        )
        fixed_sens = _check_input_and_seed_sensitivities(
            base_url, smolvla_root, w0, w_alt, seed_mode="fixed", seed_base=args.seed
        )
        fixed_replay = _run_sequential_replay_pass(
            base_url, smolvla_root, manifest["windows"],
            session_id="g3-replay-fixed", seed_mode="fixed", seed_base=args.seed,
            artifact_dir=out_dir / "fixed_mode",
        )
    finally:
        _stop_server(fixed_proc, fixed_drain, args.host, args.port)

    # Phase B: --seed-mode sequence (default production runtime mode) + full fault injection
    seq_proc, seq_drain, seq_health, seq_ready_s = _start_candidate_server(
        server_script, checkpoint, args.host, args.port, args.seed, "sequence",
        out_dir / "server_sequence.log",
    )
    try:
        validate_right7_health(
            seq_health, str(checkpoint), expected_seed_mode="sequence", expected_lerobot=args.expected_lerobot
        )
        seq_sens = _check_input_and_seed_sensitivities(
            base_url, smolvla_root, w0, w_alt, seed_mode="sequence", seed_base=args.seed
        )
        seq_replay = _run_sequential_replay_pass(
            base_url, smolvla_root, manifest["windows"],
            session_id="g3-replay-sequence", seed_mode="sequence", seed_base=args.seed,
            artifact_dir=out_dir / "sequence_mode",
        )
        validation_matrix = _check_validation_and_malformed_requests(base_url, valid_payload)
        timeout_reconnect = _check_timeout_reconnect_and_stale(
            args.host, args.port, valid_payload, seed_mode="sequence", seed_base=args.seed
        )
    finally:
        _stop_server(seq_proc, seq_drain, args.host, args.port)

    env_info = verify_env_identity(
        expected_prefix=Path(args.expected_prefix),
        expected_lerobot_version=args.expected_lerobot,
        require_site_packages=True,
    )

    passed = bool(
        port_8010_guard["refused_port_8010"]
        and fixed_sens["passed"]
        and seq_sens["passed"]
        and validation_matrix["passed"]
        and timeout_reconnect["passed"]
        and fixed_replay["window_count"] == manifest["window_count"]
        and seq_replay["window_count"] == manifest["window_count"]
    )
    report = {
        "gate": "G3_http_compatibility_gate",
        "status": "PASSED" if passed else "FAILED",
        "endpoint": base_url,
        "hardware_connected": False,
        "production_port_8010_untouched_and_refused": port_8010_guard,
        "env_identity": env_info,
        "checkpoint": ckpt_info,
        "server_provenance": {
            "candidate_server_path": str(server_script),
            "candidate_server_sha256": sha256_file(server_script),
            "baseline_server_sha256": sha256_file(baseline_server_script) if baseline_server_script.is_file() else None,
        },
        "health_fixed": fixed_health,
        "health_sequence": seq_health,
        "processor_identity": seq_health.get("candidate_runtime", {}).get("processor_identity"),
        "cold_startup_seconds": {"fixed": round(fixed_ready_s, 3), "sequence": round(seq_ready_s, 3)},
        "fixed_seed_sensitivity": fixed_sens,
        "sequence_seed_sensitivity": seq_sens,
        "fixed_mode_replay": fixed_replay,
        "sequence_mode_replay": seq_replay,
        "validation_and_malformed_requests": validation_matrix,
        "timeout_reconnect_and_stale": timeout_reconnect,
    }
    report_path = out_dir / "g3_http_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate": report["gate"],
        "status": report["status"],
        "endpoint": base_url,
        "hardware_connected": False,
        "port_8010_refused": port_8010_guard["refused_port_8010"],
        "fixed_repeat_diff": fixed_sens["repeat_seq0_max_abs_diff"],
        "fixed_seq1_diff": fixed_sens["seq1_same_obs_max_abs_diff"],
        "sequence_seq1_diff": seq_sens["seq1_same_obs_max_abs_diff"],
        "cam_high_sensitivity": seq_sens["cam_high_changed_max_abs_diff"],
        "cam_wrist_sensitivity": seq_sens["cam_right_wrist_changed_max_abs_diff"],
        "fixed_rtt_p50_ms": round(fixed_replay["rtt_ms_summary"]["p50"], 2),
        "sequence_rtt_p50_ms": round(seq_replay["rtt_ms_summary"]["p50"], 2),
        "sequence_infer_p50_ms": round(seq_replay["inference_ms_summary"]["p50"], 2),
        "validation_cases_passed": f"{sum(c['passed'] for c in validation_matrix['cases'])}/{len(validation_matrix['cases'])}",
        "timeout_reconnect_passed": timeout_reconnect["passed"],
        "report_json": str(report_path),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
