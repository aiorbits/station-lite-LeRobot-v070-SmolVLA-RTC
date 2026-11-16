"""Localhost interactive session panel and HTTP API for Stage D1."""

from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .contracts import CommandType
from .executor_bridge import SCENE_PRESETS
from .session_service import SessionService


PANEL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LeRobot v0.70 Three-Object Sorting Session Panel (D1 Mock)</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --err: #ef4444;
  }
  * { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  body { margin: 0; padding: 16px; background: var(--bg); color: var(--text); font-size: 13px; }
  h1 { margin: 0 0 8px 0; font-size: 18px; }
  h2 { margin: 0 0 8px 0; font-size: 14px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .banner { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 10px 14px; background: #172554; border: 1px solid #2563eb; border-radius: 8px; margin-bottom: 14px; }
  .pill { padding: 3px 9px; border-radius: 999px; font-weight: 600; font-size: 12px; background: #0f172a; border: 1px solid var(--border); }
  .pill.sim { border-color: var(--warn); color: #fcd34d; }
  .pill.ok { border-color: var(--ok); color: #86efac; }
  .pill.warn { border-color: var(--warn); color: #fde68a; }
  .pill.err { border-color: var(--err); color: #fca5a5; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  textarea, input[type="text"], select { width: 100%; background: #0f172a; color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; margin-top: 4px; margin-bottom: 8px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 6px; }
  button { padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border); background: #0284c7; color: #fff; font-weight: 600; cursor: pointer; font-size: 12px; }
  button.secondary { background: #334155; }
  button.warn { background: #b45309; }
  button.danger { background: #b91c1c; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 12px; }
  th, td { text-align: left; border-bottom: 1px solid var(--border); padding: 5px 6px; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
  .log { max-height: 210px; overflow-y: auto; background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 8px; }
  .headline { font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }
  svg { width: 100%; max-height: 210px; background: #0b1120; border: 1px solid var(--border); border-radius: 6px; }
  .note { color: var(--muted); font-size: 11.5px; margin-top: 4px; }
</style>
</head>
<body>
  <div class="banner">
    <strong>Station Lite &middot; LeRobot v0.7.0 Closed-Loop Sorting Panel</strong>
    <span id="badgeAdmission" class="pill sim">SIMULATION_MOCK</span>
    <span id="badgeSession" class="pill">PREPARING</span>
    <span id="badgeContact" class="pill">EMPTY_CONFIRMED</span>
    <span id="badgeGen" class="pill">Gen #0</span>
    <span id="badgeLease" class="pill">Viewer</span>
    <button id="btnClaim" class="secondary" onclick="claimControl()">Claim Control</button>
    <span id="heartbeatInfo" class="mono"></span>
  </div>

  <div class="grid">
    <div class="card">
      <h2>1. Task Setup &amp; Bound Object Table</h2>
      <label>Scene Preset
        <select id="sceneSelect" onchange="onSceneChange()">
          <option value="scene_1_standard">Scene 1: Standard Separated (R/Y/B)</option>
          <option value="scene_2_mixed_distance">Scene 2: Mixed Distance (Yellow Nearest)</option>
          <option value="scene_3_empty_grasp_recovery">Scene 3: Empty-Grasp Recovery Test</option>
          <option value="scene_clear_table">Clear the table (Loop Until Empty)</option>
          <option value="scene_ambiguous_duplicate">Fault: Ambiguous Duplicate Red Blocks</option>
        </select>
      </label>
      <label>Initial Batch Instruction
        <textarea id="batchPrompt" rows="3">Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them on the brown target area. Re-observe and retry on empty grasp (max 2 attempts per block).</textarea>
      </label>
      <div class="row">
        <button id="btnStartSession" onclick="startSession()">Start Session (No Robot Connect)</button>
        <button id="btnStartBatch" onclick="submitCommand('START_BATCH')" disabled>Start Batch</button>
        <button class="secondary" onclick="fillPrompt('Pick up the yellow wooden block and place it on the brown target area.')">Single Block (Yellow)</button>
        <button class="secondary" onclick="fillPrompt('Pick up the red wooden block and place it on the brown target area.')">Single Block (Red)</button>
        <button class="secondary" onclick="fillPrompt(SCENE1_PROMPT)">Three-Block Preset</button>
        <button class="secondary" onclick="selectClearTable()">Clear the table</button>
      </div>
      <table>
        <thead><tr><th>ID</th><th>Color</th><th>Status</th><th>Att</th><th>Reason</th></tr></thead>
        <tbody id="objectRows"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>2. In-Flight Priority Update &amp; Receipts</h2>
      <div id="activePromptBox" class="mono">Active: (none)</div>
      <div id="pendingPromptBox" class="mono" style="color:#fde68a;margin-top:4px;">Pending: (none)</div>
      <label>New In-Flight Instruction
        <input id="liveCmdInput" type="text" value="Prioritize red for the remaining objects" />
      </label>
      <label>Update Scope
        <select id="scopeSelect">
          <option value="remaining_objects">remaining_objects (Phase 1 Safe Checkpoint)</option>
          <option value="current_approach" disabled>current_approach (Locked until D4 Verification)</option>
        </select>
      </label>
      <div class="row">
        <button id="btnSendUpdate" onclick="submitCommand('UPDATE_PRIORITY')" disabled>Send Priority Command</button>
        <button class="secondary" onclick="quickUpdate('Prioritize red for the remaining objects')">Prioritize Red</button>
        <button class="secondary" onclick="quickUpdate('Prioritize yellow for the remaining objects')">Prioritize Yellow</button>
        <button class="secondary" onclick="quickUpdate('Prioritize blue for the remaining objects')">Prioritize Blue</button>
      </div>
      <table>
        <thead><tr><th>Cmd</th><th>Type</th><th>Status</th><th>Gen</th><th>Receipt / Boundary</th></tr></thead>
        <tbody id="commandRows"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>3. Observation &amp; Progress</h2>
      <svg id="sceneCanvas" viewBox="0 0 360 190"></svg>
      <div id="liveFramesRow" class="row" style="display:none;gap:8px;margin-top:6px;">
        <img id="camHighImg" alt="cam_high" style="width:48%;border-radius:6px;border:1px solid #334155;" />
        <img id="camWristImg" alt="cam_right_wrist" style="width:48%;border-radius:6px;border:1px solid #334155;" />
      </div>
      <div class="row mono" style="justify-content:space-between;">
        <span id="phaseInfo">Phase: IDLE</span>
        <span id="targetInfo">Current: -</span>
        <span id="obsAgeInfo">Obs Age: -</span>
      </div>
      <div id="runtimeInfo" class="mono" style="margin-top:4px;color:#93c5fd;"></div>
      <h2 style="margin-top:12px;">4. Session Controls (No Auto-Home / Reset)</h2>
      <div class="row">
        <button id="btnPause" class="warn" onclick="submitCommand('PAUSE')" disabled>Pause</button>
        <button id="btnResume" onclick="submitCommand('RESUME')" disabled>Resume (Re-observe)</button>
        <button id="btnCancel" class="danger" onclick="submitCommand('CANCEL_BATCH')" disabled>Cancel Batch</button>
        <button id="btnClose" class="secondary" onclick="submitCommand('CLOSE_SESSION')" disabled>Close Session</button>
        <button disabled title="Blocked until D4 trajectory-cancel verification">Switch Active Target Mid-Approach (D4)</button>
      </div>
      <div class="note">Hardware safety: browser controls cannot replace physical E-stop and do not send implicit /reset or go_home().</div>
    </div>

    <div class="card">
      <h2>5. Headline Outcome &amp; Event Timeline</h2>
      <div id="headlineMetric" class="headline">Completed 0/0 | Time 0.00s | Interventions 0</div>
      <div id="eventLog" class="log mono"></div>
    </div>
  </div>

<script>
const clientId = (() => {
  let cid = sessionStorage.getItem('v070_demo_client_id');
  if (!cid) { cid = 'ui_' + Math.random().toString(16).slice(2, 10); sessionStorage.setItem('v070_demo_client_id', cid); }
  return cid;
})();
let lastSeq = 0;
let currentLeaseEpoch = 0;
let activeGen = 0;
let isOwner = false;
let currentState = 'PREPARING';
const eventHistory = [];

function nextCmdId(prefix) { return prefix + '_' + Date.now() + '_' + Math.random().toString(16).slice(2, 6); }
function quickUpdate(text) { document.getElementById('liveCmdInput').value = text; submitCommand('UPDATE_PRIORITY'); }

async function apiPost(path, body) {
  const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return await res.json();
}

async function startSession() {
  const sceneId = document.getElementById('sceneSelect').value;
  await apiPost('/api/session/start', { client_id: clientId, scene_id: sceneId });
  await pollSnapshot();
}

async function claimControl() {
  await apiPost('/api/session/claim', { client_id: clientId });
  await pollSnapshot();
}

async function submitCommand(commandType) {
  let rawText = '';
  let scope = 'remaining_objects';
  if (commandType === 'START_BATCH') rawText = document.getElementById('batchPrompt').value;
  if (commandType === 'UPDATE_PRIORITY') {
    rawText = document.getElementById('liveCmdInput').value;
    scope = document.getElementById('scopeSelect').value;
  }
  await apiPost('/api/command', {
    client_id: clientId,
    command_id: nextCmdId(commandType.toLowerCase()),
    command_type: commandType,
    raw_text: rawText,
    scope: scope,
    scene_id: document.getElementById('sceneSelect').value,
    lease_epoch: currentLeaseEpoch,
    expected_active_generation: commandType === 'UPDATE_PRIORITY' ? activeGen : null
  });
  await pollSnapshot();
}

const SCENE1_PROMPT = 'Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them on the brown target area. Re-observe and retry on empty grasp (max 2 attempts per block).';
const CLEAR_TABLE_PROMPT = 'Clear the table: continuously pick up any wooden blocks (red, yellow, or blue) from the white grid and place them on the brown target area until the white grid is empty.';

function fillPrompt(txt) {
  const sel = document.getElementById('sceneSelect');
  if (sel.value === 'scene_clear_table') sel.value = 'scene_1_standard';
  document.getElementById('batchPrompt').value = txt;
}

function selectClearTable() {
  document.getElementById('sceneSelect').value = 'scene_clear_table';
  document.getElementById('batchPrompt').value = CLEAR_TABLE_PROMPT;
}

function onSceneChange() {
  const v = document.getElementById('sceneSelect').value;
  if (v === 'scene_clear_table') {
    document.getElementById('batchPrompt').value = CLEAR_TABLE_PROMPT;
  } else if (v === 'scene_1_standard') {
    document.getElementById('batchPrompt').value = SCENE1_PROMPT;
  }
}

function quickUpdate(txt) {
  document.getElementById('liveCmdInput').value = txt;
}

function colorHex(c) { return c === 'red' ? '#ef4444' : c === 'yellow' ? '#eab308' : '#3b82f6'; }
function escHtml(v) {
  return String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderCanvas(snap) {
  const svg = document.getElementById('sceneCanvas');
  let html = '<rect x="14" y="14" width="215" height="162" fill="#1e293b" stroke="#475569" rx="6"/>';
  html += '<text x="22" y="30" fill="#94a3b8" font-size="11">White Workspace Grid</text>';
  html += '<rect x="242" y="14" width="104" height="162" fill="#451a03" stroke="#92400e" rx="6"/>';
  html += '<text x="252" y="30" fill="#fcd34d" font-size="11">Brown Zone</text>';
  const objs = snap.objects || [];
  let placedIdx = 0;
  objs.forEach((o) => {
    let cx = 60 + ((o.pose_estimate?.y || 0) + 0.12) * 580;
    let cy = 155 - ((o.pose_estimate?.x || 0.22) - 0.15) * 650;
    if (o.status === 'PLACED_CONFIRMED') { cx = 294; cy = 58 + placedIdx * 42; placedIdx++; }
    const activeRing = snap.current_object_id === o.object_id ? '<circle cx="'+cx+'" cy="'+cy+'" r="17" fill="none" stroke="#38bdf8" stroke-width="2.5"/>' : '';
    html += activeRing + '<rect x="'+(cx-11)+'" y="'+(cy-11)+'" width="22" height="22" rx="4" fill="'+colorHex(o.color)+'" stroke="#f8fafc"/>';
    html += '<text x="'+cx+'" y="'+(cy+23)+'" text-anchor="middle" fill="#cbd5e1" font-size="9.5">'+o.color+' ('+o.status.slice(0,8)+')</text>';
  });
  svg.innerHTML = html;
}

function renderSnapshot(snap) {
  currentState = snap.session_state;
  activeGen = snap.active_generation || 0;
  currentLeaseEpoch = snap.lease?.lease_epoch || 0;
  isOwner = Boolean(snap.lease?.is_owner);

  document.getElementById('badgeAdmission').textContent = snap.hardware_admission;
  document.getElementById('badgeSession').textContent = snap.session_state;
  document.getElementById('badgeContact').textContent = snap.contact_held_state;
  document.getElementById('badgeGen').textContent = 'Gen #' + activeGen;
  const leasePill = document.getElementById('badgeLease');
  leasePill.textContent = isOwner ? ('OWNER (' + clientId + ')') : ('READ-ONLY | Owner=' + (snap.lease?.owner_client_id || 'none'));
  leasePill.className = 'pill ' + (snap.lease?.expired ? 'err' : (isOwner ? 'ok' : 'warn'));
  document.getElementById('heartbeatInfo').textContent = 'TTL=' + (snap.lease?.remaining_ttl_s ?? 0) + 's | Pubs=' + snap.publisher_count;

  document.getElementById('btnStartSession').disabled = currentState !== 'PREPARING';
  document.getElementById('btnStartBatch').disabled = !(isOwner && currentState === 'READY');
  document.getElementById('btnSendUpdate').disabled = !(isOwner && ['RUNNING','PAUSED','SWITCH_PENDING'].includes(currentState));
  document.getElementById('btnPause').disabled = !(isOwner && ['RUNNING','SWITCH_PENDING'].includes(currentState));
  document.getElementById('btnResume').disabled = !(isOwner && currentState === 'PAUSED');
  document.getElementById('btnCancel').disabled = !(isOwner && ['RUNNING','PAUSE_PENDING','PAUSED','SWITCH_PENDING'].includes(currentState));
  document.getElementById('btnClose').disabled = !isOwner || currentState === 'CLOSED';

  document.getElementById('activePromptBox').textContent = 'Active (Gen #' + activeGen + '): ' + (snap.active_instruction || '(none)');
  const pend = snap.pending_instruction;
  document.getElementById('pendingPromptBox').textContent = pend
    ? ('Pending [' + pend.status + ']: ' + pend.raw_text + ' -- ' + pend.reason)
    : 'Pending: (none)';

  document.getElementById('phaseInfo').textContent = 'Phase: ' + snap.execution_phase;
  document.getElementById('targetInfo').textContent = 'Current: ' + (snap.current_object_id || '-');
  const age = snap.latest_observation_age_ms;
  document.getElementById('obsAgeInfo').textContent = 'Obs Age: ' + (age !== null ? (age + ' ms ' + (snap.observation_fresh ? '(FRESH)' : '(STALE!)')) : '-');
  const rt = snap.runtime_status;
  if (rt) {
    document.getElementById('runtimeInfo').textContent =
      'Server=' + (rt.server_url || '-') +
      ' | HardwareAttached=' + Boolean(rt.hardware_attached) +
      ' | DryRunCases=' + (rt.dryrun_cases_passed ?? '-') +
      ' | Warmup=' + (rt.warmup_infer_ms ?? '-') + 'ms' +
      ' | Chunks=' + (rt.sent_chunks ?? 0);
    const fr = document.getElementById('liveFramesRow');
    if (rt.hardware_attached) {
      fr.style.display = 'flex';
      const ts = Date.now();
      document.getElementById('camHighImg').src = '/api/frame/cam_high?t=' + ts;
      document.getElementById('camWristImg').src = '/api/frame/cam_right_wrist?t=' + ts;
    } else {
      fr.style.display = 'none';
    }
  }

  document.getElementById('objectRows').innerHTML = (snap.objects || []).map((o) =>
    '<tr><td class="mono">'+escHtml(o.object_id)+'</td><td>'+escHtml(o.color)+'</td><td class="mono">'+escHtml(o.status)+'</td><td>'+escHtml(o.attempt_count)+'</td><td class="mono">'+escHtml(o.rejection_reason || o.selection_reason || '-')+'</td></tr>'
  ).join('');

  document.getElementById('commandRows').innerHTML = (snap.commands || []).slice().reverse().map((c) => {
    const boundary = c.switch_boundary ? ('gen '+c.switch_boundary.from_generation+'->'+c.switch_boundary.to_generation+' ('+c.switch_boundary.applied_latency_ms+'ms)') : c.reason;
    const genStr = c.applied_generation ?? '-';
    return '<tr><td class="mono">'+escHtml(c.command_id.slice(-10))+'</td><td class="mono">'+escHtml(c.command_type)+'</td><td class="mono">'+escHtml(c.status)+'</td><td>'+escHtml(genStr)+'</td><td class="mono">'+escHtml(boundary)+'</td></tr>';
  }).join('');

  const m = snap.metrics || {};
  document.getElementById('headlineMetric').textContent =
    'Completed ' + (m.headline_score || '0/0') + ' | Time ' + (m.batch_elapsed_s ?? 0) + 's | Interventions ' + (m.operator_interventions ?? 0);

  if (snap.events && snap.events.length) {
    snap.events.forEach((ev) => {
      eventHistory.push('#' + ev.seq + ' [g' + ev.generation + '] ' + ev.kind + ' ' + JSON.stringify(ev));
      lastSeq = Math.max(lastSeq, ev.seq);
    });
    const el = document.getElementById('eventLog');
    el.textContent = eventHistory.slice(-40).join('\\n');
    el.scrollTop = el.scrollHeight;
  }
  renderCanvas(snap);
}

async function pollSnapshot() {
  const res = await fetch('/api/snapshot?client_id=' + encodeURIComponent(clientId) + '&since_event_seq=' + lastSeq);
  const snap = await res.json();
  renderSnapshot(snap);
}

setInterval(() => { pollSnapshot().catch(() => {}); }, 350);
setInterval(() => {
  if (isOwner && currentState !== 'PREPARING' && currentState !== 'CLOSED') {
    apiPost('/api/heartbeat', { client_id: clientId, lease_epoch: currentLeaseEpoch }).catch(() => {});
  }
}, 900);
pollSnapshot();
</script>
</body>
</html>
"""


def create_panel_handler(service: SessionService) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP handler bound cleanly to a single SessionService owner."""

    class _PanelHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length_raw = self.headers.get("Content-Length", "0")
            try:
                length = int(length_raw)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header") from exc
            if length < 0 or length > 1024 * 1024:
                raise ValueError("Payload size out of bounds")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(PANEL_HTML)
                return
            if parsed.path == "/api/scenes":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "scenes": [
                            {"scene_id": s.scene_id, "description": s.description}
                            for s in SCENE_PRESETS.values()
                        ]
                    },
                )
                return
            if parsed.path == "/api/snapshot":
                qs = parse_qs(parsed.query)
                client_id = qs.get("client_id", [None])[0]
                since_raw = qs.get("since_event_seq", ["0"])[0]
                try:
                    since_seq = int(since_raw)
                except ValueError:
                    since_seq = 0
                snap = service.get_snapshot(client_id=client_id, since_event_seq=since_seq)
                self._send_json(HTTPStatus.OK, snap)
                return
            if parsed.path.startswith("/api/frame/"):
                cam_name = parsed.path.rsplit("/", 1)[-1]
                if cam_name not in ("cam_high", "cam_right_wrist"):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown camera: {cam_name}"})
                    return
                jpeg_fn = getattr(service.bridge, "latest_jpeg", None)
                jpeg = jpeg_fn(cam_name) if callable(jpeg_fn) else None
                if not jpeg:
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {parsed.path}"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            client_id = str(body.get("client_id") or "").strip()
            if not client_id:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "client_id is required"})
                return

            if parsed.path == "/api/session/start":
                scene_id = str(body.get("scene_id") or "scene_1_standard")
                try:
                    snap = service.start_session(client_id, scene_id=scene_id)
                except Exception as exc:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.OK, snap)
                return

            if parsed.path == "/api/session/claim":
                snap = service.claim_control(client_id)
                self._send_json(HTTPStatus.OK, snap)
                return

            if parsed.path == "/api/heartbeat":
                epoch_val = body.get("lease_epoch")
                epoch = int(epoch_val) if epoch_val is not None else None
                ok, reason = service.heartbeat(client_id, epoch)
                status = HTTPStatus.OK if ok else HTTPStatus.FORBIDDEN
                self._send_json(status, {"ok": ok, "reason": reason})
                return

            if parsed.path == "/api/command":
                cmd_type_raw = str(body.get("command_type") or "")
                try:
                    cmd_type = CommandType(cmd_type_raw)
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Unknown command_type: {cmd_type_raw}"})
                    return
                cmd_id = str(body.get("command_id") or f"cmd_{uuid.uuid4().hex[:10]}")
                lease_epoch_val = body.get("lease_epoch")
                lease_epoch = int(lease_epoch_val) if lease_epoch_val is not None else None
                exp_gen_val = body.get("expected_active_generation")
                exp_gen = int(exp_gen_val) if exp_gen_val is not None else None

                rec = service.submit_command(
                    client_id=client_id,
                    command_id=cmd_id,
                    command_type=cmd_type,
                    raw_text=str(body.get("raw_text") or ""),
                    scope=str(body.get("scope") or "remaining_objects"),
                    scene_id=body.get("scene_id"),
                    lease_epoch=lease_epoch,
                    expected_active_generation=exp_gen,
                )
                self._send_json(HTTPStatus.OK, rec.to_dict())
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {parsed.path}"})

    return _PanelHandler


class DemoPanelServer:
    """Managed localhost web server wrapping ThreadingHTTPServer cleanly."""

    def __init__(
        self,
        service: SessionService,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        allow_lan: bool = False,
    ) -> None:
        if not allow_lan and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"Non-loopback host {host!r} requires explicit allow_lan=True.")
        self.service = service
        self.httpd = ThreadingHTTPServer((host, port), create_panel_handler(service))
        self.host, self.port = self.httpd.server_address[:2]
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
