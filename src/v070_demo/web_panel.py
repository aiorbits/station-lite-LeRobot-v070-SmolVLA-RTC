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
<title>Station Lite - LeRobot v0.70 Sorting Session Panel</title>
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
  .pill { padding: 3px 9px; border-radius: 999px; font-weight: 600; font-size: 12px; background: #0f172a; border: 1px solid var(--border); display: inline-block; }
  .pill.sim { border-color: var(--warn); color: #fcd34d; }
  .pill.ok { border-color: var(--ok); color: #86efac; }
  .pill.warn { border-color: var(--warn); color: #fde68a; }
  .pill.err { border-color: var(--err); color: #fca5a5; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; align-items: start; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0; }
  textarea, input[type="text"], select { width: 100%; background: #0f172a; color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; margin-top: 4px; margin-bottom: 8px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 6px; }
  button { padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border); background: #0284c7; color: #fff; font-weight: 600; cursor: pointer; font-size: 12px; }
  button.secondary { background: #334155; }
  button.warn { background: #b45309; }
  button.danger { background: #b91c1c; }
  button.pri-btn { flex: 1 1 95px; padding: 10px 12px; font-size: 13px; border: 2px solid transparent; transition: transform 0.08s ease; }
  button.pri-red { background: #b91c1c; border-color: #ef4444; }
  button.pri-yellow { background: #a16207; border-color: #eab308; color: #fefce8; }
  button.pri-blue { background: #1d4ed8; border-color: #3b82f6; }
  button.pri-active { box-shadow: 0 0 0 2px #f8fafc; }
  button:disabled { opacity: 0.4; cursor: not-allowed; box-shadow: none; }
  table { width: 100%; table-layout: fixed; border-collapse: collapse; margin-top: 8px; font-size: 12px; }
  th, td { text-align: left; border-bottom: 1px solid var(--border); padding: 5px 6px; vertical-align: top; word-break: break-word; overflow-wrap: anywhere; }
  th { color: var(--muted); font-weight: 600; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; word-break: break-word; overflow-wrap: anywhere; }
  .log { max-height: 290px; overflow-y: auto; overflow-x: hidden; background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 8px; white-space: pre-wrap; word-break: break-word; line-height: 1.4; }
  .headline { font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }
  svg { width: 100%; height: auto; display: block; background: #0b1120; border: 1px solid var(--border); border-radius: 6px; }
  .cam-tile { width: calc(50% - 4px); background: #0b1120; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .cam-tile img { width: 100%; display: block; }
  .cam-cap { padding: 3px 6px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); }
  .note { color: var(--muted); font-size: 11.5px; margin-top: 6px; line-height: 1.4; }
  .box { background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 8px; margin-top: 6px; }
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
        <colgroup>
          <col style="width:22%" />
          <col style="width:12%" />
          <col style="width:22%" />
          <col style="width:8%" />
          <col style="width:36%" />
        </colgroup>
        <thead><tr><th>ID</th><th>Color</th><th>Status</th><th>Att</th><th>Reason</th></tr></thead>
        <tbody id="objectRows"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>2. In-Flight Priority (1-Click Next Block)</h2>
      <div class="box">
        <div id="activePromptBox" class="mono">Active (Gen #0): (none)</div>
        <div id="pendingPromptBox" class="mono" style="color:#fde68a;margin-top:4px;">Next Target: (default sequence)</div>
        <div style="margin-top:6px;">
          <span id="switchWindowBadge" class="pill">Switch Window: Idle</span>
        </div>
      </div>
      <div class="row" style="margin-top:10px;">
        <button id="btnPriRed" class="pri-btn pri-red" onclick="quickPriority('red')" disabled>Red</button>
        <button id="btnPriYellow" class="pri-btn pri-yellow" onclick="quickPriority('yellow')" disabled>Yellow</button>
        <button id="btnPriBlue" class="pri-btn pri-blue" onclick="quickPriority('blue')" disabled>Blue</button>
      </div>
      <div class="note">Click any color to set the next block to pick: switches immediately before Chunk 3 (GRASPING) starts, or queues until placement once grasping begins.</div>
      <table>
        <colgroup>
          <col style="width:18%" />
          <col style="width:22%" />
          <col style="width:18%" />
          <col style="width:8%" />
          <col style="width:34%" />
        </colgroup>
        <thead><tr><th>Cmd</th><th>Type</th><th>Status</th><th>Gen</th><th>Receipt / Boundary</th></tr></thead>
        <tbody id="commandRows"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>3. Observation &amp; Progress</h2>
      <svg id="sceneCanvas" viewBox="0 0 380 215"></svg>
      <div id="liveFramesRow" class="row" style="display:none;gap:8px;margin-top:8px;">
        <div class="cam-tile">
          <img id="camHighImg" alt="cam_high" />
          <div class="cam-cap mono">cam_high (overhead)</div>
        </div>
        <div class="cam-tile">
          <img id="camWristImg" alt="cam_right_wrist" />
          <div class="cam-cap mono">cam_right_wrist</div>
        </div>
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
        <button id="btnReturnHome" style="background:#0f766e;" onclick="submitCommand('RETURN_HOME')" disabled>Return Home (Empty Gripper)</button>
        <button id="btnClose" class="secondary" onclick="submitCommand('CLOSE_SESSION')" disabled>Close Session</button>
      </div>
      <div class="note">Hardware safety: browser controls cannot replace physical E-stop; Return Home requires idle batch and verified open gripper.</div>
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
const seenEventSeqs = new Set();
let pollInFlight = false;

function nextCmdId(prefix) { return prefix + '_' + Date.now() + '_' + Math.random().toString(16).slice(2, 6); }
function quickPriority(color) {
  submitCommand('UPDATE_PRIORITY', 'Prioritize ' + color + ' for the remaining objects');
}

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

async function submitCommand(commandType, overrideRawText) {
  let rawText = overrideRawText || '';
  const scope = 'remaining_objects';
  if (commandType === 'START_BATCH') rawText = document.getElementById('batchPrompt').value;
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

function colorHex(c) { return c === 'red' ? '#ef4444' : c === 'yellow' ? '#eab308' : '#3b82f6'; }
function escHtml(v) {
  return String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function clampVal(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function shortStatusLabel(st) {
  if (st === 'PLACED_CONFIRMED') return 'PLACED';
  if (st === 'NEEDS_OPERATOR') return 'OP!';
  if (st === 'HELD_CONFIRMED') return 'HELD';
  if (st === 'FAILED') return 'FAIL';
  return 'WAIT';
}

function formatEventLine(ev) {
  const parts = [];
  if (ev.object_id) parts.push('obj=' + ev.object_id);
  if (ev.preempted_object_id) parts.push('preempted=' + ev.preempted_object_id);
  if (ev.after_phase) parts.push('after=' + ev.after_phase);
  if (ev.priority_color) parts.push('priority=' + ev.priority_color);
  if (ev.command_type) parts.push(ev.command_type + ':' + (ev.status || ''));
  if (ev.metrics && ev.metrics.headline_score) parts.push('score=' + ev.metrics.headline_score);
  if (ev.reason) parts.push(ev.reason);
  const tail = parts.length ? (' | ' + parts.join(' | ')) : '';
  return '#' + ev.seq + ' [g' + ev.generation + '] ' + ev.kind + tail;
}

function renderCanvas(snap) {
  const svg = document.getElementById('sceneCanvas');
  const gx = 20, gy = 40, cw = 68, ch = 50;
  let html = '<rect x="12" y="12" width="224" height="190" fill="#1e293b" stroke="#475569" rx="6"/>';
  html += '<text x="20" y="29" fill="#cbd5e1" font-size="11" font-weight="600">White Workspace Grid (R1-R9)</text>';
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const idx = r * 3 + c + 1;
      const rx = gx + c * cw;
      const ry = gy + r * ch;
      html += '<rect x="'+rx+'" y="'+ry+'" width="'+(cw-2)+'" height="'+(ch-2)+'" fill="#0f172a" stroke="#334155" stroke-dasharray="3 2" rx="3"/>';
      html += '<text x="'+(rx+4)+'" y="'+(ry+11)+'" fill="#64748b" font-size="9">R'+idx+'</text>';
    }
  }
  html += '<rect x="248" y="12" width="120" height="190" fill="#451a03" stroke="#92400e" rx="6"/>';
  html += '<text x="256" y="29" fill="#fcd34d" font-size="11" font-weight="600">Brown Target Zone</text>';
  for (let s = 0; s < 3; s++) {
    const sy = 42 + s * 50;
    html += '<rect x="268" y="'+sy+'" width="80" height="44" fill="#291202" stroke="#b45309" stroke-dasharray="3 2" rx="4"/>';
    html += '<text x="273" y="'+(sy+12)+'" fill="#d97706" font-size="8.5">Slot '+s+'</text>';
  }
  const objs = snap.objects || [];
  let placedIdx = 0;
  objs.forEach((o) => {
    let cx = 0;
    let cy = 0;
    if (o.status === 'PLACED_CONFIRMED') {
      const slot = (o.destination_slot !== null && o.destination_slot !== undefined)
        ? clampVal(Number(o.destination_slot), 0, 2)
        : clampVal(placedIdx, 0, 2);
      cx = 308;
      cy = 62 + slot * 50;
      placedIdx++;
    } else {
      const gm = /^R([1-9])$/.exec(String(o.grid_cell || ''));
      if (gm) {
        const idx = parseInt(gm[1], 10) - 1;
        const row = Math.floor(idx / 3);
        const col = idx % 3;
        cx = gx + col * cw + 33;
        cy = gy + row * ch + 22;
      } else {
        const px = Number(o.pose_estimate?.x ?? 0.25);
        const py = Number(o.pose_estimate?.y ?? 0.0);
        if (py > 0.06) {
          cx = clampVal(gx + 33 + ((0.175 - py) / 0.08) * (2 * cw), gx + 28, gx + 3 * cw - 28);
          cy = clampVal(gy + 22 + ((0.29 - px) / 0.08) * (2 * ch), gy + 20, gy + 3 * ch - 26);
        } else {
          cx = clampVal(gx + 33 + ((py + 0.09) / 0.18) * (2 * cw), gx + 28, gx + 3 * cw - 28);
          cy = clampVal(gy + 22 + ((0.30 - px) / 0.14) * (2 * ch), gy + 20, gy + 3 * ch - 26);
        }
      }
    }
    const isCurrent = snap.current_object_id === o.object_id;
    const isQueued = !isCurrent && snap.priority_color === o.color && o.status !== 'PLACED_CONFIRMED';
    const activeRing = isCurrent ? '<circle cx="'+cx+'" cy="'+(cy-3)+'" r="15" fill="none" stroke="#38bdf8" stroke-width="2.5"/>' : '';
    const queuedRing = isQueued ? '<circle cx="'+cx+'" cy="'+(cy-3)+'" r="15" fill="none" stroke="#facc15" stroke-width="2" stroke-dasharray="3 2"/>' : '';
    html += activeRing + queuedRing + '<rect x="'+(cx-10)+'" y="'+(cy-13)+'" width="20" height="20" rx="4" fill="'+colorHex(o.color)+'" stroke="#f8fafc" stroke-width="1.2"/>';
    html += '<text x="'+cx+'" y="'+(cy+19)+'" text-anchor="middle" fill="#e2e8f0" font-size="8.5">'+escHtml(o.color)+' ('+shortStatusLabel(o.status)+')</text>';
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

  const busyHome = Boolean(snap.returning_home);
  const workerActive = Boolean(snap.worker_active);
  document.getElementById('btnStartSession').disabled = currentState !== 'PREPARING';
  document.getElementById('btnStartBatch').disabled = !(isOwner && currentState === 'READY' && !busyHome);
  const canPriority = isOwner && !busyHome && ['RUNNING','PAUSED','SWITCH_PENDING'].includes(currentState) && workerActive;
  ['red', 'yellow', 'blue'].forEach((c) => {
    const id = 'btnPri' + c.charAt(0).toUpperCase() + c.slice(1);
    const btn = document.getElementById(id);
    if (btn) {
      btn.disabled = !canPriority;
      btn.classList.toggle('pri-active', snap.priority_color === c);
    }
  });
  document.getElementById('btnPause').disabled = !(isOwner && ['RUNNING','SWITCH_PENDING'].includes(currentState));
  document.getElementById('btnResume').disabled = !(isOwner && currentState === 'PAUSED' && workerActive);
  document.getElementById('btnCancel').disabled = !(isOwner && ['RUNNING','PAUSE_PENDING','PAUSED','SWITCH_PENDING'].includes(currentState) && workerActive);
  document.getElementById('btnReturnHome').disabled = !(isOwner && ['READY','PAUSED'].includes(currentState) && !workerActive && !busyHome);
  document.getElementById('btnClose').disabled = !isOwner || currentState === 'CLOSED';

  document.getElementById('activePromptBox').textContent = 'Active (Gen #' + activeGen + '): ' + (snap.active_instruction || '(none)');
  const pend = snap.pending_instruction;
  const priText = snap.priority_color ? ('Active Priority: ' + snap.priority_color.toUpperCase()) : 'Next Target: (default sequence)';
  document.getElementById('pendingPromptBox').textContent = pend
    ? ('Pending [' + pend.status + ']: ' + pend.raw_text + ' -- ' + pend.reason)
    : priText;
  const swEl = document.getElementById('switchWindowBadge');
  const sw = snap.switch_window || 'IDLE';
  if (sw === 'PRE_GRASP_IMMEDIATE') {
    swEl.className = 'pill ok';
    swEl.textContent = 'Switch Window: Immediate (Before Chunk 3 GRASPING)';
  } else if (sw === 'HOLDING_QUEUED_NEXT') {
    swEl.className = 'pill warn';
    swEl.textContent = 'Switch Window: Queued After Place (Chunk 3+ Active)';
  } else {
    swEl.className = 'pill';
    swEl.textContent = 'Switch Window: Idle';
  }

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
    let added = false;
    snap.events.forEach((ev) => {
      if (!seenEventSeqs.has(ev.seq)) {
        seenEventSeqs.add(ev.seq);
        eventHistory.push(formatEventLine(ev));
        added = true;
      }
      lastSeq = Math.max(lastSeq, ev.seq);
    });
    if (added) {
      const el = document.getElementById('eventLog');
      el.textContent = eventHistory.slice(-40).join('\\n');
      el.scrollTop = el.scrollHeight;
    }
  }
  renderCanvas(snap);
}

async function pollSnapshot() {
  if (pollInFlight) return;
  pollInFlight = true;
  try {
    const res = await fetch('/api/snapshot?client_id=' + encodeURIComponent(clientId) + '&since_event_seq=' + lastSeq);
    const snap = await res.json();
    renderSnapshot(snap);
  } finally {
    pollInFlight = false;
  }
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
