"""HTTP integration tests for Stage D1 localhost interactive panel."""

from __future__ import annotations

import json
import time
import unittest
from urllib import error, request

from v070_demo.contracts import CommandStatus, SessionState
from v070_demo.executor_bridge import MockRobotPublisher
from v070_demo.session_service import SessionService
from v070_demo.task_parser import parse_batch_instruction, parse_priority_command
from v070_demo.web_panel import PANEL_HTML, DemoPanelServer


BATCH_PROMPT = "把桌上的红、黄、蓝三个木块逐个放到棕色区域，优先抓容易的。"


class TestD1WebPanelHTTP(unittest.TestCase):
    def _req_json(self, base_url: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        req = request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=3.0) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_refuses_non_loopback_without_explicit_allow_lan(self) -> None:
        svc = SessionService(MockRobotPublisher(step_delay_s=0.0))
        with self.assertRaises(ValueError):
            DemoPanelServer(svc, host="0.0.0.0", port=0, allow_lan=False)

    def test_panel_http_lifecycle_readonly_isolation_and_incremental_events(self) -> None:
        bridge = MockRobotPublisher(step_delay_s=0.01)
        svc = SessionService(bridge)
        server = DemoPanelServer(svc, host="127.0.0.1", port=0)
        server.start()
        base = f"http://127.0.0.1:{server.port}"
        try:
            # Opening GET / never triggers hardware or session start
            with request.urlopen(f"{base}/", timeout=3.0) as resp:
                html = resp.read().decode("utf-8")
                self.assertEqual(resp.status, 200)
                self.assertIn("SIMULATION_MOCK", html)
                self.assertIn("In-Flight Priority (1-Click Next Block)", html)
                self.assertIn("btnPriRed", html)
                self.assertIn("btnPriYellow", html)
                self.assertIn("btnPriBlue", html)
                self.assertIn("btnReturnHome", html)
                self.assertIn("White Workspace Grid (R1-R9)", html)

            st, snap0 = self._req_json(base, "GET", "/api/snapshot?client_id=tab1&since_event_seq=0")
            self.assertEqual(st, 200)
            self.assertEqual(snap0["session_state"], SessionState.PREPARING.value)
            self.assertEqual(snap0["switch_window"], "IDLE")

            # Start session as tab1
            st, snap1 = self._req_json(base, "POST", "/api/session/start", {"client_id": "tab1", "scene_id": "scene_1_standard"})
            self.assertEqual(st, 200)
            self.assertEqual(snap1["session_state"], SessionState.READY.value)
            self.assertTrue(snap1["lease"]["is_owner"])

            # Second viewer tab2 is strictly read-only
            _, view_tab2 = self._req_json(base, "GET", "/api/snapshot?client_id=tab2&since_event_seq=0")
            self.assertFalse(view_tab2["lease"]["is_owner"])
            _, denied = self._req_json(
                base,
                "POST",
                "/api/command",
                {
                    "client_id": "tab2",
                    "command_id": "tab2_try",
                    "command_type": "START_BATCH",
                    "raw_text": BATCH_PROMPT,
                },
            )
            self.assertEqual(denied["status"], CommandStatus.REJECTED.value)

            # Heartbeat + START_BATCH as owner tab1
            hb_st, hb = self._req_json(base, "POST", "/api/heartbeat", {"client_id": "tab1", "lease_epoch": snap1["lease"]["lease_epoch"]})
            self.assertEqual(hb_st, 200)
            self.assertTrue(hb["ok"])

            st_b, rec_b = self._req_json(
                base,
                "POST",
                "/api/command",
                {
                    "client_id": "tab1",
                    "command_id": "http_batch_1",
                    "command_type": "START_BATCH",
                    "raw_text": BATCH_PROMPT,
                    "scene_id": "scene_1_standard",
                },
            )
            self.assertEqual(st_b, 200)
            self.assertEqual(rec_b["status"], CommandStatus.APPLIED.value)

            deadline = time.monotonic() + 5.0
            final_snap = None
            while time.monotonic() < deadline:
                _, cur = self._req_json(base, "GET", "/api/snapshot?client_id=tab1&since_event_seq=0")
                if cur["session_state"] == SessionState.READY.value and cur["metrics"]["completed_count"] == 3:
                    final_snap = cur
                    break
                time.sleep(0.03)
            self.assertIsNotNone(final_snap)
            self.assertEqual(final_snap["metrics"]["headline_score"], "3/3")

            # Incremental polling returns only new events after latest_event_seq
            seq = final_snap["latest_event_seq"]
            _, inc = self._req_json(base, "GET", f"/api/snapshot?client_id=tab1&since_event_seq={seq}")
            self.assertEqual(inc["events"], [])
        finally:
            server.stop()

    def test_panel_html_is_english_only_and_english_presets_parse_validly(self) -> None:
        self.assertTrue(PANEL_HTML.isascii())
        self.assertIn("scene_clear_table", PANEL_HTML)
        self.assertIn("Clear the table", PANEL_HTML)
        for prompt in (
            "Pick up the yellow wooden block and place it on the brown target area.",
            "Pick up the red wooden block and place it on the brown target area.",
            "Pick up the three wooden blocks (red, yellow, and blue) in exact order and place them on the brown target area. Re-observe and retry on empty grasp (max 2 attempts per block).",
        ):
            parsed = parse_batch_instruction(prompt)
            self.assertTrue(parsed.valid, msg=f"{prompt} -> {parsed}")
        clear_parsed = parse_batch_instruction(
            "Clear the table: continuously pick up any wooden blocks (red, yellow, or blue) from the white grid and place them on the brown target area until the white grid is empty.",
            scene_id="scene_clear_table",
        )
        self.assertTrue(clear_parsed.valid)
        self.assertTrue(clear_parsed.clear_table)
        self.assertIsNone(clear_parsed.expected_count)
        for color in ("red", "yellow", "blue"):
            upd = parse_priority_command(f"Prioritize {color} for the remaining objects")
            self.assertTrue(upd.valid, msg=f"{color} -> {upd}")
            self.assertEqual(upd.priority_color, color)


if __name__ == "__main__":
    unittest.main()

