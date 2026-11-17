import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODAL_BASE = os.environ.get(
    "MODAL_BASE_URL",
    "https://<YOUR-WORKSPACE>--smolvla-live-server-serve.modal.run",
).rstrip("/")


class ProxyHandler(BaseHTTPRequestHandler):
    def _forward(self, method: str) -> None:
        target_url = MODAL_BASE + self.path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None

        headers = {}
        if "Content-Type" in self.headers:
            headers["Content-Type"] = self.headers["Content-Type"]

        req = urllib.request.Request(
            target_url, data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type",
                    resp.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as exc:
            msg = str(exc).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")


if __name__ == "__main__":
    host = os.environ.get("BRIDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("BRIDGE_PORT", "8008"))
    server = ThreadingHTTPServer((host, port), ProxyHandler)
    print(f"Bridge listening on {host}:{port} -> {MODAL_BASE}")
    server.serve_forever()
