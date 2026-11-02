"""Dependency-free local web app. Run: python server.py"""
from __future__ import annotations

import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from predict import Predictor, ROOT

PORT = int(os.environ.get("PORT", "8000"))
predictor = None


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, data, mime="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.reply(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if parsed.path == "/api/health":
            return self.reply(200, {"ready": predictor is not None,
                                    "model": "EfficientNetV2-B0 + Grad-CAM"})
        if parsed.path == "/sample":
            klass = parse_qs(parsed.query).get("class", [""])[0]
            if klass not in ("bottle", "cup", "pen", "spoon"):
                return self.reply(404, {"error": "Unknown sample"})
            candidate = ROOT / "web" / "samples" / f"{klass}.jpg"
            if not candidate.exists():
                return self.reply(404, {"error": "Sample missing"})
            return self.reply(200, candidate.read_bytes(), "image/jpeg")
        return self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self.reply(404, {"error": "Not found"})
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 15_000_000:
            return self.reply(413, {"error": "Image must be under 10 MB"})
        try:
            request = json.loads(self.rfile.read(length))
            value = request["image"].split(",", 1)[-1]
            raw = base64.b64decode(value, validate=True)
            result = predictor.predict_bytes(raw)
        except (KeyError, ValueError, TypeError) as exc:
            return self.reply(400, {"error": str(exc)})
        return self.reply(200, result)


if __name__ == "__main__":
    predictor = Predictor()
    print(f"Open http://127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
