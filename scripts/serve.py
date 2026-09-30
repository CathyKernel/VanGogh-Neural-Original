#!/usr/bin/env python3
"""Development web server for the renderer.

Serves ``renderer/web`` with correct MIME types for .glsl/.vert/.frag and
``Cache-Control: no-store`` so shader edits show up on every reload.

    python scripts/serve.py [--port 8899] [--root renderer/web]
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path

MIME = {
    ".glsl": "text/plain; charset=utf-8",
    ".vert": "text/plain; charset=utf-8",
    ".frag": "text/plain; charset=utf-8",
    ".bin": "application/octet-stream",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in MIME:
            return MIME[suffix]
        return super().guess_type(path)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--root", default="renderer/web")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / args.root
    if not root.exists():
        raise SystemExit(f"renderer root not found: {root}")

    import os

    os.chdir(root)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), NoCacheHandler) as httpd:
        print(f"serving {root} at http://127.0.0.1:{args.port}/ ( Ctrl-C to stop )",
              flush=True)
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
