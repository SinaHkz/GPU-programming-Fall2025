import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


def read_metrics_points(metrics_path: Path):
    points = []
    if not metrics_path.exists():
        return points
    with metrics_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return points


class Handler(BaseHTTPRequestHandler):
    run_dir: Path = None
    dash_dir: Path = None

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/metrics":
            metrics_path = self.run_dir / "metrics.jsonl"
            points = read_metrics_points(metrics_path)
            body = json.dumps({"run_dir": str(self.run_dir), "points": points}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if u.path == "/" or u.path == "/index.html":
            p = self.dash_dir / "index.html"
        elif u.path == "/app.js":
            p = self.dash_dir / "app.js"
        else:
            self.send_response(404)
            self.end_headers()
            return

        data = p.read_bytes()
        self.send_response(200)
        if p.suffix == ".js":
            self.send_header("Content-Type", "application/javascript")
        else:
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory (e.g. runs/.... or runs/latest)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.name == "latest":
        latest_txt = run_dir.parent / "latest.txt"
        if latest_txt.exists():
            run_dir = Path(latest_txt.read_text(encoding="utf-8").strip())
    run_dir = run_dir.resolve()

    dash_dir = (Path(__file__).parent.parent / "dashboard").resolve()
    if not dash_dir.exists():
        raise SystemExit(f"Missing dashboard directory: {dash_dir}")

    Handler.run_dir = run_dir
    Handler.dash_dir = dash_dir

    host = os.environ.get("HOST", "127.0.0.1")
    httpd = HTTPServer((host, args.port), Handler)
    print(f"Dashboard: http://{host}:{args.port}  (run_dir={run_dir})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

