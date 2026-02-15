import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import parse_qs


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


def _commonpath_ok(root: Path, child: Path) -> bool:
    try:
        root_s = str(root.resolve())
        child_s = str(child.resolve())
        return os.path.commonpath([root_s, child_s]) == root_s
    except Exception:
        return False


def _resolve_run_dir(runs_root: Path, run_dir_str: str | None, default_run_dir: Path) -> Path:
    if not run_dir_str:
        return default_run_dir
    run_dir = (runs_root / run_dir_str).resolve()
    if not _commonpath_ok(runs_root, run_dir):
        raise ValueError("run_dir outside runs_root")
    return run_dir


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _read_csv_rows(path: Path, tail: int = 2000):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) > tail + 1:
        lines = [lines[0]] + lines[-tail:]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    rows = []
    for r in reader:
        rows.append(r)
    return rows


def _resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    if not p.is_absolute():
        p = (Path(__file__).parent.parent / p).resolve()
    return p


def _ensure_dataset_ready(dataset: str, data_dir: str, log_fn=None) -> None:
    ds = (dataset or "").lower()
    base = _resolve_data_dir(data_dir)
    if ds == "mnist":
        root = base / "mnist"
        required = [
            root / "train-images-idx3-ubyte",
            root / "train-labels-idx1-ubyte",
            root / "t10k-images-idx3-ubyte",
            root / "t10k-labels-idx1-ubyte",
        ]
        if all(p.exists() for p in required):
            if log_fn: log_fn(f"[dashboard] Dataset '{ds}' already present.\n")
            return
        cmd = [sys.executable, str(Path(__file__).parent / "download_mnist.py"), "--out", str(base)]
    elif ds == "cifar10":
        root = base / "cifar10" / "cifar-10-batches-bin"
        if root.exists():
            if log_fn: log_fn(f"[dashboard] Dataset '{ds}' already present.\n")
            return
        cmd = [sys.executable, str(Path(__file__).parent / "download_cifar10.py"), "--out", str(base)]
    else:
        return

    if log_fn: log_fn(f"[dashboard] Downloading dataset '{ds}'...\n")
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent.parent),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if log_fn: log_fn(f"[download] {line}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Download script exited with code {proc.returncode}")
    if log_fn: log_fn(f"[dashboard] Dataset '{ds}' ready.\n")


class Handler(BaseHTTPRequestHandler):
    runs_root: Path = None
    default_run_dir: Path = None
    dash_dir: Path = None

    proc_lock = threading.Lock()
    proc: subprocess.Popen | None = None
    log_buffer = []
    last_exit_code: int | None = None
    last_status: str = "idle"  # idle | running | stopped | error | completed

    def _send_json(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query or "")
        run_dir_q = (qs.get("run_dir") or [None])[0]

        try:
            run_dir = _resolve_run_dir(self.runs_root, run_dir_q, self.default_run_dir)
        except Exception as e:
            self._send_json({"error": str(e)}, code=400)
            return

        if u.path == "/api/terminal":
            offset = int((qs.get("offset") or ["0"])[0])
            with Handler.proc_lock:
                logs = Handler.log_buffer[offset:]
                new_offset = len(Handler.log_buffer)
            self._send_json({"logs": logs, "offset": new_offset})
            return

        if u.path == "/api/metrics":
            metrics_path = run_dir / "metrics.jsonl"
            points = read_metrics_points(metrics_path)
            self._send_json({"run_dir": str(run_dir), "points": points})
            return

        if u.path == "/api/runs":
            out = []
            if self.runs_root.exists():
                for p in sorted(self.runs_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if not p.is_dir():
                        continue
                    cfg = _read_json(p / "config.json")
                    out.append(
                        {
                            "name": p.name,
                            "run_dir": p.name,
                            "mtime": p.stat().st_mtime,
                            "arch": (cfg or {}).get("arch"),
                            "dataset": (cfg or {}).get("dataset"),
                        }
                    )
            self._send_json({"runs_root": str(self.runs_root), "runs": out})
            return

        if u.path == "/api/run_info":
            config = _read_json(run_dir / "config.json")
            # Try to parse speedup from log_buffer if this is the active run
            speedup_info = {}
            with Handler.proc_lock:
                search_buffer = list(Handler.log_buffer)
            
            for line in reversed(search_buffer):
                if "BENCHMARK_SPEEDUP" in line:
                    speedup_info["raw"] = line.strip()
                    break

            self._send_json(
                {
                    "run_dir": str(run_dir),
                    "config": config,
                    "device": _read_json(run_dir / "device.json"),
                    "has_profiling": (run_dir / "profiling").exists(),
                    "speedup": speedup_info
                }
            )
            return

        if u.path == "/api/csv":
            file_key = (qs.get("file") or [""])[0]
            tail = int((qs.get("tail") or ["2000"])[0])
            allowed = {
                "train_metrics": Path("profiling/train_metrics.csv"),
                "gpu_metrics": Path("profiling/gpu_metrics.csv"),
                "system_metrics": Path("profiling/system_metrics.csv"),
                "functions_events": Path("profiling/functions_events.csv"),
                "functions_summary": Path("profiling/functions_summary.csv"),
                "kernel_launches": Path("profiling/kernel_launches.csv"),
                "kernel_metrics": Path("profiling/kernel_metrics.csv"),
            }
            rel = allowed.get(file_key)
            if rel is None:
                self._send_json({"error": "unknown file key"}, code=400)
                return
            rows = _read_csv_rows(run_dir / rel, tail=tail)
            self._send_json({"run_dir": str(run_dir), "file": file_key, "rows": rows})
            return

        if u.path == "/api/proc_status":
            with Handler.proc_lock:
                p = Handler.proc
                running = p is not None and p.poll() is None
                pid = p.pid if p is not None else None
                status = Handler.last_status
                exit_code = Handler.last_exit_code
                # Update status if process just finished
                if p is not None and not running and status == "running":
                    exit_code = p.returncode
                    Handler.last_exit_code = exit_code
                    Handler.last_status = "completed" if exit_code == 0 else "error"
                    status = Handler.last_status
            self._send_json({"running": running, "pid": pid, "status": status, "exit_code": exit_code})
            return

        if u.path == "/" or u.path == "/index.html":
            p = self.dash_dir / "index.html"
        else:
            rel = u.path.lstrip("/")
            p = (self.dash_dir / rel).resolve()
            if not _commonpath_ok(self.dash_dir, p) or not p.exists() or not p.is_file():
                self.send_response(404)
                self.end_headers()
                return

        if p.suffix not in (".html", ".js", ".css", ".png", ".svg", ".ico", ".txt"):
            self.send_response(415)
            self.end_headers()
            return

        data = p.read_bytes()
        self.send_response(200)
        if p.suffix == ".js":
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
        elif p.suffix == ".css":
            self.send_header("Content-Type", "text/css; charset=utf-8")
        elif p.suffix == ".png":
            self.send_header("Content-Type", "image/png")
        elif p.suffix == ".svg":
            self.send_header("Content-Type", "image/svg+xml")
        elif p.suffix == ".ico":
            self.send_header("Content-Type", "image/x-icon")
        else:
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_logs(self, pipe):
        while True:
            line = pipe.readline()
            if not line:
                break
            print(f"Captured: {line.strip()}")
            with Handler.proc_lock:
                Handler.log_buffer.append(line)
                if len(Handler.log_buffer) > 5000:
                    Handler.log_buffer.pop(0)

    def do_POST(self):
        print(f"POST {self.path}")
        u = urlparse(self.path)
        if u.path not in ("/api/start", "/api/stop"):
            self.send_response(404)
            self.end_headers()
            return

        n = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(n) if n > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8", errors="ignore") or "{}")
        except Exception:
            payload = {}

        if u.path == "/api/stop":
            with Handler.proc_lock:
                p = Handler.proc
                Handler.proc = None
                Handler.last_status = "stopped"
                Handler.last_exit_code = None
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
            self._send_json({"ok": True})
            return

        # /api/start
        with Handler.proc_lock:
            if Handler.proc is not None and Handler.proc.poll() is None:
                self._send_json({"error": "process already running"}, code=409)
                return
            Handler.log_buffer.clear()
            Handler.last_status = "running"
            Handler.last_exit_code = None

        exe = payload.get("exe") or ""
        if not exe:
            # Default to build output.
            build_dir = Path(__file__).parent.parent / "build"
            exe_name = "gpu_trainer.exe" if os.name == "nt" else "gpu_trainer"
            exe_path = build_dir / exe_name
            # Fallback for WSL builds on Windows (no .exe)
            if not exe_path.exists() and os.name == "nt":
                alt_path = build_dir / "gpu_trainer"
                if alt_path.exists():
                    exe_path = alt_path
            exe = str(exe_path.resolve())

        arch = payload.get("arch") or "lenet"
        dataset = payload.get("dataset") or "mnist"
        epochs = int(payload.get("epochs") or 2)
        batch = int(payload.get("batch") or payload.get("batch_size") or 64)
        lr = float(payload.get("lr") or 0.01)
        seed = int(payload.get("seed") or 1337)
        data_dir = payload.get("data_dir") or "data"
        run_name = payload.get("run_name") or ""
        save_every = int(payload.get("save_every") or 0)
        resume = payload.get("resume") or payload.get("resume_from") or ""
        profile_interval_ms = int(payload.get("profile_interval_ms") or 200)
        
        # New Params
        weight_decay = float(payload.get("weight_decay") or 0.0)
        log_every = int(payload.get("log_every") or 50)
        eval_every = int(payload.get("eval_every") or 200)
        max_steps = int(payload.get("max_steps") or 0)
        norm_log_multiplier = int(payload.get("norm_log_multiplier") or 5)
        benchmark_compare = bool(payload.get("benchmark_compare") or False)
        benchmark_steps = int(payload.get("benchmark_steps") or 200)

        enable_h2d_pipeline = bool(payload.get("enable_h2d_pipeline", True))
        enable_log_sync_optimizations = bool(payload.get("enable_log_sync_optimizations", True))
        enable_async_checkpoint = bool(payload.get("enable_async_checkpoint", True))
        enable_cuda_graph_sgd = bool(payload.get("enable_cuda_graph_sgd", False))
        enable_async_eval = bool(payload.get("enable_async_eval", False))

        if "shuffle_train" in payload:
            no_shuffle = not bool(payload.get("shuffle_train"))
        else:
            no_shuffle = bool(payload.get("no_shuffle") or False)

        def _log_to_terminal(msg):
            with Handler.proc_lock:
                Handler.log_buffer.append(msg)
                if len(Handler.log_buffer) > 5000:
                    Handler.log_buffer.pop(0)
            print(msg.rstrip())

        try:
            _ensure_dataset_ready(dataset, data_dir, log_fn=_log_to_terminal)
        except Exception as e:
            _log_to_terminal(f"[dashboard] ERROR: dataset download failed: {e}\n")
            Handler.last_status = "error"
            self._send_json({"error": f"dataset download failed: {e}"}, code=500)
            return

        # Ensure runs land under the server's runs_root for monitoring.
        out_dir = str(self.runs_root)

        print(f"Launching trainer: {exe}")
        print(f"Run Name: {run_name}")
        args = [
            exe,
            "--arch", arch,
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--lr", str(lr),
            "--weight-decay", str(weight_decay),
            "--seed", str(seed),
            "--data-dir", data_dir,
            "--out-dir", out_dir,
            "--run-name", run_name,
            "--save-every", str(save_every),
            "--log-every", str(log_every),
            "--eval-every", str(eval_every),
            "--max-steps", str(max_steps),
            "--norm-log-mult", str(norm_log_multiplier),
            "--benchmark-steps", str(benchmark_steps),
            "--profile-interval-ms", str(profile_interval_ms),
        ]
        if resume:
            args += ["--resume", resume]
        if no_shuffle:
            args.append("--no-shuffle")
        if benchmark_compare:
            args.append("--benchmark-compare")
        
        # Performance flags (need careful handling of defaults in executable)
        if not enable_h2d_pipeline: args.append("--no-h2d-pipeline") 
        # Note: Need to check if executable supports --no-xxx style or if it just takes boolean.
        # Based on args.cu logic (guessed), I'll use standard flags if they exist.
        if not enable_log_sync_optimizations: args.append("--no-log-sync-opt")
        if not enable_async_checkpoint: args.append("--no-async-checkpoint")
        if enable_cuda_graph_sgd: args.append("--cuda-graph-sgd")
        if enable_async_eval: args.append("--async-eval")

        latest_txt = self.runs_root / "latest.txt"
        prev_latest = latest_txt.read_text(encoding="utf-8", errors="ignore").strip() if latest_txt.exists() else ""

        try:
            p = subprocess.Popen(
                args, 
                cwd=str(Path(__file__).parent.parent), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except Exception as e:
            self._send_json({"error": str(e), "args": args}, code=500)
            return

        with Handler.proc_lock:
            Handler.proc = p
            Handler.log_buffer.clear() # Clear logs for new run
        
        threading.Thread(target=self._read_logs, args=(p.stdout,), daemon=True).start()

        # Try to resolve the new run dir by watching latest.txt for a few seconds.
        new_dir = ""
        t0 = time.time()
        while time.time() - t0 < 8.0:
            if p.poll() is not None:
                break
            if latest_txt.exists():
                cur = latest_txt.read_text(encoding="utf-8", errors="ignore").strip()
                if cur and cur != prev_latest and Path(cur).exists():
                    new_dir = Path(cur).name
                    break
            time.sleep(0.1)

        poll_res = p.poll()
        if poll_res is not None:
            # Process died immediately. Get last few logs if possible.
            with Handler.proc_lock:
                err_msg = "".join(Handler.log_buffer[-10:])
                Handler.last_status = "error"
                Handler.last_exit_code = poll_res
            response = {
                "ok": False, 
                "error": f"Trainer exited immediately with code {poll_res}. Check terminal output.",
                "details": err_msg,
                "args": args
            }
            self._send_json(response, code=400)
            return

        if new_dir:
            try:
                self.default_run_dir = (self.runs_root / new_dir).resolve()
            except Exception:
                pass

        self._send_json({"ok": True, "pid": p.pid, "run_dir": new_dir})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/latest", help="Default run directory (e.g. runs/.... or runs/latest)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.name == "latest":
        latest_txt = run_dir.parent / "latest.txt"
        if latest_txt.exists():
            run_dir = Path(latest_txt.read_text(encoding="utf-8").strip())
    run_dir = run_dir.resolve()
    runs_root = run_dir.parent.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    dash_dir = (Path(__file__).parent.parent / "dashboard").resolve()
    if not dash_dir.exists():
        raise SystemExit(f"Missing dashboard directory: {dash_dir}")

    Handler.runs_root = runs_root
    Handler.default_run_dir = run_dir
    Handler.dash_dir = dash_dir

    host = os.environ.get("HOST", "127.0.0.1")
    httpd = HTTPServer((host, args.port), Handler)
    print(f"Dashboard: http://{host}:{args.port}  (runs_root={runs_root}, default_run={run_dir.name})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
