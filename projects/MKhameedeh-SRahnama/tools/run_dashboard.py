import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON config: {path} ({e})")


def _post_json(url: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"POST {url} failed: HTTP {e.code} {msg}".strip())
    except Exception as e:
        raise SystemExit(f"POST {url} failed: {e}")
    try:
        return json.loads(data.decode("utf-8", errors="ignore") or "{}")
    except Exception:
        return {}


def _get_json(url: str):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="ignore") or "{}")


def _wait_http_ready(base_url: str, timeout_s: float = 10.0):
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            _get_json(f"{base_url}/api/proc_status")
            return
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    raise SystemExit(f"Dashboard server did not become ready at {base_url} ({last_err})")


def _coalesce(cfg: dict, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None and cfg[k] != "":
            return cfg[k]
    return default


def main() -> int:
    repo_root = Path(__file__).parent.parent.resolve()

    ap = argparse.ArgumentParser(description="Launch the dashboard UI and optionally auto-start training.")
    ap.add_argument("--config", default="", help="Optional path to a JSON config file.")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-open", action="store_true", help="Do not open the browser.")
    ap.add_argument("--start", action="store_true", help="Auto-start training after the server starts.")
    ap.add_argument("--no-start", action="store_true", help="Do not auto-start training (default).")
    ap.add_argument("--exe", default="", help="Optional override for the trainer executable path.")
    ap.add_argument("--server-only", action="store_true", help="Alias for --no-start.")
    args = ap.parse_args()
    if args.server_only:
        args.no_start = True
    auto_start = args.start and not args.no_start

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        cfg = _read_json(cfg_path)
    else:
        cfg = {
            "arch": "lenet",
            "dataset": "mnist",
            "epochs": 2,
            "batch": 64,
            "lr": 0.01,
            "seed": 1337,
            "data_dir": "data",
            "out_dir": "runs",
            "run_name": "lenet_mnist",
            "save_every": 200,
            "profile_interval_ms": 200,
            "no_shuffle": False,
        }

    out_dir = Path(_coalesce(cfg, "out_dir", default="runs"))
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"

    server_env = os.environ.copy()
    server_env["HOST"] = args.host

    server_cmd = [sys.executable, str((repo_root / "tools" / "serve_dashboard.py").resolve()), "--run-dir", str(out_dir / "latest"), "--port", str(args.port)]
    server_proc = subprocess.Popen(server_cmd, cwd=str(repo_root), env=server_env)

    try:
        _wait_http_ready(base_url, timeout_s=10.0)

        if not args.no_open:
            try:
                webbrowser.open(base_url, new=1)
            except Exception:
                pass

        if auto_start:
            payload = {
                "exe": args.exe or _coalesce(cfg, "exe", default=""),
                "arch": _coalesce(cfg, "arch", default="lenet"),
                "dataset": _coalesce(cfg, "dataset", default="mnist"),
                "epochs": int(_coalesce(cfg, "epochs", default=2)),
                "batch": int(_coalesce(cfg, "batch", "batch_size", default=64)),
                "lr": float(_coalesce(cfg, "lr", default=0.01)),
                "seed": int(_coalesce(cfg, "seed", default=1337)),
                "data_dir": _coalesce(cfg, "data_dir", default="data"),
                "run_name": _coalesce(cfg, "run_name", default=""),
                "save_every": int(_coalesce(cfg, "save_every", default=0)),
                "resume": _coalesce(cfg, "resume", default=""),
                "profile_interval_ms": int(_coalesce(cfg, "profile_interval_ms", default=200)),
                "no_shuffle": bool(_coalesce(cfg, "no_shuffle", default=False)),
            }
            r = _post_json(f"{base_url}/api/start", payload)
            run_dir = r.get("run_dir") or ""
            if run_dir:
                print(f"Started run: {out_dir / run_dir}")
            else:
                print("Started run.")

        print(f"Dashboard: {base_url}")
        return server_proc.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            _post_json(f"{base_url}/api/stop", {})
        except Exception:
            pass
        try:
            if server_proc.poll() is None:
                server_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
