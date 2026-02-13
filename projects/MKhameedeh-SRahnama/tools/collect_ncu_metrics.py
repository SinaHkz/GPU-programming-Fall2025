import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON config: {path} ({e})")


def _coalesce(cfg: dict, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None and cfg[k] != "":
            return cfg[k]
    return default


def _parse_metric_value(raw: str) -> float:
    s = (raw or "").strip().replace(",", "")
    if not s:
        return float("nan")
    if s.endswith("%"):
        try:
            return float(s[:-1].strip())
        except Exception:
            return float("nan")
    for tok in s.split():
        try:
            return float(tok)
        except Exception:
            continue
    try:
        return float(s)
    except Exception:
        return float("nan")


def _metric_to_col(metric: str) -> str:
    if metric == "achieved_occupancy":
        return "achieved_occupancy_pct"
    return metric.replace(".", "_").replace(" ", "_")


def _parse_ncu_csv(text: str, metrics: set[str]):
    rows = csv.reader(io.StringIO(text))
    header = None
    idx_kernel = idx_metric = idx_value = None
    data: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        if not row:
            continue
        first = row[0].strip()
        if first.startswith("==") or first.startswith("#"):
            continue
        if "Kernel Name" in row and "Metric Name" in row and "Metric Value" in row:
            header = row
            idx_kernel = header.index("Kernel Name")
            idx_metric = header.index("Metric Name")
            idx_value = header.index("Metric Value")
            continue
        if header is None or idx_kernel is None or idx_metric is None or idx_value is None:
            continue
        if len(row) <= max(idx_kernel, idx_metric, idx_value):
            continue
        kernel = row[idx_kernel].strip()
        metric = row[idx_metric].strip()
        if not kernel or metric not in metrics:
            continue
        val = _parse_metric_value(row[idx_value])
        if val != val:  # NaN
            continue
        data.setdefault(kernel, {}).setdefault(metric, []).append(val)
    return data


def _write_kernel_metrics(run_dir: Path, data: dict[str, dict[str, list[float]]], metrics: list[str]):
    profiling = run_dir / "profiling"
    profiling.mkdir(parents=True, exist_ok=True)
    out_path = profiling / "kernel_metrics.csv"

    cols = ["name"] + [_metric_to_col(m) for m in metrics]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, m in sorted(data.items()):
            row = [name]
            for metric in metrics:
                vals = m.get(metric, [])
                if not vals:
                    row.append("")
                    continue
                avg = sum(vals) / float(len(vals))
                if metric == "achieved_occupancy" and avg <= 1.0:
                    avg *= 100.0
                row.append(f"{avg:.6f}")
            w.writerow(row)
    return out_path


def main() -> int:
    repo_root = Path(__file__).parent.parent.resolve()

    ap = argparse.ArgumentParser(
        description="Run Nsight Compute (ncu) and export kernel metrics to runs/<run>/profiling/kernel_metrics.csv"
    )
    ap.add_argument("--config", help="Path to JSON config. Runs ncu using this config.")
    ap.add_argument("--exe", default="", help="Optional override for trainer executable.")
    ap.add_argument("--ncu", default="ncu", help="Path to ncu (Nsight Compute CLI).")
    ap.add_argument("--metrics", default="achieved_occupancy", help="Comma-separated NCU metric names.")
    ap.add_argument("--ncu-csv", help="Parse an existing ncu --csv output instead of running ncu.")
    ap.add_argument("--run-dir", help="Run dir to write kernel_metrics.csv when using --ncu-csv.")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")

    if args.ncu_csv:
        if not args.run_dir:
            raise SystemExit("--run-dir is required with --ncu-csv")
        run_dir = Path(args.run_dir)
        if run_dir.name == "latest":
            latest_txt = run_dir.parent / "latest.txt"
            if latest_txt.exists():
                run_dir = Path(latest_txt.read_text(encoding="utf-8").strip())
        text = Path(args.ncu_csv).read_text(encoding="utf-8", errors="ignore")
        data = _parse_ncu_csv(text, set(metrics))
        if not data:
            raise SystemExit("No matching kernel metrics found in ncu CSV.")
        out_path = _write_kernel_metrics(run_dir, data, metrics)
        print(f"Wrote {out_path}")
        return 0

    if not args.config:
        raise SystemExit("Provide --config or --ncu-csv.")

    if shutil.which(args.ncu) is None:
        raise SystemExit(f"ncu not found in PATH: {args.ncu}")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg = _read_json(cfg_path)

    exe = args.exe or _coalesce(cfg, "exe", default="")
    if not exe:
        exe = str((repo_root / "build" / ("gpu_trainer.exe" if sys.platform.startswith("win") else "gpu_trainer")).resolve())

    out_dir = Path(_coalesce(cfg, "out_dir", default="runs"))
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_txt = out_dir / "latest.txt"
    prev_latest = latest_txt.read_text(encoding="utf-8", errors="ignore").strip() if latest_txt.exists() else ""

    train_args = [
        exe,
        "--arch",
        _coalesce(cfg, "arch", default="lenet"),
        "--dataset",
        _coalesce(cfg, "dataset", default="mnist"),
        "--epochs",
        str(int(_coalesce(cfg, "epochs", default=2))),
        "--batch",
        str(int(_coalesce(cfg, "batch", "batch_size", default=64))),
        "--lr",
        str(float(_coalesce(cfg, "lr", default=0.01))),
        "--seed",
        str(int(_coalesce(cfg, "seed", default=1337))),
        "--data-dir",
        _coalesce(cfg, "data_dir", default="data"),
        "--out-dir",
        str(out_dir),
        "--run-name",
        _coalesce(cfg, "run_name", default=""),
        "--save-every",
        str(int(_coalesce(cfg, "save_every", default=0))),
        "--profile-interval-ms",
        str(int(_coalesce(cfg, "profile_interval_ms", default=200))),
    ]
    resume = _coalesce(cfg, "resume", "resume_from", default="")
    if resume:
        train_args += ["--resume", resume]
    if _coalesce(cfg, "shuffle_train", default=True) is False or _coalesce(cfg, "no_shuffle", default=False):
        train_args.append("--no-shuffle")

    cmd = [args.ncu, "--target-processes", "all", "--metrics", ",".join(metrics), "--csv"] + train_args
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "ncu failed")

    text = proc.stdout or ""
    data = _parse_ncu_csv(text, set(metrics))
    if not data:
        raise SystemExit("No matching kernel metrics found in ncu output.")

    # Resolve run dir via latest.txt after run.
    new_dir = ""
    t0 = time.time()
    while time.time() - t0 < 5.0:
        if latest_txt.exists():
            cur = latest_txt.read_text(encoding="utf-8", errors="ignore").strip()
            if cur and cur != prev_latest and Path(cur).exists():
                new_dir = cur
                break
        time.sleep(0.1)
    if not new_dir and latest_txt.exists():
        cur = latest_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if cur and Path(cur).exists():
            new_dir = cur

    if not new_dir:
        raise SystemExit("Could not resolve run directory from latest.txt.")

    run_dir = Path(new_dir)
    out_path = _write_kernel_metrics(run_dir, data, metrics)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
