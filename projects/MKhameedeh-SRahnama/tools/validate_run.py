import argparse
import json
from pathlib import Path


def load_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.name == "latest":
        latest_txt = run_dir.parent / "latest.txt"
        run_dir = Path(latest_txt.read_text(encoding="utf-8").strip())

    required = ["config.json", "device.json", "train.log", "metrics.jsonl"]
    missing = [f for f in required if not (run_dir / f).exists()]
    if missing:
        raise SystemExit(f"Missing files in {run_dir}: {missing}")

    points = load_jsonl(run_dir / "metrics.jsonl")
    if not points:
        raise SystemExit("No metrics points found")

    # Heuristic checks.
    any_loss = any("loss" in p for p in points)
    any_acc = any("acc" in p for p in points)
    if not any_loss:
        raise SystemExit("metrics.jsonl has no 'loss' entries")
    if not any_acc:
        raise SystemExit("metrics.jsonl has no 'acc' entries")

    print(f"OK: {run_dir} ({len(points)} metric points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

