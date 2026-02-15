import argparse
import gzip
import os
import shutil
import urllib.request
from pathlib import Path


MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def gunzip(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (will create mnist/ under it)")
    args = ap.parse_args()

    root = Path(args.out) / "mnist"
    root.mkdir(parents=True, exist_ok=True)

    for fn in MNIST_FILES:
        gz_path = root / fn
        raw_path = root / fn.replace(".gz", "")
        if raw_path.exists():
            print(f"[skip] {raw_path}")
            continue

        ok = False
        for base in MIRRORS:
            url = base + fn
            try:
                print(f"[dl] {url}")
                download(url, gz_path)
                ok = True
                break
            except Exception as e:
                print(f"[warn] failed {url}: {e}")

        if not ok:
            raise SystemExit(f"Failed downloading {fn} from all mirrors")

        print(f"[gunzip] {gz_path} -> {raw_path}")
        gunzip(gz_path, raw_path)
        try:
            os.remove(gz_path)
        except OSError:
            pass

    print(f"MNIST ready in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

