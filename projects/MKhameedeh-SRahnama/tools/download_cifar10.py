import argparse
import tarfile
import urllib.request
from pathlib import Path


URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    base = path.resolve()
    for member in tar.getmembers():
        target = (path / member.name).resolve()
        if not str(target).startswith(str(base)):
            raise RuntimeError(f"Blocked path traversal in tar member: {member.name}")
    tar.extractall(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (will create cifar10/ under it)")
    args = ap.parse_args()

    root = Path(args.out) / "cifar10"
    root.mkdir(parents=True, exist_ok=True)
    tgt = root / "cifar-10-binary.tar.gz"

    extracted_dir = root / "cifar-10-batches-bin"
    if extracted_dir.exists():
        print(f"[skip] {extracted_dir}")
        return 0

    print(f"[dl] {URL}")
    with urllib.request.urlopen(URL) as r, open(tgt, "wb") as f:
        f.write(r.read())

    print(f"[extract] {tgt}")
    with tarfile.open(tgt, "r:gz") as tar:
        safe_extract(tar, root)

    print(f"CIFAR-10 ready in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
