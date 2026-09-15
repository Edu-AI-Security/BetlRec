from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    root = Path(__file__).resolve().parent
    python = sys.executable
    run([python, str(root / "generate_synthetic.py"), "--out", str(root / "data/synthetic_beltrec")])
    run([python, str(root / "train.py"), "--data", str(root / "data/synthetic_beltrec"), "--out", str(root / "checkpoints/beltrec.pt"), "--epochs", "3"])
    run([python, str(root / "evaluate.py"), "--data", str(root / "data/synthetic_beltrec"), "--checkpoint", str(root / "checkpoints/beltrec.pt")])


if __name__ == "__main__":
    main()
