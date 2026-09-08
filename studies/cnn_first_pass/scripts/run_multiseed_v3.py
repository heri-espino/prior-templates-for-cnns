from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--regimes", type=str, nargs="+", default=["random","template_init","frozen_templates"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--train_size", type=int, default=8000)
    ap.add_argument("--val_size", type=int, default=1000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--num_pairs", type=int, default=1500)
    ap.add_argument("--max_probe_items", type=int, default=2000)
    ap.add_argument("--runs_root", type=str, default="runs_v3")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    for regime in args.regimes:
        for seed in args.seeds:
            outdir = runs_root / regime / f"seed{seed:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python","-m","src.train",
                "--regime", regime,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--channels", str(args.channels),
                "--template_channels", str(min(args.channels, 16)),
                "--train_size", str(args.train_size),
                "--val_size", str(args.val_size),
                "--test_size", str(args.test_size),
                "--num_pairs", str(args.num_pairs),
                "--max_probe_items", str(args.max_probe_items),
                "--outdir", str(outdir),
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
