from __future__ import annotations
import subprocess
from itertools import product

def main():
    regimes = ["random", "template_init", "frozen_templates"]
    seeds = [0, 1, 2, 3, 4]
    channels_list = [8, 16]
    for regime, seed, ch in product(regimes, seeds, channels_list):
        cmd = [
            "python", "-m", "src.train",
            "--regime", regime,
            "--seed", str(seed),
            "--channels", str(ch),
            "--template_channels", str(min(ch, 16)),
            "--epochs", "5",
            "--num_pairs", "1500",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
