"""Train the predeclared exhaustive template-prior robustness grid.

This is separate from the frozen 20-block prospective confirmation. It uses
blocks 5000--5049 by default and never chooses profiles/checkpoints from test
outcomes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "studies" / "cnn_release_experiment"))
import core  # noqa: E402


PROFILES = {
    "template_init": {"kind": "constant", "lambda": 0.0},
    "retention_0p1": {"kind": "constant", "lambda": 0.1},
    "retention_1": {"kind": "constant", "lambda": 1.0},
    "release_early": {"kind": "linear_release", "start": 5, "end": 40},
    "release_default": {"kind": "linear_release", "start": 10, "end": 80},
    "release_late": {"kind": "linear_release", "start": 40, "end": 160},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_checkpoint(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def retention_strength(profile: str, epoch: int) -> float:
    spec = PROFILES[profile]
    if spec["kind"] == "constant":
        return float(spec["lambda"])
    start, end = int(spec["start"]), int(spec["end"])
    return float(np.clip((end - epoch) / (end - start), 0.0, 1.0))


def load_data(root: Path, task: str, block: int):
    core.ROOT = root
    return core.data_for(task, block)


def train_one(job: dict) -> dict:
    root = Path(job["root"]).resolve()
    task = job["task"]
    arch = job["architecture"]
    block = int(job["block"])
    profile = job["profile"]
    epochs = int(job["epochs"])
    threads = int(job["threads_per_worker"])

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    core.ROOT = root

    out = root / "runs" / task / arch / f"block{block:04d}" / profile
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return {"status": "skip", "task": task, "architecture": arch, "block": block, "profile": profile}

    data = core.data_for(task, block)
    model, anchor, _ = core.setup(arch, "template_init", block, task)
    assert anchor is not None
    init = model.conv.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    shuffle = torch.Generator().manual_seed(core.seed(21, block, core.TASKS.index(task)))

    config = {
        "task": task,
        "architecture": arch,
        "block": block,
        "profile": profile,
        "profile_spec": PROFILES[profile],
        "epochs": epochs,
        "lr": 0.003,
        "batch_size": 128,
        "threads_per_worker": threads,
        "master_seed": core.MASTER,
        "data_hashes": {k: core.digest(v["x"]) for k, v in data.items()},
        "source_sha256": {
            "core.py": sha256(Path(core.__file__)),
            "train_grid.py": sha256(Path(__file__)),
            "PROTOCOL.md": sha256(Path(__file__).with_name("PROTOCOL.md")),
        },
    }
    atomic_json(out / "config.json", config)

    history = []
    first_epoch = 1
    resume = out / "latest.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        shuffle.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        history = saved["history"]
        first_epoch = int(saved["epoch"]) + 1
    else:
        np.save(out / "conv_initial.npy", init.numpy())
        atomic_checkpoint(out / "epoch_0000.pt", {"epoch": 0, "model": model.state_dict()})

    X = torch.from_numpy(data["train"]["x"])
    Y = torch.from_numpy(data["train"]["y"])

    for epoch in range(first_epoch, epochs + 1):
        start = time.perf_counter()
        model.train()
        order = torch.randperm(len(X), generator=shuffle)
        ce_sum = reg_sum = correct = grad_sum = 0.0
        batches = 0
        lam = retention_strength(profile, epoch)

        for st in range(0, len(X), 128):
            ix = order[st : st + 128]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X[ix])
            ce = F.cross_entropy(logits, Y[ix])
            reg = (model.conv.weight - anchor).square().sum() / anchor.square().sum()
            (ce + lam * reg).backward()
            grad_sum += float(model.conv.weight.grad.norm())
            batches += 1
            optimizer.step()
            ce_sum += float(ce.detach()) * len(ix)
            reg_sum += float(reg.detach()) * len(ix)
            correct += int((logits.detach().argmax(1) == Y[ix]).sum())

        model.eval()
        vl, vz, _, _ = core.collect(model, data["val"]["x"])
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": epoch * 4,
                "lambda_retention": lam,
                "train_ce": ce_sum / len(X),
                "train_acc": correct / len(X),
                "retention_penalty": reg_sum / len(X),
                "val_ce": float(F.cross_entropy(vl, torch.from_numpy(data["val"]["y"]))),
                "val_acc": float((vl.argmax(1).numpy() == data["val"]["y"]).mean()),
                "alignment": core.alignment(model.conv.weight.detach().numpy(), core.bank()),
                "kernel_norm": float(model.conv.weight.detach().norm()),
                "conv_gradient_norm": grad_sum / batches,
                "val_pooled_feature_norm": float(vz.norm(dim=1).mean()),
                "train_seconds": time.perf_counter() - start,
            }
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "shuffle_rng": shuffle.get_state(),
            "torch_rng": torch.get_rng_state(),
            "history": history,
        }
        atomic_checkpoint(resume, state)
        atomic_json(out / "history.json", history)

        if epoch == epochs:
            atomic_checkpoint(out / f"epoch_{epoch:04d}.pt", state)

    atomic_json(out / "complete.json", {"epochs": epochs, "status": "complete"})
    return {
        "status": "done",
        "task": task,
        "architecture": arch,
        "block": block,
        "profile": profile,
        "final_val_acc": history[-1]["val_acc"],
        "final_alignment": history[-1]["alignment"],
    }


def build_jobs(args, root: Path):
    return [
        {
            "root": str(root),
            "task": task,
            "architecture": arch,
            "block": block,
            "profile": profile,
            "epochs": args.epochs,
            "threads_per_worker": args.threads_per_worker,
        }
        for block in range(args.start_block, args.start_block + args.blocks)
        for task in args.tasks
        for arch in args.architectures
        for profile in args.profiles
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/exhaustive_robustness_001/training")
    p.add_argument("--start-block", type=int, default=5000)
    p.add_argument("--blocks", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--tasks", nargs="+", choices=core.TASKS, default=core.TASKS)
    p.add_argument("--architectures", nargs="+", choices=core.ARCHS, default=core.ARCHS)
    p.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=2)
    args = p.parse_args()

    if args.start_block < 5000:
        p.error("Exhaustive robustness blocks must start at 5000 or later.")
    if min(args.blocks, args.epochs, args.workers, args.threads_per_worker) < 1:
        p.error("blocks/epochs/workers/threads-per-worker must be positive.")

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    core.ROOT = root

    design = {
        "study": "cnn_exhaustive_robustness",
        "start_block": args.start_block,
        "blocks": args.blocks,
        "epochs": args.epochs,
        "tasks": args.tasks,
        "architectures": args.architectures,
        "profiles": args.profiles,
        "profile_specs": {k: PROFILES[k] for k in args.profiles},
        "lr": 0.003,
        "batch_size": 128,
        "threads_per_worker": args.threads_per_worker,
        "master_seed": core.MASTER,
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "core_sha256": sha256(Path(core.__file__)),
        "trainer_sha256": sha256(Path(__file__)),
    }
    design_path = root / "design.json"
    if design_path.exists():
        if json.loads(design_path.read_text()) != design:
            raise SystemExit("Existing training design differs. Use a new --output root.")
    else:
        atomic_json(design_path, design)

    atomic_json(
        root / "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
            "cpu_count": os.cpu_count(),
        },
    )

    # Generate every dataset before multiprocessing begins. This avoids races on
    # compressed NPZ/JSON creation while preserving the exact renderer streams.
    print("Pre-generating frozen datasets...", flush=True)
    for block in range(args.start_block, args.start_block + args.blocks):
        for task in args.tasks:
            core.data_for(task, block)

    jobs = build_jobs(args, root)
    print(f"Planned model jobs: {len(jobs)} | workers={args.workers} | threads/worker={args.threads_per_worker}", flush=True)

    if args.workers == 1:
        for i, job in enumerate(jobs, 1):
            r = train_one(job)
            print(f"[{i}/{len(jobs)}] {r}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        done = 0
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            future_to_job = {ex.submit(train_one, job): job for job in jobs}
            for fut in cf.as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    print("FAILED JOB:", job, flush=True)
                    raise RuntimeError(f"Training job failed: {job}") from exc
                done += 1
                print(f"[{done}/{len(jobs)}] {r}", flush=True)

    completed = list((root / "runs").glob("*/*/block*/*/complete.json"))
    if len(completed) != len(jobs):
        raise SystemExit(f"Incomplete grid: {len(completed)} / {len(jobs)} models complete.")
    atomic_json(root / "GRID_COMPLETE.json", {"planned": len(jobs), "complete": len(completed)})
    print("Training grid complete:", root, flush=True)


if __name__ == "__main__":
    main()
