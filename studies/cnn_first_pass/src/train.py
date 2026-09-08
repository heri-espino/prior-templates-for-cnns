from __future__ import annotations
import argparse, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.io import save_json
from src.utils.viz import save_kernels_grid
from src.data.shapes import ShapesDataset, default_splits
from src.templates.primitives import TemplateSpec, make_template_bank, template_names
from src.models.cnn import TinyCNN, ModelConfig
from src.interpret.alignment import compute_alignment
from src.interpret.ablation import class_conditional_ablation
from src.interpret.patching import single_channel_patching, topk_patching
from src.interpret.drift import compute_drift

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())

@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    accs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        accs.append(accuracy_from_logits(logits, y))
    return float(np.mean(accs)) if accs else 0.0

def train_one(model, train_loader, val_loader, device: torch.device, epochs: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, acc_sum, n_batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            acc_sum += accuracy_from_logits(logits.detach(), y)
            n_batches += 1
            pbar.set_postfix(loss=loss_sum/max(1,n_batches), acc=acc_sum/max(1,n_batches))

        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": loss_sum/max(1,n_batches), "train_acc": acc_sum/max(1,n_batches), "val_acc": val_acc})
    return history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=str, default="random", choices=["random","template_init","frozen_templates"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--kernel_size", type=int, default=9)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--template_channels", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_size", type=int, default=8000)
    ap.add_argument("--val_size", type=int, default=1000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--max_probe_items", type=int, default=2000)
    ap.add_argument("--num_pairs", type=int, default=1000)
    ap.add_argument("--run_probes", type=int, default=1, help="1=run ablation/patching probes; 0=skip (faster)")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    splits = default_splits(args.seed, args.train_size, args.val_size, args.test_size)
    mk_ds = lambda key: ShapesDataset(splits[key], image_size=args.image_size)
    ds = {k: mk_ds(k) for k in splits.keys()}
    mk_dl = lambda dset, shuf: DataLoader(dset, batch_size=args.batch_size, shuffle=shuf, num_workers=0)

    loaders = {
        "train": mk_dl(ds["train"], True),
        "val": mk_dl(ds["val"], False),
        "test": mk_dl(ds["test"], False),
        "ood_rot": mk_dl(ds["ood_rot"], False),
        "ood_thick": mk_dl(ds["ood_thick"], False),
        "ood_occ": mk_dl(ds["ood_occ"], False),
    }

    t_spec = TemplateSpec(k=args.kernel_size)
    templates = make_template_bank(t_spec)
    names = template_names(t_spec)

    mcfg = ModelConfig(
        image_size=args.image_size,
        kernel_size=args.kernel_size,
        channels=args.channels,
        regime=args.regime,  # type: ignore
        template_channels=args.template_channels,
    )
    model = TinyCNN(mcfg, templates=templates if args.regime != "random" else None).to(device)

    # store init kernels for drift
    W_init = model.conv.weight.detach().clone()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) if args.outdir else Path("runs")/f"{ts}_{args.regime}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir/"config.json", {
        "args": vars(args),
        "model": asdict(mcfg),
        "template_spec": asdict(t_spec),
        "template_names": names,
        "split_names": list(splits.keys()),
    })

    history = train_one(model, loaders["train"], loaders["val"], device=device, epochs=args.epochs, lr=args.lr)
    save_json(run_dir/"train_log.json", {"rows": history})

    id_acc = evaluate(model, loaders["test"], device)
    ood_rot_acc = evaluate(model, loaders["ood_rot"], device)
    ood_thick_acc = evaluate(model, loaders["ood_thick"], device)
    ood_occ_acc = evaluate(model, loaders["ood_occ"], device)
    # MI probes
    # Alignment is cheap; compute it even when run_probes=0 so we can compare regimes.
    align = compute_alignment(model.conv.weight, templates)

    cond_abl = None
    patch1 = None
    patchk = None
    if args.run_probes == 1:
        cond_abl = class_conditional_ablation(
            model, loaders["test"], device,
            num_classes=mcfg.num_classes,
            max_items=args.max_probe_items,
        )
        patch1 = single_channel_patching(
            model, loaders["test"], device,
            num_classes=mcfg.num_classes,
            max_items=args.max_probe_items,
            num_pairs=args.num_pairs,
            seed=args.seed,
        )
        patchk = topk_patching(
            model, loaders["test"], device,
            delta_cond=cond_abl.delta_cond,
            num_classes=mcfg.num_classes,
            max_items=args.max_probe_items,
            num_pairs=args.num_pairs,
            seed=args.seed,
        )

    # Drift is always defined (needed even in low-data sweeps with run_probes=0)
    drift = compute_drift(W_init, model.conv.weight)

    # Save kernels + weights (always)
    save_kernels_grid(
        model.conv.weight.detach().cpu().numpy(),
        run_dir/"kernels_final.png",
        title=f"{args.regime} kernels (final)",
    )
    np.save(run_dir/"conv_W_init.npy", W_init.detach().cpu().numpy())
    np.save(run_dir/"conv_W_final.npy", model.conv.weight.detach().cpu().numpy())

    metrics = {
        "id_acc": id_acc,
        "ood_rot_acc": ood_rot_acc,
        "ood_thick_acc": ood_thick_acc,
        "ood_occ_acc": ood_occ_acc,
        "alignment": align.summary,
        "conditional_ablation": (cond_abl.summary if cond_abl is not None else {"mean_specificity": float("nan")}),
        "patching_single": (patch1.summary if patch1 is not None else {"mean_success": float("nan"), "max_success": float("nan"), "mean_shift": float("nan")}),
        "patching_topk": (patchk.summary if patchk is not None else {"success@maxk": float("nan"), "mean_success": float("nan"), "mean_shift": float("nan"), "num_pairs": 0}),
        "topk_ks": (patchk.ks if patchk is not None else []),
        "run_probes": int(args.run_probes),
    }
    save_json(run_dir/"metrics.json", metrics)
    save_json(run_dir/"drift.json", drift.summary)

    # raw arrays
    np.save(run_dir/"alignment_matrix.npy", align.A)
    np.save(run_dir/"alignment_best_template.npy", align.best_template)
    np.save(run_dir/"alignment_best_score.npy", align.best_score)

    if cond_abl is not None:
        np.save(run_dir/"delta_cond.npy", cond_abl.delta_cond)
        np.save(run_dir/"specificity.npy", cond_abl.specificity)
        np.save(run_dir/"mean_z_by_class.npy", cond_abl.mean_z_by_class)
    else:
        np.save(run_dir/"delta_cond.npy", np.zeros((0,), dtype=np.float32))
        np.save(run_dir/"specificity.npy", np.zeros((0,), dtype=np.float32))
        np.save(run_dir/"mean_z_by_class.npy", np.zeros((0,), dtype=np.float32))

    if patch1 is not None:
        np.save(run_dir/"patch_single_success.npy", patch1.success_rate)
        np.save(run_dir/"patch_single_shift.npy", patch1.mean_margin_shift)
    else:
        np.save(run_dir/"patch_single_success.npy", np.zeros((0,), dtype=np.float32))
        np.save(run_dir/"patch_single_shift.npy", np.zeros((0,), dtype=np.float32))

    if patchk is not None:
        np.save(run_dir/"patch_topk_success.npy", patchk.success_rate)
        np.save(run_dir/"patch_topk_shift.npy", patchk.mean_margin_shift)
    else:
        np.save(run_dir/"patch_topk_success.npy", np.zeros((0,), dtype=np.float32))
        np.save(run_dir/"patch_topk_shift.npy", np.zeros((0,), dtype=np.float32))

    np.save(run_dir/"drift_cosine.npy", drift.cosine)
    np.save(run_dir/"drift_l2.npy", drift.l2)

    torch.save(model.state_dict(), run_dir/"model.pt")

    print(json.dumps({"run_dir": str(run_dir), **metrics, "drift": drift.summary}, indent=2))
if __name__ == "__main__":
    main()
