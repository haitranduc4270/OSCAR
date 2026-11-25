"""TokenOSCAR gated-fusion summary for thesis Figure 5.4 (BRCA validation folds).

Same protocol as biomarker_gate_summary.py, adapted to TokenOSCAR's token-wise gates:
each gate is (B, T, M) = (256, 32) instead of a 128-dim vector, so per sample we take
the mean over all gate entries and the fraction of entries above 0.5. Folds use the
retrained weights from token_oscar_retrain_save.py.

Usage:
  .venv/bin/python token_oscar_gate_fig.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch
import yaml

import biomarker_gate_summary as gs
from token_oscar_biomarker_figs import CKPT_DIR, CONFIG, ASSETS_DIR, load_token_oscar
from visualize_attention import build_datamodule
from train import set_seed


def collect_gates(model, dataloader, device) -> list[dict]:
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        model.gate_mrna.gate.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("mRNA", out.detach())
        ),
        model.gate_prot.gate.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("Proteomics", out.detach())
        ),
    ]
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            batch_omics = {k: v.to(device) for k, v in batch.items() if k != "label"}
            captured.clear()
            model({**batch_omics, "label": labels})
            labels_np = labels.cpu().numpy()
            for omic, gate in captured.items():
                g = gate.reshape(gate.shape[0], -1)  # (B, T*M)
                mean_g = g.mean(dim=1).cpu().numpy()
                frac_g = (g > 0.5).float().mean(dim=1).cpu().numpy()
                for i in range(len(labels_np)):
                    rows.append(
                        {
                            "class_idx": int(labels_np[i]),
                            "omic": omic,
                            "mean_gate": float(mean_g[i]),
                            "frac_gate_gt_0.5": float(frac_g[i]),
                        }
                    )
    for h in hooks:
        h.remove()
    return rows


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(yaml.safe_load(open(CONFIG)))
    cfg["train"]["num_workers"] = 0
    set_seed(cfg["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audit = pd.read_csv(ASSETS_DIR / "label_audit.csv")
    names = {int(r.class_idx): str(r.paper_name) for r in audit.itertuples()}

    rows_all = []
    for fold_idx in range(5):
        fold_num = fold_idx + 1
        ckpt = CKPT_DIR / f"fold_{fold_num}.pt"
        cfg["data"]["current_fold"] = fold_idx
        dm = build_datamodule(cfg)
        model = load_token_oscar(cfg, dm, ckpt, device)
        rows = collect_gates(model, dm.val_dataloader(), device)
        for r in rows:
            r["fold"] = fold_num
            r["class_name"] = names.get(r["class_idx"], str(r["class_idx"]))
        rows_all.extend(rows)
        print(f"fold {fold_num}: {len(rows)} gate rows", flush=True)

    samples = pd.DataFrame(rows_all)
    samples.to_csv(ASSETS_DIR / "token_gate_samples.csv", index=False)
    fold_means = samples.groupby(["fold", "class_name", "omic"], as_index=False).agg(
        mean_of_mean_gate=("mean_gate", "mean"),
        mean_frac_gt_0_5=("frac_gate_gt_0.5", "mean"),
    )
    summary = fold_means.groupby(["class_name", "omic"], as_index=False).agg(
        mean_of_mean_gate=("mean_of_mean_gate", "mean"),
        std_mean_gate=("mean_of_mean_gate", "std"),
        mean_frac_gt_0_5=("mean_frac_gt_0_5", "mean"),
        std_frac_gt_0_5=("mean_frac_gt_0_5", "std"),
        n_folds=("fold", "nunique"),
    )
    summary.to_csv(ASSETS_DIR / "token_gate_summary.csv", index=False)
    print(summary.to_string(index=False))
    gs.plot_gates(summary, ASSETS_DIR / "token_gate_by_subtype.png", ylim=(0.45, 0.80))
    print(f"Wrote {ASSETS_DIR / 'token_gate_by_subtype.png'}")


if __name__ == "__main__":
    main()
