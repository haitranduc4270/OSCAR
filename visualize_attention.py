import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import pandas as pd

from datamodule import BRCADataModule
from models.self_attn_fusion import SelfAttentionFusionModel
from train import extract_omics_config, set_seed


def build_datamodule(cfg):
    """Build BRCADataModule consistent with training config."""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    dm = BRCADataModule(
        omics_config=extract_omics_config(data_cfg),
        y_path=data_cfg["y_path"],
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 4),
        val_split=data_cfg.get("val_split", 0.2),
        seed=train_cfg.get("seed", 42),
        normalize=train_cfg.get("normalize", True),
        use_stratified_kfold=data_cfg.get("use_stratified_kfold", False),
        n_folds=data_cfg.get("n_folds", 5),
        current_fold=data_cfg.get("current_fold", 0),
    )
    dm.setup(stage="fit")
    return dm


def load_model_from_checkpoint(cfg, dm, checkpoint_path, device):
    """Instantiate SelfAttentionFusionModel and load weights from checkpoint."""
    model_cfg = cfg["model"]

    input_dims = (
        dm.feature_dims
        if model_cfg.get("input_dims") is None
        else model_cfg["input_dims"]
    )

    model = SelfAttentionFusionModel.load_from_checkpoint(
        checkpoint_path,
        input_dims=input_dims,
        num_classes=model_cfg["num_classes"],
        latent_dim=model_cfg.get("latent_dim", 128),
        encoder_hidden_dim=model_cfg.get("hidden_dim", 512),
        fusion_hidden=model_cfg.get("fusion_hidden", 256),
        dropout=model_cfg.get("dropout", 0.3),
        lr=float(cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 2),
        fusion_type=model_cfg.get("fusion_type", "attention"),
        use_cross_attention=model_cfg.get("use_cross_attention", False),
        cross_attn_hidden=model_cfg.get("cross_attn_hidden", 128),
        use_original_for_proteomics=model_cfg.get("use_original_for_proteomics", True),
        use_gated_fusion=model_cfg.get("use_gated_fusion", False),
        gated_fusion_hidden=model_cfg.get("gated_fusion_hidden", 128),
    )

    model.to(device)
    model.eval()
    return model


def compute_gradient_feature_importance(
    model,
    dataloader,
    device,
    num_classes,
    max_batches=None,
    use_true_label=True,
):
    """
    Compute gradient-based feature importance per omic and per class.

    For each batch, we compute gradients of the selected logit (true or
    predicted class) w.r.t. each omic input feature. We accumulate the
    mean absolute gradient per feature separately for each class.
    """
    # omic_name -> feature_dim (infer from first batch)
    omic_names = None

    # class_idx -> omic_name -> np.array(feature_dim,)
    grad_sums = defaultdict(dict)
    # class_idx -> count of samples
    class_counts = defaultdict(int)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move to device and enable gradients on omic inputs
        labels = batch["label"].to(device)
        batch_omics = {}
        for key, value in batch.items():
            if key == "label":
                continue
            x = value.to(device)
            x.requires_grad_(True)
            batch_omics[key] = x

        if omic_names is None:
            omic_names = sorted(batch_omics.keys())

        # Forward pass
        batch_for_model = {**batch_omics, "label": labels}
        logits = model(batch_for_model)  # [B, num_classes]

        if use_true_label:
            target_classes = labels
        else:
            target_classes = torch.argmax(logits, dim=1)

        # Select logit for target class for each sample
        selected_logits = logits.gather(1, target_classes.view(-1, 1)).squeeze(1)

        # Backward on sum of selected logits
        model.zero_grad()
        for x in batch_omics.values():
            if x.grad is not None:
                x.grad.zero_()

        selected_logits.sum().backward()

        # Accumulate absolute gradients per class
        labels_np = labels.detach().cpu().numpy()
        for omic in omic_names:
            grads = batch_omics[omic].grad.detach().cpu().numpy()  # [B, F]
            abs_grads = np.abs(grads)

            for c in range(num_classes):
                mask = labels_np == c
                if not np.any(mask):
                    continue
                grads_c = abs_grads[mask]  # [Nc, F]

                if omic not in grad_sums[c]:
                    grad_sums[c][omic] = grads_c.sum(axis=0)
                else:
                    grad_sums[c][omic] += grads_c.sum(axis=0)
                class_counts[c] += int(mask.sum())

    # Normalize by number of samples per class
    importance = {}
    for c in range(num_classes):
        if class_counts[c] == 0:
            continue
        importance[c] = {}
        for omic in omic_names:
            if omic not in grad_sums[c]:
                continue
            importance[c][omic] = grad_sums[c][omic] / float(class_counts[c])

    return importance, omic_names


def plot_topk_barplots_per_class(
    importance,
    omic_names,
    output_dir,
    top_k=10,
    class_name_map=None,
    feature_name_map=None,
):
    os.makedirs(output_dir, exist_ok=True)

    for c, omic_dict in importance.items():
        class_label = class_name_map[c] if class_name_map and c in class_name_map else f"class_{c}"
        print(f"Plotting class {class_label}")
        for omic in omic_names:
            if omic not in omic_dict:
                continue
            scores = omic_dict[omic]
            if scores.ndim != 1:
                scores = scores.reshape(-1)

            # Top-k indices
            k = min(top_k, len(scores))
            top_idx = np.argsort(-scores)[:k]
            top_scores = scores[top_idx]

            # Map feature indices to names when available
            feature_labels = []
            for i in top_idx:
                if feature_name_map and omic in feature_name_map:
                    names = feature_name_map[omic]
                    if 0 <= i < len(names):
                        feature_labels.append(names[i])
                        continue
                # Fallback: generic index label
                feature_labels.append(f"{omic}_feat_{int(i)}")

            plt.figure(figsize=(8, 4))
            plt.bar(range(k), top_scores)
            plt.xticks(range(k), feature_labels, rotation=60, ha="right", fontsize=8)
            plt.ylabel("Mean |∂logit/∂x|")
            plt.title(f"Top-{k} {omic} features for {class_label}")
            plt.tight_layout()

            fname = os.path.join(output_dir, f"grad_importance_{class_label}_{omic}.png")
            plt.savefig(fname, dpi=200)
            plt.close()


def plot_omic_heatmaps(
    importance,
    omic_names,
    output_dir,
    top_k=10,
    class_name_map=None,
    feature_name_map=None,
):
    """
    For each omic, build a heatmap combining all subtypes in one plot.

    - Rows: classes / subtypes
    - Cols: union of top-k features (per class) for that omic
    - Values: mean |∂logit/∂x| importance
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine sorted list of classes we actually have importance for
    class_ids = sorted(importance.keys())

    for omic in omic_names:
        # Collect per-class scores for this omic
        per_class_scores = {}
        for c in class_ids:
            omic_dict = importance.get(c, {})
            if omic in omic_dict:
                scores = omic_dict[omic]
                if scores.ndim != 1:
                    scores = scores.reshape(-1)
                per_class_scores[c] = scores

        if not per_class_scores:
            continue

        # Union of top-k indices across all classes for this omic
        union_idx = set()
        for c, scores in per_class_scores.items():
            k = min(top_k, len(scores))
            top_idx = np.argsort(-scores)[:k]
            union_idx.update(top_idx.tolist())

        if not union_idx:
            continue

        union_idx = sorted(union_idx)

        # Build matrix [n_classes, n_features_union]
        heatmap = np.zeros((len(class_ids), len(union_idx)), dtype=float)
        for row, c in enumerate(class_ids):
            scores = per_class_scores.get(c)
            if scores is None:
                continue
            heatmap[row] = scores[union_idx]

        # Build labels
        y_labels = [
            class_name_map[c] if class_name_map and c in class_name_map else f"class_{c}"
            for c in class_ids
        ]

        x_labels = []
        for idx in union_idx:
            label = None
            if feature_name_map and omic in feature_name_map:
                names = feature_name_map[omic]
                if 0 <= idx < len(names):
                    label = names[idx]
            if label is None:
                label = f"{omic}_feat_{int(idx)}"
            x_labels.append(label)

        plt.figure(figsize=(max(8, 0.4 * len(union_idx)), 1.2 * len(class_ids)))
        im = plt.imshow(heatmap, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Mean |∂logit/∂x|")
        plt.yticks(np.arange(len(class_ids)), y_labels)
        plt.xticks(np.arange(len(union_idx)), x_labels, rotation=60, ha="right", fontsize=8)
        plt.title(f"Gradient-based feature importance heatmap for {omic}")
        plt.tight_layout()

        fname = os.path.join(output_dir, f"grad_importance_heatmap_{omic}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_omic_grouped_barplots(
    importance,
    omic_names,
    output_dir,
    top_k=10,
    class_name_map=None,
    feature_name_map=None,
):
    """
    For each omic, create a single barplot figure that combines all subtypes.

    - X-axis: union of top-k features across all subtypes for that omic
    - Bars: grouped by class (different colors) at each feature
    - Y-axis: mean |∂logit/∂x|
    """
    os.makedirs(output_dir, exist_ok=True)

    class_ids = sorted(importance.keys())

    for omic in omic_names:
        per_class_scores = {}
        for c in class_ids:
            omic_dict = importance.get(c, {})
            if omic in omic_dict:
                scores = omic_dict[omic]
                if scores.ndim != 1:
                    scores = scores.reshape(-1)
                per_class_scores[c] = scores

        if not per_class_scores:
            continue

        # Union of top-k indices across classes
        union_idx = set()
        for c, scores in per_class_scores.items():
            k = min(top_k, len(scores))
            top_idx = np.argsort(-scores)[:k]
            union_idx.update(top_idx.tolist())

        if not union_idx:
            continue

        union_idx = sorted(union_idx)

        # Build feature labels for x-axis
        x_labels = []
        for idx in union_idx:
            label = None
            if feature_name_map and omic in feature_name_map:
                names = feature_name_map[omic]
                if 0 <= idx < len(names):
                    label = names[idx]
            if label is None:
                label = f"{omic}_feat_{int(idx)}"
            x_labels.append(label)

        x = np.arange(len(union_idx))
        n_classes = len(class_ids)
        width = 0.8 / max(1, n_classes)

        plt.figure(figsize=(max(10, 0.6 * len(union_idx)), 6))

        for i, c in enumerate(class_ids):
            scores = per_class_scores.get(c)
            if scores is None:
                continue
            vals = scores[union_idx]
            offset = (i - (n_classes - 1) / 2.0) * width
            label = class_name_map[c] if class_name_map and c in class_name_map else f"class_{c}"
            plt.bar(x + offset, vals, width, label=label)

        plt.xticks(x, x_labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Mean |∂logit/∂x|")
        plt.title(f"Top features per subtype for {omic}")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(output_dir, f"grad_importance_combined_{omic}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_omic_2x2_barplots(
    importance,
    omic_names,
    output_dir,
    top_k=10,
    class_name_map=None,
    feature_name_map=None,
):
    """
    For each omic, create a single figure with 4 barplots (2x2), one per subtype.
    """
    os.makedirs(output_dir, exist_ok=True)

    class_ids = sorted(importance.keys())

    for omic in omic_names:
        # Collect per-class scores for this omic
        per_class_scores = {}
        for c in class_ids:
            omic_dict = importance.get(c, {})
            if omic in omic_dict:
                scores = omic_dict[omic]
                if scores.ndim != 1:
                    scores = scores.reshape(-1)
                per_class_scores[c] = scores

        if not per_class_scores:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, c in enumerate(class_ids):
            if idx >= 4:
                break
            ax = axes[idx]
            scores = per_class_scores.get(c)
            if scores is None:
                ax.axis("off")
                continue

            k = min(top_k, len(scores))
            top_idx = np.argsort(-scores)[:k]
            top_scores = scores[top_idx]

            # Labels for this class and omic
            x_labels = []
            for i in top_idx:
                label = None
                if feature_name_map and omic in feature_name_map:
                    names = feature_name_map[omic]
                    if 0 <= i < len(names):
                        label = names[int(i)]
                if label is None:
                    label = f"{omic}_feat_{int(i)}"
                x_labels.append(label)

            x = np.arange(k)
            ax.bar(x, top_scores)
            class_label = (
                class_name_map[c] if class_name_map and c in class_name_map else f"class_{c}"
            )
            ax.set_title(class_label)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)

        # Turn off any unused subplots
        for j in range(len(class_ids), 4):
            axes[j].axis("off")

        fig.suptitle(f"Top-{top_k} {omic} features per subtype")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = os.path.join(output_dir, f"grad_importance_2x2_{omic}.png")
        fig.savefig(fname, dpi=200)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradient-based feature importance visualization for SelfAttentionFusionModel."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_self_attn_cross_attn_mogcn_data.yaml",
        help="Path to YAML config used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="attention_plots/grad_importance",
        help="Directory to save barplot figures.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=10,
        help="Number of top features to plot per omic and per class.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally limit number of validation batches to use for gradients.",
    )
    parser.add_argument(
        "--use-true-label",
        action="store_true",
        help="If set, attribute gradients to the true label; otherwise to predicted label.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Optional comma-separated class names in index order, e.g. 'Basal,HER2,LumA,LumB'.",
    )
    return parser.parse_args()


def load_feature_name_map(cfg):
    """
    Load feature (gene/protein) names for each omic from the original CSVs.

    We assume the CSVs are in the same format as used by BRCADataModule:
    rows = features, columns = [sample_id, samples...], index_col=0 is feature name.
    """
    data_cfg = cfg["data"]
    omics_config = extract_omics_config(data_cfg)
    feature_name_map = {}

    for omic, path in omics_config.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        feature_name_map[omic] = df.index.astype(str).tolist()

    return feature_name_map


def main():
    args = parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Set seed for reproducibility
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = build_datamodule(cfg)
    val_loader = dm.val_dataloader()

    num_classes = cfg["model"]["num_classes"]

    model = load_model_from_checkpoint(cfg, dm, args.checkpoint, device)

    feature_name_map = load_feature_name_map(cfg)

    # Optional human-readable class names
    class_name_map = None
    if args.class_names is not None:
        names = [n.strip() for n in args.class_names.split(",")]
        class_name_map = {i: names[i] for i in range(min(len(names), num_classes))}

    importance, omic_names = compute_gradient_feature_importance(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=num_classes,
        max_batches=args.max_batches,
        use_true_label=args.use_true_label,
    )

    plot_topk_barplots_per_class(
        importance=importance,
        omic_names=omic_names,
        output_dir=args.output_dir,
        top_k=args.top_k_features,
        class_name_map=class_name_map,
        feature_name_map=feature_name_map,
    )

    # Combined per-omic heatmaps over all subtypes
    plot_omic_grouped_barplots(
        importance=importance,
        omic_names=omic_names,
        output_dir=args.output_dir,
        top_k=args.top_k_features,
        class_name_map=class_name_map,
        feature_name_map=feature_name_map,
    )

    # 2x2 layout: one barplot per subtype in a single figure per omic
    plot_omic_2x2_barplots(
        importance=importance,
        omic_names=omic_names,
        output_dir=args.output_dir,
        top_k=args.top_k_features,
        class_name_map=class_name_map,
        feature_name_map=feature_name_map,
    )


if __name__ == "__main__":
    main()


