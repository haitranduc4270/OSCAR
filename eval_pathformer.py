"""Evaluate an OSCAR model with Pathformer's train/validation/test protocol.

The full cohort is rebuilt from one pre-split fold (train and validation files
together). Samples are then stratified into test (20%) and, from what remains,
validation (20%). Early stopping and the saved checkpoint follow validation
macro-F1. Test metrics are computed every epoch and do not affect that choice.
"""

import argparse
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from datamodule import BRCADataModule
from train import build_model, create_trainer_kwargs, extract_omics_config, set_seed


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """Metrics Pathformer writes for a multi-class run, using softmax probabilities."""
    y_true = np.asarray(y_true).astype(int)
    prob = softmax(logits)
    pred = prob.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted")),
        "auc_macro_ovr": float(roc_auc_score(y_true, prob, average="macro", multi_class="ovr")),
        "auc_macro_ovo": float(roc_auc_score(y_true, prob, average="macro", multi_class="ovo")),
        "auc_weighted_ovr": float(roc_auc_score(y_true, prob, average="weighted", multi_class="ovr")),
        "auc_weighted_ovo": float(roc_auc_score(y_true, prob, average="weighted", multi_class="ovo")),
    }

    y_binary = (y_true >= 1).astype(int)
    score_binary = 1.0 - prob[:, 0]
    pred_binary = (score_binary > 0.5).astype(int)
    metrics["accuracy_class0_vs_rest"] = float(accuracy_score(y_binary, pred_binary))
    metrics["auc_class0_vs_rest"] = float(roc_auc_score(y_binary, score_binary))
    metrics["f1_macro_class0_vs_rest"] = float(f1_score(y_binary, pred_binary, average="macro"))
    metrics["f1_weighted_class0_vs_rest"] = float(f1_score(y_binary, pred_binary, average="weighted"))
    return metrics


@torch.no_grad()
def predict_split(model, dataset, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels, logits = [], []
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    for batch in loader:
        labels.append(batch["label"].numpy())
        batch = {name: value.to(device) for name, value in batch.items()}
        logits.append(model(batch).detach().float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(labels), np.concatenate(logits)


class PathformerSelection:
    """Same accept/stop rule as Pathformer's EarlyStopping on a score to maximize."""

    def __init__(self, patience: int = 10, delta: float = 1e-2):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_epoch = True

    def update(self, score: float, epoch: int) -> None:
        if self.best_score is None or epoch == 1:
            self.best_score = score
            self.counter = 0
            self.save_epoch = True
            self.early_stop = False
            return
        if score < self.best_score - self.delta:
            self.counter += 1
            self.save_epoch = False
            self.early_stop = self.counter >= self.patience
            return
        self.best_score = score
        self.counter = 0
        self.save_epoch = True
        self.early_stop = False


class PathformerEvalCallback(Callback):
    def __init__(self, patience: int, delta: float, ckpt_path: str):
        self.selection = PathformerSelection(patience=patience, delta=delta)
        self.ckpt_path = ckpt_path
        self.history = []
        self.checkpoint = None
        self.peak = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        datamodule = trainer.datamodule
        epoch = trainer.current_epoch + 1
        y_val, logits_val = predict_split(pl_module, datamodule.val_set, datamodule.batch_size)
        y_test, logits_test = predict_split(pl_module, datamodule.test_set, datamodule.batch_size)
        val_metrics = classification_metrics(y_val, logits_val)
        test_metrics = classification_metrics(y_test, logits_test)
        row = {"epoch": epoch, "val": val_metrics, "test": test_metrics}
        self.history.append(row)

        self.selection.update(val_metrics["f1_macro"], epoch)
        if self.selection.save_epoch:
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            torch.save(pl_module.state_dict(), self.ckpt_path)
            self.checkpoint = {
                "epoch": epoch,
                "val": val_metrics,
                "test": test_metrics,
            }
        if self.peak is None or val_metrics["f1_macro"] > self.peak["val"]["f1_macro"]:
            self.peak = {
                "epoch": epoch,
                "val": val_metrics,
                "test": test_metrics,
            }

        print(
            f"epoch {epoch}: val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"test_f1_macro={test_metrics['f1_macro']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"patience {self.selection.counter}/{self.selection.patience}"
        )
        if self.selection.early_stop:
            trainer.should_stop = True


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    seed = 2022
    set_seed(seed)
    print(f"Pathformer split protocol, seed {seed}")

    fold_dir = os.path.join(cfg["data"]["pre_split_base"], "fold_1")
    dm = BRCADataModule(
        omics_config=extract_omics_config(cfg["data"]),
        y_path=cfg["data"]["y_path"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=0,
        val_split=0.2,
        seed=seed,
        normalize=cfg["train"]["normalize"],
        pre_split_fold_dir=fold_dir,
        split_protocol="pathformer",
        test_size=0.2,
    )
    dm.setup(stage="fit")
    print("split", dm.split_counts)

    model = build_model(cfg, dm)
    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "lightning_logs")
    base_name = log_cfg.get("log_name", "experiment")
    ckpt_path = os.path.join(log_dir, f"{base_name}_pathformer", "checkpoints", "best.pt")
    patience = 10
    delta = 1e-2
    callback = PathformerEvalCallback(patience=patience, delta=delta, ckpt_path=ckpt_path)

    trainer = pl.Trainer(
        **create_trainer_kwargs(cfg),
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)

    report = {
        "protocol": "pathformer",
        "config": config_path,
        "model": cfg.get("model", {}).get("name"),
        "seed": seed,
        "split": {
            "test_size": 0.2,
            "val_fraction_of_remaining": 0.2,
            "source": fold_dir,
            **dm.split_counts,
        },
        "selection": {
            "monitor": "val_f1_macro",
            "patience": patience,
            "delta": delta,
            "max_epochs": cfg["train"]["epochs"],
        },
        "checkpoint": callback.checkpoint,
        "peak_val_f1_macro": callback.peak,
        "history": callback.history,
    }
    report_path = os.path.join(log_dir, f"{base_name}_pathformer_report.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved report to: {report_path}")
    chosen = callback.checkpoint
    print(
        f"checkpoint epoch {chosen['epoch']}: "
        f"test accuracy {chosen['test']['accuracy']:.4f}, "
        f"test macro-F1 {chosen['test']['f1_macro']:.4f}, "
        f"test macro-AUC OVR {chosen['test']['auc_macro_ovr']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_self_attn_cross_attn_brca.yaml")
    args = parser.parse_args()
    main(args.config)
