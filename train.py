import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from datamodule import BRCADataModule
from models.self_attn_fusion import SelfAttentionFusionModel
import yaml
import argparse
import os
import json
import gc
import torch
import numpy as np
import random


class ValAccAtBestF1Callback(Callback):
    """Track val_acc at the epoch where val_f1 reaches its peak."""

    def __init__(self):
        self.best_val_f1 = None
        self.val_acc_at_best_f1 = None

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_f1 = metrics.get("val_f1")
        if val_f1 is None:
            return
        f1 = float(val_f1.item() if hasattr(val_f1, "item") else val_f1)
        if self.best_val_f1 is None or f1 > self.best_val_f1:
            self.best_val_f1 = f1
            val_acc = metrics.get("val_acc")
            if val_acc is not None:
                self.val_acc_at_best_f1 = float(
                    val_acc.item() if hasattr(val_acc, "item") else val_acc
                )


def create_early_stopping_callback(cfg):
    """Create a fresh early stopping callback from config."""
    if cfg["train"].get("early_stopping", {}).get("enable", False):
        return EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=cfg["train"]["early_stopping"].get("patience", 10),
            min_delta=cfg["train"]["early_stopping"].get("min_delta", 0.0),
            verbose=True
        )
    return None

def create_trainer_kwargs(cfg):
    """Build shared Trainer kwargs from config."""
    train_cfg = cfg.get("train", {})
    kwargs = {
        "max_epochs": train_cfg["epochs"],
        "accelerator": "auto",
        "devices": "auto",
        "log_every_n_steps": 10,
        "deterministic": True,
    }
    precision = train_cfg.get("precision")
    if precision is not None:
        kwargs["precision"] = precision
    return kwargs


def cleanup_fold_resources(*objects):
    """Release GPU/CPU memory between k-fold iterations."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_model_checkpoint_callbacks(cfg, dirpath=None):
    """Create ModelCheckpoint callbacks to track best val_acc and best val_f1."""
    callbacks = []
    train_cfg = cfg.get("train", {})

    checkpoint_kwargs = {
        "save_top_k": 1,
        "save_last": train_cfg.get("save_last_checkpoint", False),
        "save_weights_only": train_cfg.get("save_weights_only", True),
        "verbose": False,
    }
    if dirpath is not None:
        checkpoint_kwargs["dirpath"] = dirpath
    
    # Checkpoint for best val_acc
    checkpoint_acc = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best_val_acc_{epoch:02d}_{val_acc:.4f}",
        **checkpoint_kwargs
    )
    callbacks.append(checkpoint_acc)
    
    # Checkpoint for best val_f1
    checkpoint_f1 = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        filename="best_val_f1_{epoch:02d}_{val_f1:.4f}",
        **checkpoint_kwargs
    )
    callbacks.append(checkpoint_f1)
    
    return callbacks

def report_dir_from_cfg(cfg) -> str:
    log_cfg = cfg.get("logging", {})
    return log_cfg.get("report_dir", log_cfg.get("log_dir", "lightning_logs"))


def save_checkpoints_from_cfg(cfg) -> bool:
    return bool(cfg.get("train", {}).get("save_checkpoints", True))


def build_trainer_callbacks(cfg, fold=None):
    early_stop_callback = create_early_stopping_callback(cfg)
    if not save_checkpoints_from_cfg(cfg):
        callbacks = [early_stop_callback] if early_stop_callback else []
        return callbacks, [], early_stop_callback

    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "lightning_logs")
    base_name = log_cfg.get("log_name", "experiment")
    fold_log_name = f"{base_name}_fold_{fold + 1}" if fold is not None else base_name
    checkpoint_dir = os.path.join(log_dir, fold_log_name, "checkpoints")
    checkpoint_callbacks = create_model_checkpoint_callbacks(cfg, dirpath=checkpoint_dir)
    callbacks = checkpoint_callbacks + ([early_stop_callback] if early_stop_callback else [])
    return callbacks, checkpoint_callbacks, early_stop_callback


def create_logger(cfg, fold=None):
    """Create a logger with configurable name and directory."""
    log_config = cfg.get("logging", {})
    if not log_config.get("enabled", True):
        return False

    log_dir = log_config.get("log_dir", "lightning_logs")
    log_name = log_config.get("log_name", "experiment")
    logger_type = log_config.get("logger_type", "tensorboard")  # "tensorboard" or "csv"
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Add fold number to name if k-fold is being used
    if fold is not None:
        log_name = f"{log_name}_fold_{fold + 1}"
    
    if logger_type.lower() == "csv":
        return CSVLogger(save_dir=log_dir, name=log_name)
    else:  # tensorboard (default)
        return TensorBoardLogger(save_dir=log_dir, name=log_name)

def resolve_pre_split_fold_dir(data_cfg, fold_index_zero_based):
    """Path to fold_k directory under data.pre_split_base, or None if not using on-disk folds."""
    base = data_cfg.get("pre_split_base")
    if not base:
        return None
    return os.path.join(str(base), f"fold_{fold_index_zero_based + 1}")


def extract_omics_config(data_cfg):
    """
    Extract omics configuration from data config section.
    
    Supports two formats:
    1. Explicit omics_config dict: {"mRNA": path, "CNV": path, ...}
    2. Individual path keys: x_rna_path, x_cnv_path, x_proteomics_path, etc.
    
    Returns a dict mapping omic_name -> path.
    """
    # If omics_config is explicitly provided, use it
    if "omics_config" in data_cfg:
        return data_cfg["omics_config"]
    
    # Otherwise, extract from individual path keys
    omics_config = {}
    
    # Mapping from config keys to omic names
    path_mappings = {
        "x_rna_path": "mRNA",
        "x_cnv_path": "CNV",
        "x_meth_path": "Methylation",
        "x_proteomics_path": "Proteomics",
        "x_mirna_path": "miRNA",
    }
    
    # Extract all matching paths
    for key, omic_name in path_mappings.items():
        if key in data_cfg and data_cfg[key] is not None:
            omics_config[omic_name] = data_cfg[key]
    
    return omics_config

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    Sets seeds for Python random, NumPy, PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility, set deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(config_path):
    cfg = yaml.safe_load(open(config_path))

    # Set global random seed for reproducibility
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)
    print(f"Set random seed to: {seed}")

    use_stratified_kfold = cfg["data"].get("use_stratified_kfold", False)
    n_folds = cfg["data"].get("n_folds", 5)
    model_name = cfg.get("model", {}).get("name") or cfg.get("model", {}).get("type")

    if use_stratified_kfold:
        # Run k-fold cross-validation
        print(f"Running stratified {n_folds}-fold cross-validation")
        fold_results = []
        for fold in range(n_folds):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_folds}")
            print(f"{'='*50}")

            fold_seed = seed + fold
            set_seed(fold_seed)
            print(f"Set fold seed to: {fold_seed}")

            pre_fold_dir = resolve_pre_split_fold_dir(cfg["data"], fold)
            dm = BRCADataModule(
                omics_config=extract_omics_config(cfg["data"]),
                y_path=cfg["data"]["y_path"],
                batch_size=cfg["train"]["batch_size"],
                num_workers=cfg["train"]["num_workers"],
                val_split=cfg["data"]["val_split"],
                seed=fold_seed,
                normalize=cfg["train"]["normalize"],
                use_stratified_kfold=pre_fold_dir is None,
                n_folds=n_folds,
                current_fold=fold,
                pre_split_fold_dir=pre_fold_dir,
            )

            # Ensure we have feature dimensions from the datamodule before model init
            # so we can build models that depend on input sizes dynamically.
            dm.setup(stage="fit")

            model_name = cfg.get("model", {}).get("name") or cfg.get("model", {}).get("type")
            if model_name == "self_attn_fusion":
                model = SelfAttentionFusionModel(
                    input_dims=dm.feature_dims if cfg["model"].get("input_dims") is None else cfg["model"]["input_dims"],
                    num_classes=cfg["model"]["num_classes"],
                    latent_dim=cfg["model"].get("latent_dim", 128),
                    encoder_hidden_dim=cfg["model"].get("hidden_dim", 512),
                    fusion_hidden=cfg["model"].get("fusion_hidden", 256),
                    dropout=cfg["model"].get("dropout", 0.3),
                    lr=float(cfg["train"].get("lr", 1e-4)),
                    weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
                    n_heads=cfg["model"].get("n_heads", 4),
                    n_layers=cfg["model"].get("n_layers", 2),
                    fusion_type=cfg["model"].get("fusion_type", "attention"),
                    use_cross_attention=cfg["model"].get("use_cross_attention", False),
                    cross_attn_hidden=cfg["model"].get("cross_attn_hidden", 128),
                    use_original_for_proteomics=cfg["model"].get("use_original_for_proteomics", True),
                    use_gated_fusion=cfg["model"].get("use_gated_fusion", False),
                    gated_fusion_hidden=cfg["model"].get("gated_fusion_hidden", 128),
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            val_acc_at_best_f1_cb = ValAccAtBestF1Callback()
            callbacks, checkpoint_callbacks, early_stop_callback = build_trainer_callbacks(cfg, fold=fold)
            callbacks = [val_acc_at_best_f1_cb] + list(callbacks)
            logger = create_logger(cfg, fold=fold)
            trainer = pl.Trainer(
                **create_trainer_kwargs(cfg),
                callbacks=callbacks,
                logger=logger,
                enable_checkpointing=save_checkpoints_from_cfg(cfg),
            )

            trainer.fit(model, datamodule=dm)

            # Collect validation metrics for this fold
            metrics = trainer.validate(model, datamodule=dm, verbose=False)
            val_metrics = metrics[0] if len(metrics) > 0 else {}
            best_val_f1 = None
            best_val_acc = None
            
            # Extract best scores from ModelCheckpoint callbacks
            if len(checkpoint_callbacks) >= 2:
                # First callback monitors val_acc
                if hasattr(checkpoint_callbacks[0], "best_model_score") and checkpoint_callbacks[0].best_model_score is not None:
                    try:
                        best_val_acc = float(checkpoint_callbacks[0].best_model_score.item())
                    except Exception:
                        pass
                # Second callback monitors val_f1
                if hasattr(checkpoint_callbacks[1], "best_model_score") and checkpoint_callbacks[1].best_model_score is not None:
                    try:
                        best_val_f1 = float(checkpoint_callbacks[1].best_model_score.item())
                    except Exception:
                        pass
            
            # Fallback to early stopping callback if ModelCheckpoint didn't capture it
            if best_val_f1 is None and early_stop_callback is not None and hasattr(early_stop_callback, "best_score"):
                try:
                    best_val_f1 = float(early_stop_callback.best_score.item())
                except Exception:
                    pass

            fold_results.append({
                "fold": fold + 1,
                "f1_score": best_val_f1,
                "accuracy": val_acc_at_best_f1_cb.val_acc_at_best_f1,
            })
            cleanup_fold_resources(trainer, model, dm)

        # Aggregate averages across folds and write report
        log_cfg = cfg.get("logging", {})
        report_dir = report_dir_from_cfg(cfg)
        base_name = log_cfg.get("log_name", "experiment")
        report_path = os.path.join(report_dir, f"{base_name}_k{n_folds}_report.json")

        def _avg(key):
            vals = [
                fr[key] for fr in fold_results
                if fr.get(key) is not None
                and not (isinstance(fr.get(key), float) and np.isnan(fr.get(key)))
            ]
            if not vals:
                return None
            return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"

        averages = {
            "f1_score": _avg("f1_score"),
            "accuracy": _avg("accuracy"),
        }
        averages = {k: v for k, v in averages.items() if v is not None}

        report = {
            "n_folds": n_folds,
            "per_fold": fold_results,
            "averages": averages,
        }

        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved k-fold report to: {report_path}")
    else:
        # Original single train/val split
        dm = BRCADataModule(
            omics_config=extract_omics_config(cfg["data"]),
            y_path=cfg["data"]["y_path"],
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            val_split=cfg["data"]["val_split"],
            seed=cfg["train"]["seed"],
            normalize=cfg["train"]["normalize"]
        )

        # Ensure datamodule is set up to expose feature dimensions before building model
        dm.setup(stage="fit")

        model_name = cfg.get("model", {}).get("name") or cfg.get("model", {}).get("type")
        if model_name == "self_attn_fusion":
            model = SelfAttentionFusionModel(
                input_dims=dm.feature_dims if cfg["model"].get("input_dims") is None else cfg["model"]["input_dims"],
                num_classes=cfg["model"]["num_classes"],
                latent_dim=cfg["model"].get("latent_dim", 128),
                encoder_hidden_dim=cfg["model"].get("hidden_dim", 512),
                fusion_hidden=cfg["model"].get("fusion_hidden", 256),
                dropout=cfg["model"].get("dropout", 0.3),
                lr=float(cfg["train"].get("lr", 1e-4)),
                weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
                n_heads=cfg["model"].get("n_heads", 4),
                n_layers=cfg["model"].get("n_layers", 2),
                fusion_type=cfg["model"].get("fusion_type", "attention"),
                use_cross_attention=cfg["model"].get("use_cross_attention", False),
                cross_attn_hidden=cfg["model"].get("cross_attn_hidden", 128),
                use_original_for_proteomics=cfg["model"].get("use_original_for_proteomics", True),
                use_gated_fusion=cfg["model"].get("use_gated_fusion", False),
                gated_fusion_hidden=cfg["model"].get("gated_fusion_hidden", 128),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        val_acc_at_best_f1_cb = ValAccAtBestF1Callback()
        callbacks, checkpoint_callbacks, early_stop_callback = build_trainer_callbacks(cfg, fold=None)
        callbacks = [val_acc_at_best_f1_cb] + list(callbacks)
        logger = create_logger(cfg, fold=None)
        trainer = pl.Trainer(
            **create_trainer_kwargs(cfg),
            callbacks=callbacks,
            logger=logger,
            enable_checkpointing=save_checkpoints_from_cfg(cfg),
        )

        trainer.fit(model, datamodule=dm)

        # Single-run report
        metrics = trainer.validate(model, datamodule=dm, verbose=False)
        val_metrics = metrics[0] if len(metrics) > 0 else {}
        best_val_f1 = None
        best_val_acc = None
        
        # Extract best scores from ModelCheckpoint callbacks
        if len(checkpoint_callbacks) >= 2:
            # First callback monitors val_acc
            if hasattr(checkpoint_callbacks[0], "best_model_score") and checkpoint_callbacks[0].best_model_score is not None:
                try:
                    best_val_acc = float(checkpoint_callbacks[0].best_model_score.item())
                except Exception:
                    pass
            # Second callback monitors val_f1
            if hasattr(checkpoint_callbacks[1], "best_model_score") and checkpoint_callbacks[1].best_model_score is not None:
                try:
                    best_val_f1 = float(checkpoint_callbacks[1].best_model_score.item())
                except Exception:
                    pass
        
        # Fallback to early stopping callback if ModelCheckpoint didn't capture it
        if best_val_f1 is None and early_stop_callback is not None and hasattr(early_stop_callback, "best_score"):
            try:
                best_val_f1 = float(early_stop_callback.best_score.item())
            except Exception:
                pass

        log_cfg = cfg.get("logging", {})
        report_dir = report_dir_from_cfg(cfg)
        base_name = log_cfg.get("log_name", "experiment")
        report_path = os.path.join(report_dir, f"{base_name}_report.json")

        report = {
            "per_fold": [{
                "fold": 1,
                "f1_score": best_val_f1,
                "accuracy": val_acc_at_best_f1_cb.val_acc_at_best_f1,
            }],
            "averages": {
                "f1_score": best_val_f1,
                "accuracy": val_acc_at_best_f1_cb.val_acc_at_best_f1,
            }
        }

        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
