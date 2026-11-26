import os
import random
import numpy as np
import torch


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
