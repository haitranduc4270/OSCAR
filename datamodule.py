from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


class MultiOmicDataset(Dataset):
    def __init__(self, X_dict, y):
        """
        X_dict: dict of {omic_name: np.ndarray (N, F)}
        y: np.ndarray (N,)
        """
        self.X_dict = X_dict
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        batch = {name: torch.tensor(X[idx], dtype=torch.float32) for name, X in self.X_dict.items()}
        batch["label"] = torch.tensor(self.y[idx], dtype=torch.long)
        return batch


class BRCADataModule(LightningDataModule):
    """
    Loads omics data from CSVs where:
      - Each omic file shape = (G, N+1)
          rows = genes
          columns = [sample_id, sample_1, sample_2, ..., sample_N]
      - y.csv shape = (N, 1)
    
    Flexible omics configuration:
      - omics_config: dict mapping omic_name -> path (e.g., {"mRNA": path, "CNV": path, "Proteomics": path})
    """
    def __init__(
        self,
        omics_config: dict,
        y_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        seed: int = 42,
        normalize: bool = True,
        use_stratified_kfold: bool = False,
        n_folds: int = 5,
        current_fold: int = 0,
        pre_split_fold_dir: str | None = None,
        pin_memory: bool | None = None,
    ):
        super().__init__()
        
        if not omics_config:
            raise ValueError("omics_config must be provided and cannot be empty.")
        
        self.paths = omics_config
        
        self.y_path = y_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.normalize = normalize
        self.use_stratified_kfold = use_stratified_kfold
        self.n_folds = n_folds
        self.current_fold = current_fold
        self.pre_split_fold_dir = pre_split_fold_dir
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if self.pre_split_fold_dir:
            fold_dir = Path(self.pre_split_fold_dir)
            if not fold_dir.is_dir():
                raise NotADirectoryError(f"pre_split_fold_dir not found: {fold_dir}")

            def split_paths(stem: str) -> tuple[Path, Path]:
                tr = fold_dir / f"{stem}_train.csv"
                va = fold_dir / f"{stem}_val.csv"
                if not tr.is_file() or not va.is_file():
                    raise FileNotFoundError(
                        f"Expected {tr.name} and {va.name} under {fold_dir}"
                    )
                return tr, va

            y_stem = Path(self.y_path).stem
            y_train_path, y_val_path = split_paths(y_stem)

            omic_dfs_train = {}
            omic_dfs_val = {}
            for name, path in self.paths.items():
                stem = Path(path).stem
                p_train, p_val = split_paths(stem)
                omic_dfs_train[name] = pd.read_csv(p_train, index_col=0)
                omic_dfs_val[name] = pd.read_csv(p_val, index_col=0)

            X_train = {name: df.T for name, df in omic_dfs_train.items()}
            X_val = {name: df.T for name, df in omic_dfs_val.items()}

            y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy()
            y_val = pd.read_csv(y_val_path).iloc[:, 0].to_numpy()

            if self.normalize:
                X_train_dict = {}
                X_val_dict = {}
                for name in X_train:
                    scaler = StandardScaler()
                    X_train_dict[name] = scaler.fit_transform(X_train[name]).astype(np.float32)
                    X_val_dict[name] = scaler.transform(X_val[name]).astype(np.float32)
            else:
                X_train_dict = {name: X.to_numpy(dtype=np.float32) for name, X in X_train.items()}
                X_val_dict = {name: X.to_numpy(dtype=np.float32) for name, X in X_val.items()}

            self.train_set = MultiOmicDataset(X_train_dict, y_train)
            self.val_set = MultiOmicDataset(X_val_dict, y_val)
            self.feature_dims = {name: X_train_dict[name].shape[1] for name in X_train_dict}
            return

        omic_dfs = {name: pd.read_csv(path, index_col=0) for name, path in self.paths.items()}

        X_dict = {name: df.T for name, df in omic_dfs.items()}

        y_df = pd.read_csv(self.y_path)
        y = y_df.iloc[:, 0].to_numpy()  # shape (N,)

        if self.normalize:
            scaler = StandardScaler()
            X_dict = {name: scaler.fit_transform(X) for name, X in X_dict.items()}
        else:
            X_dict = {name: X.to_numpy(dtype=np.float32) for name, X in X_dict.items()}

        self.dataset = MultiOmicDataset(X_dict, y)

        if self.use_stratified_kfold:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            splits = list(skf.split(np.arange(len(y)), y))
            train_indices, val_indices = splits[self.current_fold]
            
            self.train_set = Subset(self.dataset, train_indices)
            self.val_set = Subset(self.dataset, val_indices)
        else:
            n_total = len(y)
            n_val = int(self.val_split * n_total)
            n_train = n_total - n_val
            self.train_set, self.val_set = random_split(
                self.dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

        # Store feature dims for model config
        self.feature_dims = {name: X_dict[name].shape[1] for name in X_dict}

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
