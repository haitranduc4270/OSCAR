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

    def setup(self, stage=None):
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
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
