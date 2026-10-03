"""PyTorch Lightning port of MOGONET (Wang et al., Nature Communications 2021).

Kept from the official code:
  - omic-specific GCN + linear head, then VCDN product fusion
  - cosine kNN adjacency, bipartite train-eval graph at inference (transductive)
  - per-view CE then VCDN CE; GCN weights are NOT updated by the VCDN loss
  - class-frequency sample weights (as published; not inverse-frequency)

Training is full-batch. Extra DataLoader batches in an epoch are skipped.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLightningModel


def xavier_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1.0 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def graph_from_dist(dist: torch.Tensor, parameter: float, self_dist: bool) -> torch.Tensor:
    g = (dist <= parameter).float()
    if self_dist:
        g.fill_diagonal_(0.0)
    return g


def symmetrize_and_normalize(adj: torch.Tensor) -> torch.Tensor:
    adj_t = adj.transpose(0, 1)
    gt = (adj_t > adj).float()
    adj = adj + adj_t * gt - adj * gt
    eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    return F.normalize(adj + eye, p=1, dim=1)


def adj_threshold(data: torch.Tensor, edge_per_node: int) -> float:
    dist = cosine_distance(data)
    k = min(int(edge_per_node) * data.shape[0], dist.numel() - 1)
    return float(torch.sort(dist.reshape(-1)).values[k].item())


def train_adj(data: torch.Tensor, parameter: float) -> torch.Tensor:
    dist = cosine_distance(data)
    g = graph_from_dist(dist, parameter, self_dist=True)
    return symmetrize_and_normalize((1.0 - dist) * g)


def transductive_adj(data: torch.Tensor, n_train: int, parameter: float) -> torch.Tensor:
    """Bipartite train-eval graph from the official gen_test_adj_mat_tensor."""
    adj = torch.zeros(data.shape[0], data.shape[0], device=data.device, dtype=data.dtype)
    tr = data[:n_train]
    te = data[n_train:]
    dist_tr2te = cosine_distance(tr, te)
    dist_te2tr = cosine_distance(te, tr)
    adj[:n_train, n_train:] = (1.0 - dist_tr2te) * graph_from_dist(dist_tr2te, parameter, False)
    adj[n_train:, :n_train] = (1.0 - dist_te2tr) * graph_from_dist(dist_te2tr, parameter, False)
    return symmetrize_and_normalize(adj)


def published_sample_weight(labels: torch.Tensor, num_class: int) -> torch.Tensor:
    """Official MOGONET weights: class frequency, not inverse frequency."""
    y = labels.detach().cpu()
    count = torch.bincount(y, minlength=num_class).float()
    total = float(count.sum().clamp(min=1.0))
    w = torch.zeros(y.shape[0], dtype=torch.float32)
    for c in range(num_class):
        w[y == c] = count[c] / total
    return w


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_normal_(self.weight.data)
        self.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return adj @ (x @ self.weight) + self.bias


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.layers = nn.ModuleList(
            [GraphConvolution(dims[i], dims[i + 1]) for i in range(len(hidden_dims))]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = F.leaky_relu(layer(h, adj), 0.25)
            if i < len(self.layers) - 1:
                h = F.dropout(h, self.dropout, training=self.training)
        return h


class VCDN(nn.Module):
    def __init__(self, num_view: int, num_cls: int, hidden_dim: int):
        super().__init__()
        self.num_cls = num_cls
        self.num_view = num_view
        self.model = nn.Sequential(
            nn.Linear(num_cls**num_view, hidden_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hidden_dim, num_cls),
        )
        self.model.apply(xavier_linear)

    def forward(self, in_list: list[torch.Tensor]) -> torch.Tensor:
        probs = [torch.sigmoid(t) for t in in_list]
        x = torch.matmul(probs[0].unsqueeze(-1), probs[1].unsqueeze(1))
        x = x.reshape(-1, self.num_cls**2, 1)
        for i in range(2, self.num_view):
            x = torch.matmul(x, probs[i].unsqueeze(1)).reshape(-1, self.num_cls ** (i + 1), 1)
        return self.model(x.reshape(-1, self.num_cls**self.num_view))


class MoGONETModel(BaseLightningModel):
    def __init__(
        self,
        input_dims,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
        lr: float = 5e-4,
        lr_c: float | None = None,
        lr_e_pretrain: float | None = None,
        weight_decay: float = 0.0,
        adj_parameter: int = 10,
        pretrain_epochs: int = 100,
        loss_type: str = "ce",
        class_weights=None,
        optimizer_type: str = "adam",
    ):
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            loss_type=loss_type,
            class_weights=class_weights,
            optimizer_type=optimizer_type,
        )
        self.automatic_optimization = False
        self.valid_omics = list(input_dims.keys())
        self.hidden_dims = list(hidden_dims or [400, 400, 200])
        self.gcn_dropout = float(dropout)
        self.adj_parameter = int(adj_parameter)
        self.pretrain_epochs = int(pretrain_epochs)
        self.lr_e = float(lr)
        self.lr_c = float(lr if lr_c is None else lr_c)
        self.lr_e_pretrain = float(lr if lr_e_pretrain is None else lr_e_pretrain)

        self.encoders = nn.ModuleDict(
            {
                name: GCNEncoder(input_dims[name], self.hidden_dims, self.gcn_dropout)
                for name in self.valid_omics
            }
        )
        self.classifiers = nn.ModuleDict(
            {
                name: nn.Linear(self.hidden_dims[-1], num_classes)
                for name in self.valid_omics
            }
        )
        for clf in self.classifiers.values():
            xavier_linear(clf)
        n_view = len(self.valid_omics)
        self.vcdn = VCDN(n_view, num_classes, num_classes**n_view)

        self._x_train = None
        self._y_train = None
        self._sample_weight = None
        self._train_adj = None
        self._adj_param = None
        self._x_eval = None
        self._y_eval = None

    def configure_optimizers(self):
        opts = []
        for name in self.valid_omics:
            params = list(self.encoders[name].parameters()) + list(
                self.classifiers[name].parameters()
            )
            opts.append(
                torch.optim.Adam(params, lr=self.lr_e, weight_decay=self.weight_decay)
            )
        opts.append(
            torch.optim.Adam(
                self.vcdn.parameters(), lr=self.lr_c, weight_decay=self.weight_decay
            )
        )
        return opts

    def _tensors_from_dataset(self, dataset):
        x = {
            name: torch.as_tensor(dataset.X_dict[name], dtype=torch.float32, device=self.device)
            for name in self.valid_omics
        }
        y = torch.as_tensor(dataset.y, dtype=torch.long, device=self.device)
        return x, y

    def on_fit_start(self):
        dm = self.trainer.datamodule
        self._x_train, self._y_train = self._tensors_from_dataset(dm.train_set)
        self._sample_weight = published_sample_weight(
            self._y_train, self.num_classes
        ).to(self.device)
        self._adj_param = {
            name: adj_threshold(self._x_train[name], self.adj_parameter)
            for name in self.valid_omics
        }
        self._train_adj = {
            name: train_adj(self._x_train[name], self._adj_param[name])
            for name in self.valid_omics
        }
        self._pretrain_gcns()

    def _pretrain_gcns(self):
        if self.pretrain_epochs <= 0:
            return
        opts = [
            torch.optim.Adam(
                list(self.encoders[name].parameters())
                + list(self.classifiers[name].parameters()),
                lr=self.lr_e_pretrain,
            )
            for name in self.valid_omics
        ]
        self.train()
        for _ in range(self.pretrain_epochs):
            for i, name in enumerate(self.valid_omics):
                opts[i].zero_grad()
                logits = self.classifiers[name](
                    self.encoders[name](self._x_train[name], self._train_adj[name])
                )
                loss = self._weighted_ce(logits, self._y_train, self._sample_weight)
                loss.backward()
                opts[i].step()

    @staticmethod
    def _weighted_ce(logits, y, weight):
        return torch.mean(F.cross_entropy(logits, y, reduction="none") * weight)

    def _view_logits(self, x_dict, adj_dict):
        return [
            self.classifiers[name](self.encoders[name](x_dict[name], adj_dict[name]))
            for name in self.valid_omics
        ]

    def forward(self, batch):
        raise RuntimeError("MOGONET is full-batch; use training_step / eval splits.")

    def training_step(self, batch, batch_idx):
        if batch_idx != 0 or self._x_train is None:
            return
        opts = self.optimizers()
        view_opts, vcdn_opt = opts[:-1], opts[-1]

        for i, name in enumerate(self.valid_omics):
            view_opts[i].zero_grad()
            logits = self.classifiers[name](
                self.encoders[name](self._x_train[name], self._train_adj[name])
            )
            loss = self._weighted_ce(logits, self._y_train, self._sample_weight)
            self.manual_backward(loss)
            view_opts[i].step()

        vcdn_opt.zero_grad()
        ci_list = self._view_logits(self._x_train, self._train_adj)
        fused = self.vcdn(ci_list)
        loss_c = self._weighted_ce(fused, self._y_train, self._sample_weight)
        self.manual_backward(loss_c)
        vcdn_opt.step()

        self.train_acc(fused, self._y_train)
        self.log("train_loss", loss_c, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss_c

    def _cache_eval(self, dataset):
        self._x_eval, self._y_eval = self._tensors_from_dataset(dataset)

    def _eval_fused_logits(self):
        n_tr = self._y_train.shape[0]
        x_cat = {
            name: torch.cat([self._x_train[name], self._x_eval[name]], dim=0)
            for name in self.valid_omics
        }
        adj = {
            name: transductive_adj(x_cat[name], n_tr, self._adj_param[name])
            for name in self.valid_omics
        }
        return self.vcdn(self._view_logits(x_cat, adj))[n_tr:]

    def on_validation_epoch_start(self):
        dm = self.trainer.datamodule
        if dm.val_set is None:
            self._x_eval = None
            return
        self._cache_eval(dm.val_set)

    def validation_step(self, batch, batch_idx):
        if batch_idx != 0 or self._x_eval is None:
            return
        logits = self._eval_fused_logits()
        y = self._y_eval
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        dm = self.trainer.datamodule
        if dm.test_set is None:
            self._x_eval = None
            return
        self._cache_eval(dm.test_set)

    def test_step(self, batch, batch_idx):
        if batch_idx != 0 or self._x_eval is None:
            return
        logits = self._eval_fused_logits()
        y = self._y_eval
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
