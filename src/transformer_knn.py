"""
transformer_knn.py

Default method (mode="analog_mode")
------------------------------------------
1) Train a multi-head self-attention encoder with learned attention pooling and a linear
   forecast head, optimized end-to-end with MSE. The head only shapes the embedding during
   training and is discarded at inference.
2) Build a datastore of (encoder embedding, observed future trajectory) from all training
   sequences.
3) At forecast time, embed the current state, retrieve K nearest neighbors by Euclidean
   distance in embedding space, form softmax sampling weights with temperature tau, and let
   each of E ensemble members independently sample one analog trajectory. This preserves the
   observed multivariate/spatial dependence structure by construction.

Input features are the full hydroclimatic signal S = [W, O] (wavelet-reconstructed hydrologic
variability W and climate teleconnection indices O). The forecast TARGET is W only; O enters
as a lagged covariate and is not projected forward. Set feature_columns = [W..., O...] and
target_columns = [W...] accordingly.

Retained baselines / optionality (point forecasts only)
-------------------------------------------------------
- mean="transformer": raw transformer head forecast (ablation baseline).
- mean="knn_blend":  transformer forecast blended with KNN datastore neighbors (Zhang et al.,
  2025-style output reweighting; retained as a baseline). Controlled by KNNMeanConfig.alpha.
These run in mode="forecast_mode" for POINT forecasts. Ensemble uncertainty in this archival
version is produced exclusively by analog trajectory sampling (mode="analog_mode").

Dependencies: numpy, pandas, torch, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


# =========================
# Loss functions
# =========================

def get_loss(loss_name: str) -> nn.Module:
    """
    Returns a callable loss matching the signature loss(pred, true).

    Baseline mean squared error loss is employed, flexible in application.
    """
    name = (loss_name or "mse").lower()
    if name in {"mse", "l2"}:
        return nn.MSELoss()
    raise ValueError(
        f"Unknown/unsupported loss_name: {loss_name!r}. This archival version trains with "
        f"'mse' (as in the manuscript)."
    )


# =========================
# Model definition (self-attention encoder + forecast head)
# =========================

class AttnPool(nn.Module):
    """Attention pooling over time: [B,L,H] -> [B,H]."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)  # per-timestep score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,H]
        a = self.score(x).squeeze(-1)          # [B,L]
        w = torch.softmax(a, dim=1)            # [B,L]
        return torch.sum(w.unsqueeze(-1) * x, dim=1)  # [B,H]


class TransformerMTS_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, nhead=4, dropout=0.0, max_len=2048, pool="attn"):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if pool == "attn":
            self.pool = AttnPool(hidden_dim)
        elif pool == "mean":
            self.pool = lambda x: x.mean(dim=1)
        elif pool == "last":
            self.pool = lambda x: x[:, -1, :]
        else:
            raise ValueError(f"Unknown pool: {pool}")

    def forward(self, x):
        x = self.input_layer(x)               # [B,L,H]
        L = x.size(1)
        x = x + self.pos_emb[:, :L, :]
        x = self.transformer_encoder(x)       # [B,L,H]
        emb = self.pool(x)                    # [B,H]
        return emb


class EncoderWithForecastHead(nn.Module):
    """
    Encoder + linear head to forecast [B, pred_horizon, output_dim].

    The head is used only to shape the embedding during training; at inference the encoder
    embedding is used as the analog-retrieval key.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pred_horizon: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = TransformerMTS_Encoder(input_dim, hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout)
        self.forecast_head = nn.Linear(hidden_dim, output_dim * pred_horizon)
        self.pred_horizon = pred_horizon
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)                         # [B, H]
        out = self.forecast_head(emb)                 # [B, out_dim*h]
        return out.view(-1, self.pred_horizon, self.output_dim)


# =========================
# Sequencing / splits
# =========================

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def create_sequences_np(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    seq_length: int,
    pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: [N, seq_length, n_features]
      y: [N, pred_horizon, n_targets]
    """
    feat = data[feature_columns].to_numpy(dtype=np.float32)
    targ = data[target_columns].to_numpy(dtype=np.float32)

    n = len(data)
    last_start = n - seq_length - pred_horizon
    if last_start < 0:
        return (
            np.zeros((0, seq_length, len(feature_columns)), dtype=np.float32),
            np.zeros((0, pred_horizon, len(target_columns)), dtype=np.float32),
        )

    X = np.stack([feat[i : i + seq_length] for i in range(last_start + 1)], axis=0)
    y = np.stack([targ[i + seq_length : i + seq_length + pred_horizon] for i in range(last_start + 1)], axis=0)
    return X, y


def compute_time_splits(
    n: int,
    seq_length: int,
    pred_horizon: int,
    batch_size: int,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    require_full_train_batch: bool = True,
) -> Dict[str, int]:
    if n <= 0:
        raise ValueError("n must be positive.")

    if val_size is None:
        val_size = int(round(n * val_frac))
    if test_size is None:
        test_size = int(round(n * test_frac))

    val_size = max(val_size, pred_horizon)
    #test_size = max(test_size, pred_horizon)

    if require_full_train_batch:
        min_train_rows = seq_length + pred_horizon + batch_size - 1
    else:
        min_train_rows = seq_length + pred_horizon

    if val_size + test_size >= n:
        raise ValueError(f"val_size+test_size must be < n. Got {val_size+test_size} >= {n}.")

    train_end = n - (val_size + test_size)
    if train_end < min_train_rows:
        raise ValueError(
            "Not enough data for requested split under sequencing constraints.\n"
            f"n={n}, seq_length={seq_length}, pred_horizon={pred_horizon}, batch_size={batch_size}\n"
            f"val_size={val_size}, test_size={test_size} => train_rows={train_end}\n"
            f"Need train_rows >= {min_train_rows}."
        )

    val_start = train_end
    val_end = val_start + val_size
    test_start = val_end
    test_end = n

    val_ext_start = max(0, val_start - seq_length)
    test_ext_start = max(0, test_start - seq_length)

    n_train_seq = train_end - seq_length - pred_horizon + 1
    n_val_seq_ext = (val_end - val_ext_start) - seq_length - pred_horizon + 1
    n_test_seq_ext = (test_end - test_ext_start) - seq_length - pred_horizon + 1

    return {
        "n": n,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "val_ext_start": val_ext_start,
        "test_ext_start": test_ext_start,
        "n_train_seq": max(0, n_train_seq),
        "n_val_seq_ext": max(0, n_val_seq_ext),
        "n_test_seq_ext": max(0, n_test_seq_ext),
        "val_size": val_size,
        "test_size": test_size,
    }


# =========================
# KNN helpers (CPU)
# =========================

def _stack_keys(store: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    # store keys are CPU tensors [H]
    return torch.stack([k for k, _ in store], dim=0)  # [N, H] CPU


def knn_topk(
    query_key_cpu: torch.Tensor,  # [H] CPU
    keys_cpu: torch.Tensor,       # [N, H] CPU
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      topk_idx: [K_eff] indices (CPU, int64)
      topk_d:   [K_eff] distances (CPU, float)
    """
    if keys_cpu.numel() == 0:
        raise ValueError("Empty keys_cpu.")
    dists = torch.norm(keys_cpu - query_key_cpu[None, :], dim=1)  # [N]
    K_eff = min(int(K), int(dists.numel()))
    topk_idx = torch.topk(-dists, K_eff).indices
    topk_d = dists[topk_idx]
    return topk_idx, topk_d


def softmax_weights(topk_d: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(-topk_d / max(float(temperature), 1e-8), dim=0)


# =========================
# Configs
# =========================

MeanMethod = Literal["transformer", "knn_blend"]
UncertaintyMethod = Literal["none", "analog"]


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    patience: int = 10
    early_stop: bool = True
    min_delta: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None


@dataclass
class KNNMeanConfig:
    K: int = 5
    temperature: float = 1.0
    alpha: float = 1.0  # controls blend strength for mean="knn_blend": lam = alpha/(avg_dist+alpha)
    beta: float = 1.0   # analog_mode: weight on analog trajectory vs transformer mean (beta>=1 => pure analog)


@dataclass
class PredictConfig:
    mean: MeanMethod = "transformer"
    uncertainty: UncertaintyMethod = "none"
    n_samples: int = 200  # used if uncertainty != "none"

    knn_mean: KNNMeanConfig = field(default_factory=KNNMeanConfig)

    # If you want reproducible ensembles:
    seed: int = 0


@dataclass
class EnsembleConfig:
    """
    User-facing ensemble configuration.

    - n_samples: ensemble size
    - seed:      if None, we store -1 and let the code randomize each call;
                 otherwise, we use a fixed integer seed for reproducibility.

    Ensembles are produced by analog trajectory sampling (mode="analog_mode").
    """
    n_samples: int = 200
    seed: Optional[int] = 0        # None => -1 => "random each call"

    def to_predict_cfg(
        self,
        *,
        mean: MeanMethod,
        knn_mean_cfg: Optional[KNNMeanConfig],
    ) -> PredictConfig:
        """
        Convert to PredictConfig for internal calls. Ensembles use analog sampling.
        """
        return PredictConfig(
            mean=mean,
            uncertainty="analog",
            n_samples=self.n_samples,
            knn_mean=knn_mean_cfg or KNNMeanConfig(),
            seed=-1 if self.seed is None else int(self.seed),
        )


# =========================
# Main experiment class
# =========================

class HybridKNNTransformer:
    """
    Explicit interface for the manuscript's attention-kNN signal forecaster.

    Modes:
      - mode="analog_mode"   : analog retrieval (point + ensemble).
      - mode="forecast_mode" : transformer / knn_blend POINT baselines (no ensemble).

    Stores:
      - model:      trained transformer encoder + forecast head
      - mean_store: train-built (embedding, y) datastore used for analog retrieval and
                    optional knn_blend baseline
    """

    def __init__(
        self,
        feature_columns: List[str],
        target_columns: List[str],
        seq_length: int,
        pred_horizon: int,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
        model_kind: str = "transformer", # deploy a default self attention module here, can be modified
        train_cfg: Optional[TrainConfig] = None,
        device: Optional[str] = None,
        seed: int = 0,
        mode: str = "forecast_mode",             # "forecast_mode" | "analog_mode"
    ):
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        self.seq_length = int(seq_length)
        self.pred_horizon = int(pred_horizon)

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.nhead = int(nhead)
        self.dropout = float(dropout)
        self.model_kind = str(model_kind).lower()

        if mode not in {"forecast_mode", "analog_mode"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

        self.train_cfg = train_cfg or TrainConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.seed = int(seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model: Optional[EncoderWithForecastHead] = None
        self.best_state: Optional[Dict[str, Any]] = None

        # KNN datastore (CPU tensors): (key [H], y [Hzn,out_dim])
        self.mean_store: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._mean_keys: Optional[torch.Tensor] = None

        # bookkeeping
        self._split_info: Dict[str, Any] = {}
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    # ---------- model lifecycle ----------
    def build_model(self) -> nn.Module:
        input_dim = len(self.feature_columns)
        output_dim = len(self.target_columns)

        if self.model_kind != "transformer":
            raise ValueError(
                f"model_kind={self.model_kind!r} is not available in this archival version. "
                f"The manuscript uses the self-attention encoder (model_kind='transformer')."
            )

        self.model = EncoderWithForecastHead(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            pred_horizon=self.pred_horizon,
            num_layers=self.num_layers,
            nhead=self.nhead,
            dropout=self.dropout,
        ).to(self.device)

        return self.model

    # ---------- splitting ----------
    def time_split(
        self,
        df: pd.DataFrame,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        require_full_train_batch: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        idx = compute_time_splits(
            n=len(df),
            seq_length=self.seq_length,
            pred_horizon=self.pred_horizon,
            batch_size=self.train_cfg.batch_size,
            val_size=val_size,
            test_size=test_size,
            val_frac=val_frac,
            test_frac=test_frac,
            require_full_train_batch=require_full_train_batch,
        )

        train_raw = df.iloc[:idx["train_end"]].copy()
        val_raw   = df.iloc[idx["val_start"]:idx["val_end"]].copy()
        test_raw  = df.iloc[idx["test_start"]:idx["test_end"]].copy()

        val_ext   = df.iloc[idx["val_ext_start"]:idx["val_end"]].copy()
        test_ext  = df.iloc[idx["test_ext_start"]:idx["test_end"]].copy()

        self._split_info = idx
        return {"train_raw": train_raw, "val_raw": val_raw, "test_raw": test_raw, "val_ext": val_ext, "test_ext": test_ext}

    def _make_xy(self, df_block: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np, y_np = create_sequences_np(
            df_block,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            seq_length=self.seq_length,
            pred_horizon=self.pred_horizon,
        )
        return _to_tensor(X_np, self.device), _to_tensor(y_np, self.device)

    # ---------- training ----------
    def fit(
        self,
        df: pd.DataFrame,
        *,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        require_full_train_batch: bool = True,
        verbose_every: int = 10,
        loss_name: str = "mse",
    ) -> Dict[str, Any]:
        splits = self.time_split(
            df,
            val_size=val_size,
            test_size=test_size,
            val_frac=val_frac,
            test_frac=test_frac,
            require_full_train_batch=require_full_train_batch,
        )
        train_raw = splits["train_raw"]
        val_ext = splits["val_ext"]

        if self.model is None:
            self.build_model()

        X_train, y_train = self._make_xy(train_raw)
        X_val, y_val = self._make_xy(val_ext)
        has_val = len(X_val) > 0

        dl_train = DataLoader(TensorDataset(X_train, y_train), batch_size=self.train_cfg.batch_size, shuffle=True, drop_last=False)
        dl_val = DataLoader(TensorDataset(X_val, y_val), batch_size=self.train_cfg.batch_size, shuffle=False, drop_last=False) if has_val else None

        criterion = get_loss(loss_name)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        best_val = float("inf")
        no_improve = 0
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.train_cfg.epochs):
            self.model.train()
            se = 0.0
            n_obs = 0

            for xb, yb in dl_train:
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                if self.train_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                optimizer.step()
                se += float(loss.item()) * xb.size(0)
                n_obs += xb.size(0)

            train_loss = se / max(n_obs, 1)
            self.history["train_loss"].append(train_loss)

            if has_val:
                self.model.eval()
                vse = 0.0
                vn = 0
                with torch.no_grad():
                    for xb, yb in dl_val:
                        vloss = criterion(self.model(xb), yb)
                        vse += float(vloss.item()) * xb.size(0)
                        vn += xb.size(0)
                val_loss = vse / max(vn, 1)
            else:
                val_loss = train_loss

            self.history["val_loss"].append(val_loss)

            if verbose_every is not None and (epoch % verbose_every == 0):
                print(f"Epoch {epoch:3d}/{self.train_cfg.epochs} | Train {train_loss:.6f} | Val {val_loss:.6f}")

            if val_loss < best_val - self.train_cfg.min_delta:
                best_val = val_loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if self.train_cfg.early_stop and no_improve >= self.train_cfg.patience:
                    print("Early stopping triggered.")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # Build the datastore from train sequences (analog retrieval + optional knn_blend).
        self.build_mean_store(train_raw)

        return {
            "best_val": float(best_val),
            "split_info": dict(self._split_info),
            "history": self.history,
            "n_train_seq": int(len(X_train)),
            "n_val_seq": int(len(X_val)),
            "mean_store_size": int(len(self.mean_store)),
        }

    def predict(
        self,
        df_or_ext,
        *,
        kind: Literal["windows", "latest"],
        output: Literal["point", "ensemble"],
        mean: MeanMethod = "transformer",
        knn_mean_cfg: Optional[KNNMeanConfig] = None,
        ens_cfg: Optional["EnsembleConfig"] = None,
        return_y: bool = True,         # applies only to kind="windows"
    ) -> Dict[str, Any]:
        """
        Unified prediction API.

        Returns (consistent keys):
          - kind="windows", output="point":    {"pred": torch[N,H,D], "y": torch[N,H,D]} (if return_y)
          - kind="windows", output="ensemble": {"ens": np[E,N,H,D], "point": np[N,H,D], "y": np[N,H,D]} (if return_y)

          - kind="latest", output="point":     {"pred": np[H,D]}
          - kind="latest", output="ensemble":  {"ens": np[E,H,D], "point": np[H,D]}

        Ensembles require mode="analog_mode" (the manuscript method). In forecast_mode only
        point forecasts are available.
        """
        if kind not in {"windows", "latest"}:
            raise ValueError(f"Unknown kind: {kind}")
        if output not in {"point", "ensemble"}:
            raise ValueError(f"Unknown output: {output}")

        # -------- POINT --------
        if output == "point":
            if kind == "windows":
                out = self.predict_windows(
                    df_or_ext,
                    mean=mean,
                    knn_mean_cfg=knn_mean_cfg,
                    return_y=return_y,
                )
                return {"pred": out["pred"], **({"y": out["y"]} if return_y else {})}

            # latest
            pred = self.predict_latest(
                df_or_ext,
                mean=mean,
                knn_mean_cfg=knn_mean_cfg,
            )
            return {"pred": pred}

        # -------- ENSEMBLE --------
        if ens_cfg is None:
            raise ValueError("output='ensemble' requires ens_cfg.")

        if self.mode != "analog_mode":
            raise NotImplementedError(
                "Ensemble generation in this archival version is produced by analog trajectory "
                "sampling (mode='analog_mode'), matching manuscript Section 2.2. forecast_mode "
                "supports point forecasts only."
            )

        p_cfg = ens_cfg.to_predict_cfg(mean=mean, knn_mean_cfg=knn_mean_cfg)

        if kind == "windows":
            out = self.predict_windows_ensemble(
                df_or_ext,
                cfg=p_cfg,
                return_y=return_y,
            )
            ret = {"ens": out["ens"], "point": out["point"]}
            if return_y:
                ret["y"] = out["y"]
            return ret

        # latest ensemble
        ens = self.predict_latest_ensemble(df_or_ext, cfg=p_cfg)  # [E,H,D]
        point = ens.mean(axis=0)                                 # [H,D] cheap point summary
        return {"ens": ens, "point": point}

    # ---------- stores ----------
    @torch.no_grad()
    def build_mean_store(self, train_raw: pd.DataFrame) -> None:
        """
        Build the datastore = (embedding, y) from *train sequences*.
        Stored on CPU for KNN retrieval.
        """
        if self.model is None:
            raise RuntimeError("Model not built.")
        self.model.eval()

        X_train, y_train = self._make_xy(train_raw)
        self.mean_store = []

        for i in range(len(X_train)):
            xw = X_train[i : i + 1]  # [1,L,F] device
            key = self.model.encoder(xw).squeeze(0).detach().cpu()  # [H] CPU
            self.mean_store.append((key, y_train[i].detach().cpu()))

        self._mean_keys = _stack_keys(self.mean_store) if len(self.mean_store) else None

    # ---------- prediction (point) ----------
    @torch.no_grad()
    def predict_windows(
        self,
        df_block_ext: pd.DataFrame,
        *,
        mean: MeanMethod = "transformer",
        knn_mean_cfg: Optional[KNNMeanConfig] = None,
        return_y: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict every available sequence in df_block_ext.

        forecast_mode:
          - mean="transformer": pure transformer forecast
          - mean="knn_blend":   transformer + KNN mean blending (Zhang et al., 2025-style baseline)

        analog_mode:
          - mean argument is ignored; analog-based forecast using encoder embeddings + KNN over
            the datastore. Optional blending with the transformer mean via knn_mean_cfg.beta.

        Returns:
          pred: [N, Hzn, out_dim] on device
          y:    [N, Hzn, out_dim] on device (if return_y)
        """
        if self.model is None:
            raise RuntimeError("Model not trained/built.")

        X, y = self._make_xy(df_block_ext)
        if len(X) == 0:
            out_dim = len(self.target_columns)
            empty = torch.zeros((0, self.pred_horizon, out_dim), device=self.device)
            return {"pred": empty, "y": empty} if return_y else {"pred": empty}

        self.model.eval()
        base = self.model(X)  # [N,H,D]

        # ---- analog_mode ----
        if self.mode == "analog_mode":
            cfg = knn_mean_cfg or KNNMeanConfig()
            analog_pred = self._analog_mean_batch(X, base, cfg)
            return {"pred": analog_pred, "y": y} if return_y else {"pred": analog_pred}

        # ---- forecast_mode ----
        if mean == "transformer":
            return {"pred": base, "y": y} if return_y else {"pred": base}

        if mean != "knn_blend":
            raise ValueError(f"Unknown mean method: {mean}")

        if len(self.mean_store) == 0 or self._mean_keys is None:
            raise RuntimeError("Datastore empty. Train and call build_mean_store().")

        cfg = knn_mean_cfg or KNNMeanConfig()
        blended = self._knn_blend_batch(X, base, cfg)
        return {"pred": blended, "y": y} if return_y else {"pred": blended}

    @torch.no_grad()
    def predict_latest(
        self,
        df: pd.DataFrame,
        *,
        mean: MeanMethod = "transformer",
        knn_mean_cfg: Optional[KNNMeanConfig] = None,
    ) -> np.ndarray:
        """
        Forecast next pred_horizon from the latest seq_length window of df.
        Returns numpy array [pred_horizon, out_dim] on CPU.

        forecast_mode:
          - mean="transformer" or "knn_blend".

        analog_mode:
          - ignores mean and uses analog forecast with optional transformer blending
            via knn_mean_cfg.beta.
        """
        if self.model is None:
            raise RuntimeError("Model not trained/built.")
        if len(df) < self.seq_length:
            raise ValueError(f"Need at least seq_length={self.seq_length} rows for latest prediction.")

        window = df.iloc[-self.seq_length:][self.feature_columns].to_numpy(dtype=np.float32)
        x = _to_tensor(window[None, :, :], self.device)  # [1,L,F]

        self.model.eval()
        base = self.model(x).squeeze(0)  # [H,D]

        # ---- analog_mode ----
        if self.mode == "analog_mode":
            cfg = knn_mean_cfg or KNNMeanConfig()
            analog_pred = self._analog_mean_batch(x, base[None, :, :], cfg)[0]  # [H,D]
            return analog_pred.detach().cpu().numpy()

        # ---- forecast_mode ----
        if mean == "transformer":
            return base.detach().cpu().numpy()

        if mean != "knn_blend":
            raise ValueError(f"Unknown mean method: {mean}")

        cfg = knn_mean_cfg or KNNMeanConfig()
        blended = self._knn_blend_single(x, base, cfg)
        return blended.detach().cpu().numpy()

    # ---------- prediction (ensemble; analog_mode only) ----------
    @torch.no_grad()
    def predict_windows_ensemble(
        self,
        df_block_ext: pd.DataFrame,
        *,
        cfg: PredictConfig,
        return_y: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Produces an ensemble for every available sequence via analog trajectory sampling.

        Returns:
          ens:   [E, N, H, D] float32
          point: [N, H, D] float32 (analog-mean point forecast)
          y:     [N, H, D] float32 (if return_y)
        """
        if cfg.uncertainty == "none":
            raise ValueError("cfg.uncertainty='none' - use predict_windows() for point forecasts.")
        if self.model is None:
            raise RuntimeError("Model not trained/built.")
        if self.mode != "analog_mode":
            raise NotImplementedError(
                "Ensemble generation requires mode='analog_mode' (manuscript Section 2.2)."
            )

        # build X and base forecast
        X, y = self._make_xy(df_block_ext)
        if len(X) == 0:
            raise ValueError("No sequences available in df_block_ext for ensemble prediction.")

        self.model.eval()
        base = self.model(X)  # [N,H,D]

        rng = np.random.default_rng(None if cfg.seed < 0 else int(cfg.seed))
        ens, point_np = self._analog_ensemble_batch(
            X=X,
            base=base,
            cfg=cfg.knn_mean,
            n_samples=cfg.n_samples,
            rng=rng,
        )
        out_dict = {"ens": ens, "point": point_np}
        if return_y:
            out_dict["y"] = y.detach().cpu().numpy().astype(np.float32)
        return out_dict

    @torch.no_grad()
    def predict_latest_ensemble(self, df: pd.DataFrame, cfg: PredictConfig) -> np.ndarray:
        """
        Returns ensemble forecast for the NEXT horizon only via analog trajectory sampling.
        Output: ens_latest [E, H, D]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if cfg.uncertainty == "none":
            raise ValueError("predict_latest_ensemble requires cfg.uncertainty != 'none'.")
        if self.mode != "analog_mode":
            raise NotImplementedError(
                "Ensemble generation requires mode='analog_mode' (manuscript Section 2.2)."
            )
        if len(df) < self.seq_length:
            raise ValueError(f"Need at least seq_length={self.seq_length} rows.")

        # 1) build latest feature window
        X_last = df.iloc[-self.seq_length:][self.feature_columns].to_numpy(dtype=np.float32)
        x = torch.tensor(X_last[None, :, :], dtype=torch.float32, device=self.device)  # [1,L,F]

        self.model.eval()
        base = self.model(x).squeeze(0)  # [H,D]

        E = int(cfg.n_samples)
        H = self.pred_horizon
        D = len(self.target_columns)
        rng = np.random.default_rng(None if cfg.seed < 0 else int(cfg.seed))
        ens = np.empty((E, H, D), dtype=np.float32)

        if len(self.mean_store) == 0 or self._mean_keys is None:
            raise RuntimeError("Datastore empty. Call fit() so build_mean_store(train_raw) is run.")

        beta = float(getattr(cfg.knn_mean, "beta", 1.0))
        base_np = base.detach().cpu().numpy().astype(np.float32)

        qkey = self.model.encoder(x).squeeze(0).detach().cpu()    # [H] CPU
        topk_idx, topk_d = knn_topk(qkey, self._mean_keys, cfg.knn_mean.K)
        w = softmax_weights(topk_d, cfg.knn_mean.temperature).numpy()

        neighbor_y = torch.stack(
            [self.mean_store[int(j)][1] for j in topk_idx],
            dim=0,
        ).numpy().astype(np.float32)  # [K,H,D]

        for e in range(E):
            pick_local = rng.choice(np.arange(len(topk_idx)), p=w)
            traj = neighbor_y[pick_local]  # [H,D]

            if beta >= 1.0:
                ens[e] = traj
            elif beta <= 0.0:
                ens[e] = base_np
            else:
                ens[e] = (1.0 - beta) * base_np + beta * traj

        return ens

    # ---------- internal KNN operations (knn_blend baseline) ----------
    @torch.no_grad()
    def _knn_blend_single(self, query_window: torch.Tensor, base_forecast: torch.Tensor, cfg: KNNMeanConfig) -> torch.Tensor:
        """
        Blend a single forecast [H,D] with KNN neighbors from the datastore.
        """
        self.model.eval()
        qkey = self.model.encoder(query_window).squeeze(0).detach().cpu()  # [H] CPU
        topk_idx, topk_d = knn_topk(qkey, self._mean_keys, cfg.K)
        w = softmax_weights(topk_d, cfg.temperature)  # [K] CPU

        neighbor_y = torch.stack([self.mean_store[int(i)][1] for i in topk_idx], dim=0)  # [K,H,D] CPU
        knn_forecast = torch.sum(w[:, None, None] * neighbor_y, dim=0)  # [H,D] CPU

        avg_d = float(topk_d.mean().item()) if len(topk_d) else float("inf")
        lam = float(cfg.alpha) / (avg_d + float(cfg.alpha)) if np.isfinite(avg_d) else 0.0

        knn_forecast = knn_forecast.to(device=self.device, dtype=base_forecast.dtype)
        return (1 - lam) * base_forecast + lam * knn_forecast

    @torch.no_grad()
    def _knn_blend_batch(self, X: torch.Tensor, base: torch.Tensor, cfg: KNNMeanConfig) -> torch.Tensor:
        """
        Batch version of KNN blending. Loops over N sequences (simple and clear).
        """
        blended = []
        for i in range(len(X)):
            blended.append(self._knn_blend_single(X[i:i+1], base[i], cfg))
        return torch.stack(blended, dim=0)

    # ---------- internal analog operations (manuscript method) ----------
    @torch.no_grad()
    def _analog_mean_batch(
        self,
        X: torch.Tensor,                # [N,L,F] on device
        base: torch.Tensor,             # [N,H,D] transformer forecast on device
        cfg: KNNMeanConfig,
    ) -> torch.Tensor:
        """
        Analog-mode point forecast:

          For each window i:
            - embed state via encoder
            - find K nearest analogs in the datastore
            - take softmax-weighted mean trajectory over neighbor futures
            - optionally blend with transformer base forecast via cfg.beta

        Returns:
          blended_mean: [N,H,D] on device
        """
        if len(self.mean_store) == 0 or self._mean_keys is None:
            raise RuntimeError("Datastore empty. Call fit() so build_mean_store(train_raw) is run.")

        self.model.eval()
        N, H, D = base.shape
        out = torch.empty_like(base)

        beta = float(getattr(cfg, "beta", 1.0))

        for i in range(N):
            xw = X[i : i + 1]  # [1,L,F] device
            qkey = self.model.encoder(xw).squeeze(0).detach().cpu()   # [H] CPU

            topk_idx, topk_d = knn_topk(qkey, self._mean_keys, cfg.K)
            w = softmax_weights(topk_d, cfg.temperature)              # [K] CPU

            neighbor_y = torch.stack(
                [self.mean_store[int(j)][1] for j in topk_idx],
                dim=0,
            )  # [K,H,D] CPU

            analog_mean = torch.sum(w[:, None, None] * neighbor_y, dim=0)  # [H,D] CPU
            analog_mean = analog_mean.to(device=self.device, dtype=base.dtype)

            if beta >= 1.0:
                out[i] = analog_mean
            elif beta <= 0.0:
                out[i] = base[i]
            else:
                out[i] = (1.0 - beta) * base[i] + beta * analog_mean

        return out

    @torch.no_grad()
    def _analog_ensemble_batch(
        self,
        X: torch.Tensor,               # [N,L,F] device
        base: torch.Tensor,            # [N,H,D] device (transformer mean)
        cfg: KNNMeanConfig,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analog-mode ensemble:

          - For each window, find K nearest analogs in embedding space.
          - Use softmax on distance to get sampling weights.
          - For each ensemble member, sample one neighbor trajectory.
          - Optionally blend each sampled trajectory with transformer mean via cfg.beta.

        Returns:
          ens:   [E,N,H,D] float32
          point: [N,H,D]   float32 (analog mean used as point forecast)
        """
        if len(self.mean_store) == 0 or self._mean_keys is None:
            raise RuntimeError("Datastore empty. Call fit() so build_mean_store(train_raw) is run.")

        self.model.eval()
        N, H, D = base.shape
        E = int(n_samples)
        ens = np.empty((E, N, H, D), dtype=np.float32)

        beta = float(getattr(cfg, "beta", 1.0))
        base_np = base.detach().cpu().numpy().astype(np.float32)

        # also compute analog mean once for the point forecast
        analog_mean = self._analog_mean_batch(X, base, cfg)  # [N,H,D] device
        analog_mean_np = analog_mean.detach().cpu().numpy().astype(np.float32)

        for i in range(N):
            xw = X[i : i + 1]  # [1,L,F] device
            qkey = self.model.encoder(xw).squeeze(0).detach().cpu()   # [H] CPU

            topk_idx, topk_d = knn_topk(qkey, self._mean_keys, cfg.K)
            w = softmax_weights(topk_d, cfg.temperature).numpy()      # [K]

            # pre-stack neighbor trajectories
            neighbor_y = torch.stack(
                [self.mean_store[int(j)][1] for j in topk_idx],
                dim=0,
            ).numpy().astype(np.float32)  # [K,H,D]

            for e in range(E):
                pick_local = rng.choice(np.arange(len(topk_idx)), p=w)
                traj = neighbor_y[pick_local]  # [H,D]

                if beta >= 1.0:
                    ens[e, i] = traj
                elif beta <= 0.0:
                    ens[e, i] = base_np[i]
                else:
                    ens[e, i] = (1.0 - beta) * base_np[i] + beta * traj

        return ens, analog_mean_np

    def plot_test_forecasts_facets(
        self,
        df: pd.DataFrame,
        *,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        mean: MeanMethod = "transformer",                 # "transformer" | "knn_blend"
        knn_mean_cfg: Optional[KNNMeanConfig] = None,     # used if mean="knn_blend"
        # --- ensemble PI shading (optional; requires mode="analog_mode") ---
        ensemble_cfg: Optional[Any] = None,               # EnsembleConfig
        pi_lo: float = 0.10,
        pi_hi: float = 0.90,
        pi_color: str = "lightgray",
        pi_alpha: float = 0.5,
        # --- plotting controls ---
        history_window: Optional[int] = None,
        history_frac: Optional[float] = None,
        mode: str = "rolling_1step",
        title: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        savepath: Optional[str] = None,
        show: bool = True,
    ):
        """
        Faceted plot: one subplot per target column.

        If an internal test block exists (test_raw/test_ext non-empty):
          - Plots historical truth and *test* forecasts (rolling 1-step and/or first trajectory).
          - Optionally shades ensemble prediction interval between pi_lo/pi_hi in light gray.

        If no internal test block exists:
          - Plots historical truth and a single forecast trajectory starting *after* df.index[-1],
            with optional ensemble PI shading from the latest forecast.

        Note: ensemble PI shading (ensemble_cfg) requires mode="analog_mode".

        mode (only used when test exists):
          - "rolling_1step": overlays rolling one-step-ahead predictions across the test period (h=0)
          - "first_trajectory": overlays the first test trajectory (full pred_horizon) at test start
          - "both": does both overlays
        """
        if self.model is None:
            raise RuntimeError("Model not trained/built. Call fit() first.")

        n = len(df)

        # resolve val/test sizes if not provided explicitly
        if val_size is None:
            val_size = int(round(n * val_frac))
        if test_size is None:
            test_size = int(round(n * test_frac))

        # resolve history window for plotting
        if history_window is None and history_frac is not None:
            history_window = int(round(n * history_frac))

        # compute internal splits
        splits = self.time_split(df, val_size=val_size, test_size=test_size)
        test_raw = splits["test_raw"]
        test_ext = splits["test_ext"]

        has_test = (len(test_raw) > 0) and (len(test_ext) > 0)

        # optionally crop history for plotting
        if history_window is not None and history_window > 0:
            df_plot = df.iloc[-history_window:].copy()
        else:
            df_plot = df

        n_targets = len(self.target_columns)
        if figsize is None:
            figsize = (14, 3.2 * n_targets)

        fig, axes = plt.subplots(n_targets, 1, figsize=figsize, sharex=True)
        if n_targets == 1:
            axes = [axes]

        # ==========================================================
        # CASE A: internal test exists -> original behavior
        # ==========================================================
        if has_test:
            # ----- point predictions over test_ext windows -----
            out = self.predict_windows(test_ext, mean=mean, knn_mean_cfg=knn_mean_cfg, return_y=True)
            y_true = out["y"].detach().cpu().numpy()     # [Nseq, H, D]
            y_pred = out["pred"].detach().cpu().numpy()  # [Nseq, H, D]

            # ----- optional ensemble for PI shading -----
            ens = None
            if ensemble_cfg is not None:
                out_ens = self.predict(
                    test_ext,
                    kind="windows",
                    output="ensemble",
                    mean=mean,
                    knn_mean_cfg=knn_mean_cfg,
                    ens_cfg=ensemble_cfg,
                )
                ens = out_ens["ens"]   # [E, Nseq, H, D]

            # index handling
            idx_test = test_raw.index
            idx_test_ext = test_ext.index

            # helper: rolling 1-step-ahead aligned series over test_raw
            rolling_pred = {col: pd.Series(index=idx_test, dtype=float) for col in self.target_columns}
            rolling_lo = (
                {col: pd.Series(index=idx_test, dtype=float) for col in self.target_columns}
                if ens is not None else None
            )
            rolling_hi = (
                {col: pd.Series(index=idx_test, dtype=float) for col in self.target_columns}
                if ens is not None else None
            )

            # Each window i starts at idx_test_ext[i + seq_length]. For rolling_1step, use h=0.
            for i in range(y_pred.shape[0]):
                start_pos = i + self.seq_length
                if start_pos < 0 or start_pos >= len(idx_test_ext):
                    continue
                t0 = idx_test_ext[start_pos]  # forecast start timestamp
                if t0 not in idx_test:
                    continue

                for j, col in enumerate(self.target_columns):
                    rolling_pred[col].loc[t0] = float(y_pred[i, 0, j])

                    if ens is not None:
                        # quantiles across ensemble members at this window i, horizon 0, target j
                        rolling_lo[col].loc[t0] = float(np.quantile(ens[:, i, 0, j], pi_lo))
                        rolling_hi[col].loc[t0] = float(np.quantile(ens[:, i, 0, j], pi_hi))

            # helper: first full trajectory at the start of test_raw (if possible)
            first_traj = None
            if y_pred.shape[0] > 0 and len(idx_test) > 0:
                t_test0 = idx_test[0]
                first_i = None
                for i in range(y_pred.shape[0]):
                    start_pos = i + self.seq_length
                    if 0 <= start_pos < len(idx_test_ext) and idx_test_ext[start_pos] == t_test0:
                        first_i = i
                        break

                if first_i is not None:
                    traj_times = []
                    for h in range(self.pred_horizon):
                        pos = first_i + self.seq_length + h
                        if pos < len(idx_test_ext):
                            traj_times.append(idx_test_ext[pos])
                    traj_times = pd.Index(traj_times)
                    H_eff = len(traj_times)

                    first_traj = {
                        "times": traj_times,
                        "pred": y_pred[first_i, :H_eff, :],  # [H_eff, D]
                    }

                    if ens is not None:
                        # ens[:, first_i, :H_eff, :] -> [E, H_eff, D]
                        first_traj["lo"] = np.quantile(ens[:, first_i, :H_eff, :], pi_lo, axis=0)  # [H_eff, D]
                        first_traj["hi"] = np.quantile(ens[:, first_i, :H_eff, :], pi_hi, axis=0)  # [H_eff, D]

            # ----- plot -----
            for ax, col in zip(axes, self.target_columns):
                # historical truth (full df_plot)
                ax.plot(df_plot.index, df_plot[col].values, linewidth=1)

                # overlays: rolling 1-step
                if mode in ("rolling_1step", "both"):
                    if ens is not None:
                        ax.fill_between(
                            rolling_pred[col].index,
                            rolling_lo[col].values,
                            rolling_hi[col].values,
                            color=pi_color,
                            alpha=pi_alpha,
                            linewidth=0.0,
                        )
                    ax.plot(rolling_pred[col].index, rolling_pred[col].values, linewidth=1)

                # overlays: first trajectory
                if mode in ("first_trajectory", "both") and first_traj is not None:
                    j = self.target_columns.index(col)
                    if ens is not None and "lo" in first_traj and "hi" in first_traj:
                        ax.fill_between(
                            first_traj["times"],
                            first_traj["lo"][:, j],
                            first_traj["hi"][:, j],
                            color=pi_color,
                            alpha=pi_alpha,
                            linewidth=0.0,
                        )
                    ax.plot(first_traj["times"], first_traj["pred"][:, j], linewidth=2)

                ax.set_title(col)

            if title is None:
                title = f"Test forecasts (mean={mean}, PI={'10-90' if ens is not None else 'off'})"
            fig.suptitle(title)

        # ==========================================================
        # CASE B: no internal test -> forecast-only from end of df
        # ==========================================================
        else:
            # point forecast for next pred_horizon from end of df
            if ensemble_cfg is not None:
                out_latest = self.predict(
                    df,
                    kind="latest",
                    output="ensemble",
                    mean=mean,
                    knn_mean_cfg=knn_mean_cfg,
                    ens_cfg=ensemble_cfg,
                )
                latest_point = out_latest["point"]  # [H, D]
                ens_latest = out_latest["ens"]      # [E, H, D]
            else:
                out_latest = self.predict(
                    df,
                    kind="latest",
                    output="point",
                    mean=mean,
                    knn_mean_cfg=knn_mean_cfg,
                )
                latest_point = out_latest["pred"]   # [H, D]
                ens_latest = None

            # build forecast index for horizons 0..H-1 (one step after last df.index)
            idx = df.index
            freq = getattr(idx, "freq", None) or pd.infer_freq(idx)
            if freq is not None:
                forecast_idx = pd.date_range(
                    start=idx[-1],
                    periods=self.pred_horizon + 1,
                    freq=freq
                )[1:]
            else:
                # fallback: median delta
                delta = idx.to_series().diff().median()
                forecast_idx = pd.Index(
                    [idx[-1] + (h + 1) * delta for h in range(self.pred_horizon)]
                )

            # ----- plot -----
            for j, (ax, col) in enumerate(zip(axes, self.target_columns)):
                # historical truth
                ax.plot(df_plot.index, df_plot[col].values, linewidth=1)

                # ensemble PI shading if available
                if ens_latest is not None:
                    lo = np.quantile(ens_latest[:, :, j], pi_lo, axis=0)  # [H]
                    hi = np.quantile(ens_latest[:, :, j], pi_hi, axis=0)  # [H]
                    ax.fill_between(
                        forecast_idx,
                        lo,
                        hi,
                        color=pi_color,
                        alpha=pi_alpha,
                        linewidth=0.0,
                    )

                # point forecast
                ax.plot(forecast_idx, latest_point[:, j], linewidth=2)
                ax.set_title(col)

            if title is None:
                title = f"Forecast from end of series (mean={mean}, PI={'10-90' if ens_latest is not None else 'off'})"
            fig.suptitle(title)

        axes[-1].set_xlabel("Time")
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axes


# =========================
# Assessment / diagnostics
# =========================

class EnsembleAssessment:
    """
    Small assessment helper suite for ensembles.

    Assumes:
      ens: [E,N,H,D]
      y:   [N,H,D]
    """
    @staticmethod
    def ensemble_spread_summary(
        ens: np.ndarray,
        target_names: Sequence[str],
        horizons: Optional[Sequence[int]] = None,
        quantiles: Tuple[float, float, float] = (0.5, 0.9, 0.99),
    ) -> pd.DataFrame:
        E, N, H, D = ens.shape
        if horizons is None:
            horizons = list(range(H))

        rows = []
        std = np.std(ens, axis=0)  # [N,H,D]
        for h in horizons:
            for j, name in enumerate(target_names):
                x = std[:, h, j]
                rows.append({
                    "h": int(h),
                    "target": str(name),
                    "mean_ens_std": float(np.mean(x)),
                    "median_ens_std": float(np.median(x)),
                    "p90_ens_std": float(np.quantile(x, 0.9)),
                    "p99_ens_std": float(np.quantile(x, 0.99)),
                    "q50_ens_std": float(np.quantile(x, quantiles[0])),
                    "q90_ens_std": float(np.quantile(x, quantiles[1])),
                    "q99_ens_std": float(np.quantile(x, quantiles[2])),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def coverage(
        ens: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.1,
    ) -> np.ndarray:
        """
        Coverage of (alpha/2, 1-alpha/2) central prediction interval.

        Returns: [H,D] coverage frequency across N.
        """
        lo = np.quantile(ens, alpha / 2, axis=0)          # [N,H,D]
        hi = np.quantile(ens, 1 - alpha / 2, axis=0)      # [N,H,D]
        inside = (y >= lo) & (y <= hi)                    # [N,H,D]
        return inside.mean(axis=0)                        # [H,D]

    @staticmethod
    def variance_ratio(
        ens: np.ndarray,
        y: np.ndarray,
        *,
        point: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Variance ratio per horizon and target:
          VR = mean(ensemble variance) / mean(squared error of ensemble mean)

        Returns: [H,D]
        """
        ens_var = np.var(ens, axis=0)           # [N,H,D]
        ens_mean = np.mean(ens, axis=0) if point is None else point  # [N,H,D]
        mse = (y - ens_mean) ** 2               # [N,H,D]
        num = np.mean(ens_var, axis=0)          # [H,D]
        den = np.mean(mse, axis=0) + 1e-12      # [H,D]
        return num / den

    @staticmethod
    def qq_plot_from_ensemble(
        ens: np.ndarray,
        y: np.ndarray,
        *,
        target_j: int,
        horizon_h: int,
        mode: Literal["median", "mean", "random_draw"] = "median",
        n_quantiles: int = 100,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        seed: Optional[int] = 0,
    ) -> plt.Axes:
        """
        QQ plot of predicted distribution vs observed for a fixed (h, j).
        Predicted samples are either ensemble median/mean per time, or random draw per time.

        Parameters
        ----------
        ens : [E, N, H, D]
        y   : [N, H, D]
        target_j : which target index to plot
        horizon_h : which horizon index to plot
        mode : "median" | "mean" | "random_draw"
        seed : random seed for random_draw mode. If None, use nondeterministic RNG.
        """
        E, N, H, D = ens.shape
        if horizon_h < 0 or horizon_h >= H:
            raise ValueError(f"horizon_h out of range: {horizon_h}")
        if target_j < 0 or target_j >= D:
            raise ValueError(f"target_j out of range: {target_j}")

        if mode == "median":
            pred = np.median(ens[:, :, horizon_h, target_j], axis=0)  # [N]
        elif mode == "mean":
            pred = np.mean(ens[:, :, horizon_h, target_j], axis=0)
        elif mode == "random_draw":
            rng = np.random.default_rng(seed)
            picks = rng.integers(0, E, size=N)
            pred = ens[picks, np.arange(N), horizon_h, target_j]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        obs = y[:, horizon_h, target_j]

        # empirical quantiles
        qs = np.linspace(0.01, 0.99, n_quantiles)
        q_pred = np.quantile(pred, qs)
        q_obs = np.quantile(obs, qs)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))

        ax.plot(q_obs, q_pred, linewidth=1.5)
        lo = min(q_obs.min(), q_pred.min())
        hi = max(q_obs.max(), q_pred.max())
        ax.plot([lo, hi], [lo, hi], linewidth=1.0, linestyle="--")
        ax.set_xlabel("Observed quantiles")
        ax.set_ylabel("Predicted quantiles")
        if title is None:
            title = f"QQ plot (h={horizon_h}, j={target_j}, mode={mode})"
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        return ax


__all__ = [
    "TrainConfig",
    "KNNMeanConfig",
    "PredictConfig",
    "EnsembleConfig",
    "HybridKNNTransformer",
    "EnsembleAssessment",
    "MeanMethod",
    "UncertaintyMethod",
]
