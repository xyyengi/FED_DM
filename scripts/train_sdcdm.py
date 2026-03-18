"""
Phase 3 – Train SDCDM (Seasonally Conditional Decomposed Diffusion Model).

Conditions the diffusion model on STL-decomposed FEDformer forecasts.
Trains on the training split and saves the best model to
checkpoints/sdcdm_best.pth.
"""

import os
import sys
import time
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import STL

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.sdcdm import SDCDMNet, SDCDMDenoiser

# ─────────────────── hyper-parameters (paper Table 2) ────────────────────

PRED_LEN       = 24        # same as FEDformer
SIGMA_MIN      = 0.002
SIGMA_MAX      = 80.0
SIGMA_DATA     = 1.0
RHO            = 7.0       # Karras noise schedule exponent

BATCH_SIZE     = 64
NUM_EPOCHS     = 30
LEARNING_RATE  = 2e-4
PATIENCE       = 5

# network
BASE_CHANNELS  = 64
CHANNEL_MULTS  = (1, 2, 4)
SIGMA_EMB_DIM  = 128

# number of diffusion steps at inference
NUM_STEPS      = 50

# ──────────────── STL decomposition helper ───────────────────────────────

def stl_decompose(series: np.ndarray, period: int = 24) -> tuple:
    """
    Decompose a 1-D time series into (trend, seasonal, residual) via STL.
    series: (L,) float array
    Returns three arrays of the same length.
    """
    if len(series) < 2 * period:
        # Series is too short for STL (requires at least 2 full periods).
        # Fallback: treat the mean as trend, assume no seasonality, and
        # let the residual capture all variation around the mean.
        mean = np.mean(series)
        return (np.full_like(series, mean),
                np.zeros_like(series),
                series - mean)
    stl  = STL(series, period=period, robust=True)
    res  = stl.fit()
    return res.trend, res.seasonal, res.resid


def decompose_forecasts(forecasts: np.ndarray, period: int = 24) -> tuple:
    """
    forecasts: (N, pred_len)
    Returns (T, S, R) each (N, pred_len).
    """
    N, L = forecasts.shape
    T = np.zeros_like(forecasts)
    S = np.zeros_like(forecasts)
    R = np.zeros_like(forecasts)
    for i in range(N):
        T[i], S[i], R[i] = stl_decompose(forecasts[i], period=min(period, L))
    return T, S, R


# ──────────────── Dataset ────────────────────────────────────────────────

class SDCDMDataset(Dataset):
    """
    Pairs true wind-power sequences with STL-conditioned FEDformer forecasts.
    """

    def __init__(self, truths: np.ndarray, preds: np.ndarray):
        """
        truths: (N, pred_len) – ground truth (scaled)
        preds:  (N, pred_len) – FEDformer predictions (scaled)
        """
        T, S, R = decompose_forecasts(preds)

        self.x0    = torch.from_numpy(truths.astype(np.float32))  # (N, L)
        self.cond_T = torch.from_numpy(T.astype(np.float32))
        self.cond_S = torch.from_numpy(S.astype(np.float32))
        self.cond_R = torch.from_numpy(R.astype(np.float32))

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return self.x0[idx], self.cond_T[idx], self.cond_S[idx], self.cond_R[idx]


# ──────────────── Karras noise schedule ─────────────────────────────────

def sample_sigma(batch_size: int, device) -> torch.Tensor:
    """Log-uniform noise level sampling (Karras et al. 2022 §5)."""
    log_min = math.log(SIGMA_MIN)
    log_max = math.log(SIGMA_MAX)
    return torch.exp(
        torch.rand(batch_size, device=device) * (log_max - log_min) + log_min
    )


def karras_sigmas(n_steps: int, device) -> torch.Tensor:
    """Deterministic Karras noise schedule for inference."""
    ramp        = torch.linspace(0, 1, n_steps, device=device)
    inv_rho_min = SIGMA_MIN ** (1 / RHO)
    inv_rho_max = SIGMA_MAX ** (1 / RHO)
    sigmas      = (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** RHO
    return torch.cat([sigmas, sigmas.new_zeros(1)])   # append 0


# ──────────────── training ────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x0, cond_T, cond_S, cond_R in loader:
            x0     = x0.unsqueeze(1).to(device)      # (B, 1, L)
            cond_T = cond_T.unsqueeze(1).to(device)
            cond_S = cond_S.unsqueeze(1).to(device)
            cond_R = cond_R.unsqueeze(1).to(device)

            sigma  = sample_sigma(x0.size(0), device)   # (B,)
            noise  = torch.randn_like(x0)

            losses = model.loss(x0, noise, sigma, cond_T, cond_S, cond_R)

            # loss-weight: 1 / σ² · σ_data² (Karras eq. 7)
            weight = (SIGMA_DATA ** 2 + sigma ** 2) / (SIGMA_DATA * sigma) ** 2
            loss   = (losses * weight).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train(data_dir: str, res_dir: str, ckpt_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load FEDformer forecasts ─────────────────────────────────────────
    fc_path = os.path.join(res_dir, "fedformer_forecasts.pkl")
    with open(fc_path, "rb") as fh:
        fc = pickle.load(fh)

    train_truths = fc["train"]["truths_scaled"]   # (N, pred_len)
    train_preds  = fc["train"]["preds_scaled"]
    val_truths   = fc["val"]["truths_scaled"]
    val_preds    = fc["val"]["preds_scaled"]

    print(f"Train samples: {len(train_truths)}  Val samples: {len(val_truths)}")
    print("Computing STL decomposition for conditioning (this may take a moment)…")

    train_ds = SDCDMDataset(train_truths, train_preds)
    val_ds   = SDCDMDataset(val_truths,   val_preds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ── build model ───────────────────────────────────────────────────────
    net   = SDCDMNet(
        sigma_data    = SIGMA_DATA,
        base_channels = BASE_CHANNELS,
        channel_mults = CHANNEL_MULTS,
        sigma_emb_dim = SIGMA_EMB_DIM,
        cond_channels = 3,
        input_channels= 1,
    ).to(device)
    model = SDCDMDenoiser(net, sigma_data=SIGMA_DATA).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SDCDM parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val = float("inf")
    patience_cnt = 0
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "sdcdm_best.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0         = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "hparams": dict(
                    sigma_data    = SIGMA_DATA,
                    base_channels = BASE_CHANNELS,
                    channel_mults = CHANNEL_MULTS,
                    sigma_emb_dim = SIGMA_EMB_DIM,
                    pred_len      = PRED_LEN,
                ),
            }, ckpt_path)
            print(f"  ✓ Best model saved  (val_loss={best_val:.6f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nSDCDM training complete. Best checkpoint: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    data_dir = os.path.join(ROOT_DIR, "data")
    res_dir  = os.path.join(ROOT_DIR, "results")
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
    train(data_dir, res_dir, ckpt_dir)
