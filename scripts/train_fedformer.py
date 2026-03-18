"""
Phase 2 – Train FEDformer on my_fed_data.csv (Wind column).

Hyper-parameters are set according to Table 2 of the paper:
  seq_len=24, label_len=12, pred_len=24 (hourly data, 1-day ahead)
Best model saved to checkpoints/fedformer_best.pth.
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# ─── project root on path ────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.FEDformer import Model as FEDformer

# ─────────────────── hyper-parameters (paper Table 2) ────────────────────
class Configs:
    # dataset
    target_idx  = 2           # index of Wind in [Load, Solar, Wind]
    enc_in      = 3           # number of input features
    dec_in      = 3
    c_out       = 3           # FEDformer outputs all features; we slice Wind later

    # sequence lengths
    seq_len     = 24          # look-back window  (1 day)
    label_len   = 12          # decoder start token
    pred_len    = 24          # forecast horizon  (1 day)

    # model architecture
    version       = "Fourier"   # Fourier variant (faster, good accuracy)
    mode_select   = "random"
    modes         = 32
    d_model       = 512
    n_heads       = 8
    e_layers      = 2
    d_layers      = 1
    d_ff          = 2048
    moving_avg    = [12, 24]
    L             = 1
    base          = "legendre"
    cross_activation = "tanh"
    dropout       = 0.05
    embed         = "timeF"
    freq          = "h"
    activation    = "gelu"
    output_attention = False
    wavelet       = 0

    # NOTE: c_out == enc_in (multivariate → multivariate) per FEDformer design.
    # We slice the Wind column (target_idx) from the output for loss computation.

    # training
    learning_rate = 1e-4
    batch_size    = 32
    num_epochs    = 20
    patience      = 5          # early stopping


# ──────────────── Dataset ────────────────────────────────────────────────
class WindDataset(Dataset):
    """Sliding-window dataset built from a preprocessed .pkl split."""

    def __init__(self, pkl_path: str, seq_len: int, label_len: int, pred_len: int):
        with open(pkl_path, "rb") as fh:
            split = pickle.load(fh)
        self.data       = split["data"].astype(np.float32)      # (N, F)
        self.data_stamp = split["data_stamp"].astype(np.float32)
        self.target_idx = split["target_idx"]
        self.seq_len    = seq_len
        self.label_len  = label_len
        self.pred_len   = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_end   = idx + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = self.data[idx:s_end]               # (seq_len, F)
        seq_y      = self.data[r_begin:r_end]            # (label+pred, F)
        seq_x_mark = self.data_stamp[idx:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark


# ──────────────── training utilities ─────────────────────────────────────
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Configs) -> FEDformer:
    return FEDformer(cfg).float()


def run_epoch(model, loader, optimizer, criterion, cfg, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x      = batch_x.float().to(device)
            batch_y      = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input: zeros for the forecast horizon
            dec_inp = torch.zeros(
                batch_y.size(0), cfg.pred_len, batch_y.size(2)
            ).float().to(device)
            dec_inp = torch.cat(
                [batch_y[:, :cfg.label_len, :], dec_inp], dim=1
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # slice Wind column from multi-variate output, forecast portion only
            target     = batch_y[:, -cfg.pred_len:, cfg.target_idx:cfg.target_idx+1]
            out_wind   = outputs[:, :, cfg.target_idx:cfg.target_idx+1]
            loss       = criterion(out_wind, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train(cfg: Configs, data_dir: str, ckpt_dir: str):
    device = get_device()
    print(f"Device: {device}")

    # ── datasets ─────────────────────────────────────────────────────────
    train_ds = WindDataset(
        os.path.join(data_dir, "train.pkl"),
        cfg.seq_len, cfg.label_len, cfg.pred_len
    )
    val_ds = WindDataset(
        os.path.join(data_dir, "val.pkl"),
        cfg.seq_len, cfg.label_len, cfg.pred_len
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── model ─────────────────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_cnt  = 0
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "fedformer_best.pth")

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, cfg, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, criterion, cfg, device, train=False)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{cfg.num_epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "cfg": cfg.__dict__}, ckpt_path)
            print(f"  ✓ Best model saved  (val_loss={best_val_loss:.6f})")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best checkpoint: {ckpt_path}")
    return ckpt_path


# ──────────────── entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg      = Configs()
    data_dir = os.path.join(ROOT_DIR, "data")
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")

    train(cfg, data_dir, ckpt_dir)
