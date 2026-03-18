"""
Phase 2 – Generate deterministic predictions with the trained FEDformer.

Loads checkpoints/fedformer_best.pth, runs inference on data/test.pkl,
and saves results to results/fedformer_forecasts.pkl.
Also runs on train & val sets so that those forecasts can condition the SDCDM.
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.FEDformer import Model as FEDformer
from scripts.train_fedformer import Configs, WindDataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_split(model, loader, cfg, device):
    """Return (predictions, targets) both shape (N, pred_len)."""
    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x      = batch_x.float().to(device)
            batch_y      = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros(
                batch_y.size(0), cfg.pred_len, batch_y.size(2)
            ).float().to(device)
            dec_inp = torch.cat(
                [batch_y[:, :cfg.label_len, :], dec_inp], dim=1
            )

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # slice Wind column from multi-variate output
            outputs = outputs[:, :, cfg.target_idx:cfg.target_idx+1]
            target  = batch_y[:, -cfg.pred_len:, cfg.target_idx:cfg.target_idx+1]

            preds.append(outputs.cpu().numpy())
            truths.append(target.cpu().numpy())

    preds  = np.concatenate(preds,  axis=0)   # (N, pred_len, 1)
    truths = np.concatenate(truths, axis=0)
    return preds[..., 0], truths[..., 0]      # (N, pred_len)


def main():
    device   = get_device()
    cfg      = Configs()
    data_dir = os.path.join(ROOT_DIR, "data")
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
    res_dir  = os.path.join(ROOT_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)

    # ── load checkpoint ──────────────────────────────────────────────────
    ckpt_path = os.path.join(ckpt_dir, "fedformer_best.pth")
    ckpt      = torch.load(ckpt_path, map_location=device)
    model     = FEDformer(cfg).float().to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.6f})")

    # ── also load the scaler (for inverse transform later) ───────────────
    with open(os.path.join(data_dir, "train.pkl"), "rb") as fh:
        train_meta = pickle.load(fh)
    scaler = train_meta["scaler"]

    forecasts = {}
    for split in ("train", "val", "test"):
        ds     = WindDataset(
            os.path.join(data_dir, f"{split}.pkl"),
            cfg.seq_len, cfg.label_len, cfg.pred_len
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        preds, truths = predict_split(model, loader, cfg, device)

        # inverse-transform to original scale
        target_idx   = cfg.target_idx
        dummy_preds  = np.zeros((preds.shape[0] * preds.shape[1],
                                 len(train_meta["feat_cols"])), dtype=np.float32)
        dummy_truths = dummy_preds.copy()
        dummy_preds[:, target_idx]  = preds.reshape(-1)
        dummy_truths[:, target_idx] = truths.reshape(-1)

        inv_preds  = scaler.inverse_transform(dummy_preds)[:, target_idx].reshape(preds.shape)
        inv_truths = scaler.inverse_transform(dummy_truths)[:, target_idx].reshape(truths.shape)

        forecasts[split] = {
            "preds_scaled":    preds,
            "truths_scaled":   truths,
            "preds_original":  inv_preds,
            "truths_original": inv_truths,
        }
        print(f"  {split}: {preds.shape}")

    out_path = os.path.join(res_dir, "fedformer_forecasts.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(forecasts, fh)
    print(f"\nSaved forecasts → {out_path}")


if __name__ == "__main__":
    main()
