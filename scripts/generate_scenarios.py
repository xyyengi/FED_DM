"""
Phase 4 – Generate probabilistic wind-power scenarios with SDCDM.

Loads:
  - checkpoints/sdcdm_best.pth
  - results/fedformer_forecasts.pkl  (test split)

For each test sample generates N_SCENARIOS Monte-Carlo samples via
ancestral sampling (DDPM-style reverse diffusion with the Karras schedule).

Saves results/generated_scenarios.pkl:
  {
    "scenarios":        (N_test, N_SCENARIOS, pred_len)  – scaled
    "scenarios_orig":   (N_test, N_SCENARIOS, pred_len)  – MW
    "truths_scaled":    (N_test, pred_len)
    "truths_original":  (N_test, pred_len)
    "preds_scaled":     (N_test, pred_len)   FEDformer
    "preds_original":   (N_test, pred_len)
  }
"""

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.sdcdm import SDCDMNet, SDCDMDenoiser
from scripts.train_sdcdm import (
    decompose_forecasts, karras_sigmas,
    SIGMA_MIN, SIGMA_MAX, SIGMA_DATA, RHO, PRED_LEN, NUM_STEPS,
    BASE_CHANNELS, CHANNEL_MULTS, SIGMA_EMB_DIM,
)

N_SCENARIOS = 100   # paper uses 100 scenarios per test point
BATCH_SCEN  = 20    # how many scenarios to generate in one forward pass

# ──────────────── Karras ancestral (DDIM-like) sampler ───────────────────

@torch.no_grad()
def sample_scenarios(
    model     : SDCDMDenoiser,
    cond_T    : torch.Tensor,    # (1, 1, L) or (S, 1, L)
    cond_S    : torch.Tensor,
    cond_R    : torch.Tensor,
    n_samples : int,
    device,
) -> torch.Tensor:
    """
    Generate n_samples scenarios given one conditioning triple (T, S, R).
    Returns shape (n_samples, 1, L).
    """
    L      = cond_T.shape[-1]
    sigmas = karras_sigmas(NUM_STEPS, device)    # (steps+1,)

    # repeat conditioning for batch
    cT = cond_T.expand(n_samples, -1, -1)
    cS = cond_S.expand(n_samples, -1, -1)
    cR = cond_R.expand(n_samples, -1, -1)

    x = torch.randn(n_samples, 1, L, device=device) * sigmas[0]

    for i in range(len(sigmas) - 1):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]

        sig_vec = sigma.expand(n_samples)
        x0_hat  = model(x, sig_vec, cT, cS, cR)

        # first-order Euler step (Karras et al. Algorithm 1)
        d = (x - x0_hat) / sigma
        if sigma_next == 0:
            x = x0_hat
        else:
            # Stochastic ancestral sampling (Karras et al. Algorithm 2):
            # inject noise proportional to sigma_up, step down to sigma_down.
            variance_preserving_sigma = (
                float(sigma_next) ** 2
                * (float(sigma) ** 2 - float(sigma_next) ** 2)
                / float(sigma) ** 2
            ) ** 0.5
            sigma_up   = min(float(sigma_next), variance_preserving_sigma)
            sigma_down = (float(sigma_next) ** 2 - sigma_up ** 2) ** 0.5
            dt    = sigma_down - float(sigma)
            x     = x + d * dt
            x     = x + torch.randn_like(x) * sigma_up

    return x   # (n_samples, 1, L)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load FEDformer test forecasts ────────────────────────────────────
    res_dir = os.path.join(ROOT_DIR, "results")
    fc_path = os.path.join(res_dir, "fedformer_forecasts.pkl")
    with open(fc_path, "rb") as fh:
        fc = pickle.load(fh)

    test_preds  = fc["test"]["preds_scaled"]    # (N, pred_len)
    test_truths = fc["test"]["truths_scaled"]
    test_preds_orig  = fc["test"]["preds_original"]
    test_truths_orig = fc["test"]["truths_original"]

    print(f"Test samples: {len(test_preds)}")

    # ── load SDCDM ────────────────────────────────────────────────────────
    ckpt_dir  = os.path.join(ROOT_DIR, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "sdcdm_best.pth")
    ckpt      = torch.load(ckpt_path, map_location=device)
    hp        = ckpt["hparams"]

    net   = SDCDMNet(
        sigma_data    = hp["sigma_data"],
        base_channels = hp["base_channels"],
        channel_mults = tuple(hp["channel_mults"]),
        sigma_emb_dim = hp["sigma_emb_dim"],
        cond_channels = 3,
        input_channels= 1,
    )
    model = SDCDMDenoiser(net, sigma_data=hp["sigma_data"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded SDCDM from epoch {ckpt['epoch']}")

    # ── STL conditioning ─────────────────────────────────────────────────
    print("Computing STL decomposition for test forecasts…")
    T, S, R = decompose_forecasts(test_preds)   # each (N, pred_len)

    # ── generate scenarios ────────────────────────────────────────────────
    N    = len(test_preds)
    all_scenarios = np.zeros((N, N_SCENARIOS, PRED_LEN), dtype=np.float32)

    print(f"Generating {N_SCENARIOS} scenarios for {N} test points…")
    for i in range(N):
        cT = torch.from_numpy(T[i]).float().to(device).view(1, 1, PRED_LEN)
        cS = torch.from_numpy(S[i]).float().to(device).view(1, 1, PRED_LEN)
        cR = torch.from_numpy(R[i]).float().to(device).view(1, 1, PRED_LEN)

        scen_parts = []
        remaining  = N_SCENARIOS
        while remaining > 0:
            batch_n  = min(BATCH_SCEN, remaining)
            samples  = sample_scenarios(model, cT, cS, cR, batch_n, device)
            scen_parts.append(samples.squeeze(1).cpu().numpy())   # (batch_n, L)
            remaining -= batch_n

        all_scenarios[i] = np.concatenate(scen_parts, axis=0)[:N_SCENARIOS]

        if (i + 1) % 100 == 0 or i == N - 1:
            print(f"  {i+1}/{N} done")

    # ── inverse-transform to MW ───────────────────────────────────────────
    with open(os.path.join(ROOT_DIR, "data", "train.pkl"), "rb") as fh:
        train_meta = pickle.load(fh)
    scaler     = train_meta["scaler"]
    target_idx = train_meta["target_idx"]
    n_feats    = len(train_meta["feat_cols"])

    def inv_transform(arr2d: np.ndarray) -> np.ndarray:
        """arr2d: (M, L) → MW"""
        M, L    = arr2d.shape
        dummy   = np.zeros((M * L, n_feats), dtype=np.float32)
        dummy[:, target_idx] = arr2d.reshape(-1)
        inv     = scaler.inverse_transform(dummy)[:, target_idx]
        return inv.reshape(M, L)

    all_scenarios_orig = np.stack(
        [inv_transform(all_scenarios[i]) for i in range(N)], axis=0
    )   # (N, S, L)

    # ── save ──────────────────────────────────────────────────────────────
    out = {
        "scenarios":        all_scenarios,
        "scenarios_orig":   all_scenarios_orig,
        "truths_scaled":    test_truths,
        "truths_original":  test_truths_orig,
        "preds_scaled":     test_preds,
        "preds_original":   test_preds_orig,
    }
    out_path = os.path.join(res_dir, "generated_scenarios.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"\nSaved scenarios → {out_path}")
    print(f"  scenarios shape : {all_scenarios.shape}")


if __name__ == "__main__":
    main()
