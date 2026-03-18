"""
Phase 4 – Evaluate the two-stage scenario generation system.

Computes and prints:
  Deterministic (FEDformer):
    MAE, RMSE, R²

  Probabilistic (SDCDM scenarios):
    CRPS, PICP (at 90% interval), mean interval width

Generates visualisations in results/figures/:
  1. fedformer_forecast.png  – predicted vs actual (test set, first 200 h)
  2. scenarios_band.png      – P10/P50/P90 bands + true + FEDformer mean
"""

import os
import sys
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ──────────────── metric helpers ─────────────────────────────────────────

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-9)

def crps(y_true: np.ndarray, scenarios: np.ndarray) -> float:
    """
    Energy-form CRPS estimator:
      CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
    y_true:    (N,)
    scenarios: (N, S)
    """
    N, S = scenarios.shape
    # term 1: E[|X - y|]
    t1 = np.mean(np.abs(scenarios - y_true[:, None]), axis=1)   # (N,)
    # term 2: E[|X - X'|]  – sampled pairwise approximation for efficiency
    # fixed seed for reproducibility of this Monte-Carlo estimate
    rng  = np.random.default_rng(42)
    idx1 = rng.integers(0, S, size=(N, S))
    idx2 = rng.integers(0, S, size=(N, S))
    t2   = np.mean(np.abs(
        scenarios[np.arange(N)[:, None], idx1] -
        scenarios[np.arange(N)[:, None], idx2]
    ), axis=1)
    return float(np.mean(t1 - 0.5 * t2))

def picp_width(y_true: np.ndarray, scenarios: np.ndarray, alpha: float = 0.9):
    """
    Prediction Interval Coverage Probability and mean width at level `alpha`.
    """
    lo = np.percentile(scenarios, (1 - alpha) / 2 * 100, axis=1)
    hi = np.percentile(scenarios, (1 + alpha) / 2 * 100, axis=1)
    covered = (y_true >= lo) & (y_true <= hi)
    return float(covered.mean()), float(np.mean(hi - lo))


# ──────────────── visualisation ─────────────────────────────────────────

def plot_forecast(truths, preds, fig_dir, n_show=200):
    """Deterministic forecast vs truth (first n_show hours)."""
    hours = np.arange(n_show)
    # flatten  (N, pred_len) → hourly series
    truth_flat = truths.reshape(-1)[:n_show]
    pred_flat  = preds.reshape(-1)[:n_show]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(hours, truth_flat, label="Actual Wind Power", color="steelblue", lw=1.5)
    ax.plot(hours, pred_flat,  label="FEDformer Forecast", color="tomato",
            lw=1.5, linestyle="--")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Wind Power (MW)")
    ax.set_title("FEDformer Deterministic Forecast vs Actual (first 200 h of test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fedformer_forecast.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_scenarios(truths, preds, scenarios, fig_dir, n_show=72):
    """
    P10/P50/P90 bands from SDCDM + FEDformer mean + actual truth.
    n_show: number of test samples to show (each sample spans pred_len steps).
    """
    # each sample is pred_len steps; we'll show the first `n_show` samples
    k    = min(n_show, len(truths))
    truth_flat = truths[:k].reshape(-1)
    pred_flat  = preds[:k].reshape(-1)
    # reshape to (k*pred_len, S) for percentile computation
    scen_flat2 = scenarios[:k].reshape(k * truths.shape[1], scenarios.shape[1])

    p10 = np.percentile(scen_flat2, 10, axis=1)
    p50 = np.percentile(scen_flat2, 50, axis=1)
    p90 = np.percentile(scen_flat2, 90, axis=1)
    hours = np.arange(len(truth_flat))

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(hours, p10, p90, alpha=0.25, color="goldenrod", label="P10–P90 band")
    ax.plot(hours, p50,        color="goldenrod",  lw=1.5, label="P50 (median)")
    ax.plot(hours, truth_flat, color="steelblue",  lw=1.5, label="Actual")
    ax.plot(hours, pred_flat,  color="tomato",     lw=1.2, linestyle="--", label="FEDformer")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Wind Power (MW)")
    ax.set_title("SDCDM Probabilistic Scenarios (P10/P50/P90) – test set")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "scenarios_band.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_single_scenario(truths, preds, scenarios, fig_dir, sample_idx=0):
    """Show all 100 scenarios for one test sample."""
    L    = truths.shape[1]
    hour = np.arange(L)
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in range(scenarios.shape[1]):
        ax.plot(hour, scenarios[sample_idx, s], color="grey", alpha=0.1, lw=0.5)
    ax.plot(hour, np.percentile(scenarios[sample_idx], 50, axis=0),
            color="goldenrod", lw=2, label="P50")
    ax.plot(hour, truths[sample_idx], color="steelblue", lw=2, label="Actual")
    ax.plot(hour, preds[sample_idx],  color="tomato", lw=2,
            linestyle="--", label="FEDformer")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Wind Power (MW)")
    ax.set_title(f"100 SDCDM Scenarios for test sample #{sample_idx}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "single_sample_scenarios.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────── main ────────────────────────────────────────────────────

def main():
    res_dir = os.path.join(ROOT_DIR, "results")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────
    scen_path = os.path.join(res_dir, "generated_scenarios.pkl")
    with open(scen_path, "rb") as fh:
        data = pickle.load(fh)

    truths_orig  = data["truths_original"]    # (N, L)
    preds_orig   = data["preds_original"]
    scen_orig    = data["scenarios_orig"]     # (N, S, L)

    N, S, L = scen_orig.shape
    print(f"Test samples: {N}  |  Scenarios: {S}  |  Horizon: {L}")
    # ── deterministic metrics ────────────────────────────────────────────
    flat_true = truths_orig.reshape(-1)
    flat_pred = preds_orig.reshape(-1)

    det_mae  = mae(flat_true, flat_pred)
    det_rmse = rmse(flat_true, flat_pred)
    det_r2   = r2(flat_true, flat_pred)

    print("\n" + "=" * 50)
    print("DETERMINISTIC METRICS (FEDformer)")
    print("=" * 50)
    print(f"  MAE  : {det_mae:>10.4f} MW")
    print(f"  RMSE : {det_rmse:>10.4f} MW")
    print(f"  R²   : {det_r2:>10.4f}")

    # ── probabilistic metrics ─────────────────────────────────────────────
    # flatten samples over all forecast steps
    flat_true_rep = truths_orig.reshape(-1)                  # (N*L,)
    flat_scen     = scen_orig.reshape(N * L, S)               # (N*L, S)

    prob_crps        = crps(flat_true_rep, flat_scen)
    picp90, width90  = picp_width(flat_true_rep, flat_scen, alpha=0.90)

    print("\n" + "=" * 50)
    print("PROBABILISTIC METRICS (SDCDM)")
    print("=" * 50)
    print(f"  CRPS           : {prob_crps:>10.4f} MW")
    print(f"  PICP (90%)     : {picp90:>10.4f}  (ideal ≥ 0.90)")
    print(f"  Interval width : {width90:>10.4f} MW  (90%)")

    # ── save metrics to text ──────────────────────────────────────────────
    report_path = os.path.join(res_dir, "evaluation_report.txt")
    with open(report_path, "w") as fh:
        fh.write("EVALUATION REPORT\n")
        fh.write("=" * 50 + "\n")
        fh.write("\nDETERMINISTIC METRICS (FEDformer)\n")
        fh.write(f"  MAE  : {det_mae:.4f} MW\n")
        fh.write(f"  RMSE : {det_rmse:.4f} MW\n")
        fh.write(f"  R²   : {det_r2:.4f}\n")
        fh.write("\nPROBABILISTIC METRICS (SDCDM)\n")
        fh.write(f"  CRPS           : {prob_crps:.4f} MW\n")
        fh.write(f"  PICP (90%)     : {picp90:.4f}\n")
        fh.write(f"  Interval width : {width90:.4f} MW\n")
    print(f"\nReport saved → {report_path}")

    # ── plots ──────────────────────────────────────────────────────────────
    print("\nGenerating figures…")
    plot_forecast(truths_orig, preds_orig, fig_dir)
    plot_scenarios(truths_orig, preds_orig, scen_orig, fig_dir)
    plot_single_scenario(truths_orig, preds_orig, scen_orig, fig_dir)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
