"""
validation/validate_ht.py — confirms slow-gets-lower-shape pairing in generate_ht.

For each turbulence regime, runs generate_ht with the correct pairing
(shape_slow = min(alpha, beta), shape_fast = max(alpha, beta)) and the
wrong pairing (swapped) on identical AR(1) latent draws, then reports
mean, scintillation index, and 1/e coherence time for each, plus a
log-ACF plot contrasting the two.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import config
from channel_sim import generate_ht

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def acf_fft(x, max_lag):
    """Normalized autocorrelation via FFT (Wiener-Khinchin), truncated to max_lag."""
    x = x - x.mean()
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    fx = np.fft.fft(x, n=size)
    acf_full = np.fft.ifft(fx * np.conj(fx)).real
    acf = acf_full[: max_lag + 1]
    acf /= acf[0]
    return acf


def coherence_time(acf, dt):
    """First lag (in seconds) where the normalized ACF drops below 1/e."""
    below = np.where(acf < 1.0 / np.e)[0]
    if len(below) == 0:
        return np.nan
    return below[0] * dt


def run_one(tau_s, tau_f, shape_slow, shape_fast, n_samples, dt, seed):
    rng = np.random.default_rng(seed)
    ht = generate_ht(tau_s, tau_f, shape_slow, shape_fast, n_samples, dt, rng=rng)
    mean = ht.mean()
    si = ht.var() / mean**2
    max_lag = min(n_samples - 1, 20_000)
    acf = acf_fft(ht, max_lag)
    t_coh = coherence_time(acf, dt)
    return {"ht": ht, "mean": mean, "si": si, "acf": acf, "t_coh": t_coh}


def main():
    tau_s, tau_f, dt, n = config.TAU_SLOW, config.TAU_FAST, config.DT, config.N_SAMPLES

    rows = []
    for idx, (regime, params) in enumerate(config.TURBULENCE_REGIMES.items()):
        a, b = params["alpha"], params["beta"]
        shape_slow, shape_fast = min(a, b), max(a, b)  # correct: slow = lower shape
        seed = 1000 + idx

        correct = run_one(tau_s, tau_f, shape_slow, shape_fast, n, dt, seed=seed)
        wrong = run_one(tau_s, tau_f, shape_fast, shape_slow, n, dt, seed=seed)

        rows.append((regime, "correct", shape_slow, shape_fast, correct))
        rows.append((regime, "wrong", shape_fast, shape_slow, wrong))

        fig, ax = plt.subplots(figsize=(6, 4))
        lags_ms = np.arange(len(correct["acf"])) * dt * 1e3
        ax.semilogy(lags_ms, correct["acf"], label="correct (slow=lower shape)")
        ax.semilogy(lags_ms, wrong["acf"], label="wrong (slow=higher shape)")
        ax.axhline(1 / np.e, color="gray", linestyle="--", linewidth=0.8, label="1/e")
        ax.set_xlabel("lag (ms)")
        ax.set_ylabel("ACF (log scale)")
        ax.set_title(f"{regime} regime: alpha={a}, beta={b}")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"acf_{regime}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    header = f"{'regime':<10} {'pairing':<9} {'shape_slow':>10} {'shape_fast':>10} {'mean':>8} {'sci_idx':>8} {'t_coh_ms':>9}"
    print(header)
    print("-" * len(header))
    for regime, pairing, ss, sf, res in rows:
        t_coh_ms = res["t_coh"] * 1e3 if not np.isnan(res["t_coh"]) else float("nan")
        print(
            f"{regime:<10} {pairing:<9} {ss:>10.2f} {sf:>10.2f} "
            f"{res['mean']:>8.4f} {res['si']:>8.4f} {t_coh_ms:>9.3f}"
        )


if __name__ == "__main__":
    main()
