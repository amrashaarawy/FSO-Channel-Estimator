"""
acm.py: the use-stage. Turns a predicted channel gain into a transmission mode and
scores that choice against the true channel. Knows nothing about which model made
the prediction: it only receives an array of predicted gains.

conventions
    h       : channel gain, mean ~1 (the calibrated estimate scale)
    snr_db  : SNR at the AVERAGE channel (config.SNR_DB); instantaneous SNR = snr_db + 20 log10(h)
    power   : tx_power is the AVERAGE optical power, so the PAM peak level = 2 x average
    levels  : -1 = off, 0 = OOK, 1 = 4-PAM, 2 = 8-PAM (index into config.PAM_ORDERS)
"""
import numpy as np
from scipy.stats import norm

import config

PEAK_OVER_AVG = 2.0     # equiprobable PAM levels 0..A -> average A/2


# ---------- physics of the mode set ----------
def ber_pam(M, snr_db_inst):
    """BER of Gray-coded unipolar M-PAM in AWGN at instantaneous SNR (dB)"""
    M = np.asarray(M, dtype=float)
    rho = PEAK_OVER_AVG * 10 ** (np.asarray(snr_db_inst, dtype=float) / 20)   # peak level / sigma
    return 2 * (M - 1) / (M * np.log2(M)) * norm.sf(rho / (2 * (M - 1)))


def required_snr_db(M, ber):
    """SNR (dB) at which M-PAM reaches the given BER (inverse of ber_pam)"""
    p = ber * M * np.log2(M) / (2 * (M - 1))
    rho = 2 * (M - 1) * norm.isf(p)
    return 20 * np.log10(rho / PEAK_OVER_AVG)


def mode_thresholds(target_ber=config.TARGET_BER, off_ber=config.OFF_BER):
    """SNR each mode needs for target_ber, and the SNR below which the link switches off"""
    thr = np.array([required_snr_db(M, target_ber) for M in config.PAM_ORDERS])
    off_thr = required_snr_db(2, off_ber)           # OOK at the relaxed OFF_BER
    assert np.all(np.diff(np.r_[off_thr, thr]) > 0), "thresholds must increase: need OFF_BER > TARGET_BER"
    return thr, off_thr


# ---------- gain -> SNR ----------
def to_snr_db(h, snr_db, h_floor=config.H_FLOOR):
    """instantaneous SNR (dB) for gain h; the floor keeps log10 defined for noisy or negative values"""
    return snr_db + 20 * np.log10(np.maximum(h, h_floor))


# ---------- selection ----------
def stateless_levels(snr_db_pred, margin_db, thr, off_thr):
    """highest level whose threshold + margin the predicted SNR clears (-1 = off).
    OOK is used anywhere above the off threshold, even below its own BER target."""
    level = np.where(snr_db_pred < off_thr + margin_db, -1, 0)
    for j in range(1, len(thr)):                    # thr[0] is not used for selection, only OFF gates OOK
        level = np.where(snr_db_pred >= thr[j] + margin_db, j, level)
    return level


def select_modes(snr_db_pred, margin_db=config.MARGIN_DB, hysteresis_db=config.HYSTERESIS_DB,
                 dwell_up=config.DWELL_UP, dwell_down=config.DWELL_DOWN,
                 update_interval=config.UPDATE_INTERVAL):
    """mode per sample from predicted SNR. every switching mechanism lives here;
    with neutral knobs (hysteresis 0, dwell 1, interval 1) it reduces to stateless_levels"""
    thr, off_thr = mode_thresholds()
    l_stay = stateless_levels(snr_db_pred, margin_db, thr, off_thr)                 # what the channel supports
    l_up = stateless_levels(snr_db_pred, margin_db + hysteresis_db, thr, off_thr)   # same, with hysteresis added

    levels = np.empty(len(l_stay), dtype=int)
    cur = l_stay[0]
    up_cnt = down_cnt = 0
    for t in range(len(l_stay)):
        if t % update_interval == 0:                # decision instant, otherwise hold
            if l_stay[t] < cur:                     # supports less than we run: want to step down
                down_cnt, up_cnt = down_cnt + 1, 0
                if down_cnt >= dwell_down:
                    cur, down_cnt = l_stay[t], 0
            elif l_up[t] > cur:                     # clearly supports more: want to step up
                up_cnt, down_cnt = up_cnt + 1, 0
                if up_cnt >= dwell_up:
                    cur, up_cnt = l_up[t], 0
            else:                                   # no change wanted: reset both counters
                up_cnt = down_cnt = 0
        levels[t] = cur
    return levels


# ---------- scoring against the true channel ----------
def score(levels, h_true, snr_db, target_ber=config.TARGET_BER):
    """per-sample outcome of the chosen modes on the true gain h_true (same time index as levels)"""
    M = np.array(config.PAM_ORDERS, dtype=float)
    on = levels >= 0
    idx = np.clip(levels, 0, None)                  # off samples borrow index 0, masked below
    snr_true = to_snr_db(h_true, snr_db)
    ber_all = ber_pam(M[idx], snr_true)
    rate = np.where(on, np.log2(M[idx]), 0.0)       # bits per symbol
    return {
        "level": levels,
        "rate": rate,
        "ber": np.where(on, ber_all, np.nan),
        "violation": on & (ber_all > target_ber),   # sent, but BER above target
        "goodput": np.where(on, rate * (1 - ber_all), 0.0),
        "snr_true_db": snr_true,
    }