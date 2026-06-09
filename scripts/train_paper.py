#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_paper.py
==============
Per-subject 8-class experiments using the SAME data-loading procedure as your baseline:
- ROI selections from ../roi_selections/subject_<id>_roi_selections.json
- Left/right trialwise sources from ../trialwise_sources_mne/sources_subject_<id>_trial_<idx>-{lh,rh}.stc
- 24 ROI order identical to the baseline

Runs 9 feature configurations:
  1) 5-band power (δ,θ,α,β,γ=30-120)
  2) γ power (30-120)
  3) γ power (70-150)
  4) Line length (24 ROI)
  5) catch22 (24 ROI)  <-- computed on ORIGINAL time series (no gamma filtering)
  6) Phase–phase coupling (PLV-like) across ROI pairs (γ 30-120)
  7) Phase–amplitude coupling within ROI (φ: α 8-13; A: high-γ 70-150)
  8) Amplitude–amplitude coupling within ROI (β 13-30 vs high-γ 70-150)
  9) All couplings together (6+7+8 concatenated)

Each subject: 5-fold Stratified CV with RandomForest; saves:
- outputs_train_paper/results_summary.csv
- outputs_train_paper/feat_importance__S<id>__<combo>.csv

Usage:
  cd imed-mne/train-paper
  python3 train_paper.py --subjects all --out-dir ./outputs_train_paper
  # or specific subjects:
  # python3 train_paper.py --subjects 0 1 2 --out-dir ./outputs_train_paper
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch, butter, filtfilt, hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------- catch22 import (prefer pycatch22; fallback to catch22) ----------
try:
    import pycatch22 as catch22  # preferred
    _catch22_import_error = None
except Exception as e1:
    try:
        import catch22  # fallback
        _catch22_import_error = None
    except Exception as e2:
        catch22 = None
        _catch22_import_error = (e1, e2)

# -------------------------------
# Config (mirrors baseline logic)
# -------------------------------

CLASSES = [
    'African elephant', 'airliner', 'banana', 'electric guitar',
    'folding chair', 'desktop computer', 'lycaenid', 'revolver'
]

def normalize_class_name(class_name: str) -> str:
    """Same normalization as baseline: take text before comma."""
    if not class_name:
        return ""
    return class_name.split(',')[0].strip()

# Paths (relative to this script like baseline)
ROI_SELECTIONS_DIR = Path("../roi_selections")
SOURCES_DIR       = Path("../trialwise_sources_mne")

# Output
DEFAULT_OUT_DIR   = Path("./outputs_train_paper")
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ROI order identical to baseline (24 ROIs)
ROI_ORDER_24 = [
    'V1-lh', 'V1-rh', 'V2-lh', 'V2-rh', 'Fusiform-lh', 'Fusiform-rh',
    'IT-lh', 'IT-rh', 'SPL-lh', 'SPL-rh', 'IPL-lh', 'IPL-rh',
    'Precuneus-lh', 'Precuneus-rh', 'PCC-lh', 'PCC-rh',
    'dlPFC-lh', 'dlPFC-rh', 'SuperiorFrontal-lh', 'SuperiorFrontal-rh',
    'Parahip-lh', 'Parahip-rh', 'MedialTemporal-lh', 'MedialTemporal-rh'
]

# -------------------------------
# Small DSP helpers
# -------------------------------

def bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sfreq
    lo_n, hi_n = lo/nyq, hi/nyq
    b, a = butter(order, [lo_n, hi_n], btype='bandpass')
    return filtfilt(b, a, x, axis=-1)

def envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x, axis=-1))

def log_bandpower(x: np.ndarray, sfreq: float, f_lo: float, f_hi: float) -> float:
    nperseg = min(len(x), 256)
    noverlap = nperseg // 2
    f, pxx = welch(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    mask = (f >= f_lo) & (f <= f_hi)
    bp = np.trapz(pxx[mask], f[mask]) if np.any(mask) else 0.0
    return float(np.log(bp + 1e-12))

def line_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))

def plv(sig1: np.ndarray, sig2: np.ndarray, sfreq: float, f_lo: float, f_hi: float) -> float:
    x1 = bandpass(sig1, sfreq, f_lo, f_hi)
    x2 = bandpass(sig2, sfreq, f_lo, f_hi)
    ph1 = np.angle(hilbert(x1))
    ph2 = np.angle(hilbert(x2))
    return float(np.abs(np.mean(np.exp(1j * (ph1 - ph2)))))

def pac_within(sig: np.ndarray, sfreq: float,
               f_phase: Tuple[float, float], f_amp: Tuple[float, float]) -> float:
    lo = bandpass(sig, sfreq, f_phase[0], f_phase[1])
    hi = bandpass(sig, sfreq, f_amp[0],   f_amp[1])
    phase = np.angle(hilbert(lo))
    amp   = envelope(hi)
    if amp.max() > 0:
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-12)
    v = np.mean(amp * np.exp(1j * phase))
    return float(np.abs(v))

def aac_within(sig: np.ndarray, sfreq: float,
               f1: Tuple[float, float], f2: Tuple[float, float]) -> float:
    b1 = bandpass(sig, sfreq, f1[0], f1[1])
    b2 = bandpass(sig, sfreq, f2[0], f2[1])
    e1 = envelope(b1)
    e2 = envelope(b2)
    if np.std(e1) < 1e-12 or np.std(e2) < 1e-12:
        return 0.0
    return float(np.corrcoef(e1, e2)[0, 1])

# -------------------------------
# Data loader (mirrors baseline)
# -------------------------------

def load_subject_trials(subject_id: int) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """
    Load per-trial 24-ROI time series and labels for a subject, using the SAME mechanism as baseline:
      - read ROI selection JSON
      - read trialwise lh/rh .stc (no 'verbose' kwarg to support all MNE versions)
      - extract the first selected dipole per ROI (if any), else zeros
      - keep only CLASSES of interest

    Returns:
      X: [n_trials, 24, n_times]
      y: [n_trials] class strings
      sfreq: sampling rate (Hz) inferred from STC
      roi_names: list of 24 ROI names in ROI_ORDER_24
    """
    roi_file = ROI_SELECTIONS_DIR / f"subject_{subject_id}_roi_selections.json"
    if not roi_file.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_file}")

    with open(roi_file, "r", encoding="utf-8") as f:
        roi_data = json.load(f)

    trials_meta = []
    for trial_key, trial in roi_data.items():
        if not trial_key.startswith("trial_"):
            continue
        cls = normalize_class_name(trial.get("class_name", ""))
        if cls in CLASSES:
            trials_meta.append((trial.get("trial_idx"), cls, trial.get("roi_selections", {})))

    if len(trials_meta) == 0:
        return np.empty((0, 24, 0)), np.array([]), 0.0, ROI_ORDER_24

    X_list, y_list = [], []
    sfreq = None
    n_times_ref = None

    for trial_idx, cls, roi_sel in trials_meta:
        lh_path = SOURCES_DIR / f"sources_subject_{subject_id}_trial_{trial_idx}-lh.stc"
        rh_path = SOURCES_DIR / f"sources_subject_{subject_id}_trial_{trial_idx}-rh.stc"
        if (not lh_path.exists()) or (not rh_path.exists()):
            continue

        try:
            stc_lh = mne.read_source_estimate(str(lh_path))
            stc_rh = mne.read_source_estimate(str(rh_path))
        except Exception as e:
            print(f"[Subject {subject_id}] Skipping trial {trial_idx}: {e}")
            continue

        sfreq_trial = 1.0 / float(stc_lh.tstep)
        if sfreq is None:
            sfreq = sfreq_trial
        if abs(sfreq - sfreq_trial) > 1e-6:
            print(f"[WARN] Inconsistent sfreq at trial {trial_idx}: {sfreq_trial} vs {sfreq}")

        data_lh = np.asarray(stc_lh.data)  # [n_vertices_lh, n_times]
        data_rh = np.asarray(stc_rh.data)
        n_times = data_lh.shape[1]
        if n_times_ref is None:
            n_times_ref = n_times

        roi_ts = []
        for roi_name in ROI_ORDER_24:
            sel = roi_sel.get(roi_name, [])
            if isinstance(sel, list) and len(sel) > 0:
                idx = int(sel[0])
                if roi_name.endswith("-lh") and idx < data_lh.shape[0]:
                    ts = data_lh[idx, :]
                elif roi_name.endswith("-rh") and idx < data_rh.shape[0]:
                    ts = data_rh[idx, :]
                else:
                    ts = np.zeros((n_times,), dtype=float)
            else:
                ts = np.zeros((n_times,), dtype=float)
            roi_ts.append(ts)

        trial_mat = np.stack(roi_ts, axis=0)  # [24, n_times]
        X_list.append(trial_mat)
        y_list.append(cls)

    if len(X_list) == 0:
        return np.empty((0, 24, 0)), np.array([]), 0.0, ROI_ORDER_24

    X = np.stack(X_list, axis=0)  # [n_trials, 24, n_times]
    y = np.array(y_list, dtype=object)
    return X, y, float(sfreq), ROI_ORDER_24

# -------------------------------
# Feature builders
# -------------------------------

def features_5band(signals: np.ndarray, sfreq: float, roi_names: List[str]):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 120),  # per spec
    }
    n_trials, n_rois, _ = signals.shape
    names = []
    X = np.zeros((n_trials, n_rois * len(bands)), dtype=np.float32)
    bitems = list(bands.items())
    for r in range(n_rois):
        for bname, (lo, hi) in bitems:
            names.append(f"{roi_names[r]}__BP_{bname}[{lo}-{hi}]")
    for i in range(n_trials):
        row = []
        for r in range(n_rois):
            x = signals[i, r]
            for _, (lo, hi) in bitems:
                row.append(log_bandpower(x, sfreq, lo, hi))
        X[i, :] = row
    return X, names

def features_gamma(signals: np.ndarray, sfreq: float, roi_names: List[str],
                   band: Tuple[float, float]):
    n_trials, n_rois, _ = signals.shape
    names = [f"{roi_names[r]}__BP_gamma[{band[0]}-{band[1]}]" for r in range(n_rois)]
    X = np.zeros((n_trials, n_rois), dtype=np.float32)
    for i in range(n_trials):
        for r in range(n_rois):
            X[i, r] = log_bandpower(signals[i, r], sfreq, band[0], band[1])
    return X, names

def features_ll(signals: np.ndarray, roi_names: List[str]):
    n_trials, n_rois, _ = signals.shape
    names = [f"{roi_names[r]}__LL" for r in range(n_rois)]
    X = np.zeros((n_trials, n_rois), dtype=np.float32)
    for i in range(n_trials):
        for r in range(n_rois):
            X[i, r] = line_length(signals[i, r])
    return X, names

# ---- catch22 on ORIGINAL time series (no band filtering) ----
def _compute_c22_vec(x_series: np.ndarray):
    if catch22 is None:
        raise RuntimeError(
            f"catch22/pycatch22 package is not available: {_catch22_import_error}\n"
            "Install with: pip install pycatch22  # or: conda install -c conda-forge pycatch22"
        )
    res = catch22.catch22_all(x_series)
    if isinstance(res, dict) and "names" in res and "values" in res:
        names = list(res["names"])
        vals = np.asarray(res["values"], dtype=float)
    else:
        a, b = res
        if isinstance(a, (list, tuple, np.ndarray)) and len(a) and isinstance(a[0], str):
            names, vals = list(a), np.asarray(b, dtype=float)
        else:
            vals, names = np.asarray(a, dtype=float), list(b)
    return names, vals

def features_catch22(signals: np.ndarray, roi_names: List[str]):
    n_trials, n_rois, _ = signals.shape
    # canonical names from dummy
    dummy_names, _ = _compute_c22_vec(np.zeros(64, dtype=float))
    names = []
    for r in range(n_rois):
        for n in dummy_names:
            names.append(f"{roi_names[r]}__C22_{n}")
    X = np.zeros((n_trials, n_rois * len(dummy_names)), dtype=np.float32)
    for i in range(n_trials):
        row_feats = []
        for r in range(n_rois):
            _, vals = _compute_c22_vec(signals[i, r])  # ORIGINAL time series
            row_feats.extend(vals.tolist())
        X[i, :] = row_feats
    return X, names

def features_plv_pairs(signals: np.ndarray, sfreq: float, roi_names: List[str],
                       band: Tuple[float, float]):
    n_trials, n_rois, _ = signals.shape
    pairs = [(i, j) for i in range(n_rois) for j in range(i+1, n_rois)]
    names = [f"{roi_names[i]}__{roi_names[j]}__PLV[{band[0]}-{band[1]}]" for (i, j) in pairs]
    X = np.zeros((n_trials, len(pairs)), dtype=np.float32)
    for t in range(n_trials):
        vals = []
        for (i, j) in pairs:
            vals.append(plv(signals[t, i], signals[t, j], sfreq, band[0], band[1]))
        X[t, :] = vals
    return X, names

def features_pac(signals: np.ndarray, sfreq: float, roi_names: List[str],
                 f_phase: Tuple[float, float], f_amp: Tuple[float, float]):
    n_trials, n_rois, _ = signals.shape
    names = [f"{roi_names[r]}__PAC_phi[{f_phase[0]}-{f_phase[1]}]_A[{f_amp[0]}-{f_amp[1]}]" for r in range(n_rois)]
    X = np.zeros((n_trials, n_rois), dtype=np.float32)
    for t in range(n_trials):
        for r in range(n_rois):
            X[t, r] = pac_within(signals[t, r], sfreq, f_phase, f_amp)
    return X, names

def features_aac(signals: np.ndarray, sfreq: float, roi_names: List[str],
                 f1: Tuple[float, float], f2: Tuple[float, float]):
    n_trials, n_rois, _ = signals.shape
    names = [f"{roi_names[r]}__AAC_envCorr[{f1[0]}-{f1[1]}]_vs_[{f2[0]}-{f2[1]}]" for r in range(n_rois)]
    X = np.zeros((n_trials, n_rois), dtype=np.float32)
    for t in range(n_trials):
        for r in range(n_rois):
            X[t, r] = aac_within(signals[t, r], sfreq, f1, f2)
    return X, names

# -------------------------------
# CV trainer
# -------------------------------

def run_cv_rf(X: np.ndarray, y: np.ndarray, seed: int = 42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    imps = []
    for tr_idx, te_idx in skf.split(X, y_enc):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y_enc[tr_idx], y_enc[te_idx]
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            max_features='sqrt',
            class_weight='balanced',
            random_state=seed
        )
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        accs.append(accuracy_score(yte, yhat))
        imps.append(clf.feature_importances_)
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    mean_imp = np.mean(np.vstack(imps), axis=0) if imps else np.zeros((X.shape[1],), dtype=float)
    return mean_acc, std_acc, mean_imp

# -------------------------------
# Combo registry
# -------------------------------

COMBOS = [
    "5band_power_gamma_30_120",
    "gamma_power_30_120",
    "gamma_power_70_150",
    "linelength_only_24ROI",
    "catch22_only_24ROI",      # catch22 on ORIGINAL time series
    "ppc_only_24ROI",
    "pac_only_24ROI",
    "aac_only_24ROI",
    "all_couplings_24ROI",
]

def build_features(signals_24: np.ndarray, sfreq: float, roi_names_24: List[str], combo: str):
    if combo == "5band_power_gamma_30_120":
        return features_5band(signals_24, sfreq, roi_names_24)
    if combo == "gamma_power_30_120":
        return features_gamma(signals_24, sfreq, roi_names_24, (30, 120))
    if combo == "gamma_power_70_150":
        return features_gamma(signals_24, sfreq, roi_names_24, (70, 150))
    if combo == "linelength_only_24ROI":
        return features_ll(signals_24, roi_names_24)
    if combo == "catch22_only_24ROI":
        return features_catch22(signals_24, roi_names_24)  # ORIGINAL time series
    if combo == "ppc_only_24ROI":
        return features_plv_pairs(signals_24, sfreq, roi_names_24, (30, 120))
    if combo == "pac_only_24ROI":
        return features_pac(signals_24, sfreq, roi_names_24, (8, 13), (70, 150))
    if combo == "aac_only_24ROI":
        return features_aac(signals_24, sfreq, roi_names_24, (13, 30), (70, 150))
    if combo == "all_couplings_24ROI":
        X1, n1 = features_plv_pairs(signals_24, sfreq, roi_names_24, (30, 120))
        X2, n2 = features_pac(signals_24, sfreq, roi_names_24, (8, 13), (70, 150))
        X3, n3 = features_aac(signals_24, sfreq, roi_names_24, (13, 30), (70, 150))
        return np.concatenate([X1, X2, X3], axis=1), (n1 + n2 + n3)
    raise ValueError(f"Unknown combo: {combo}")

# -------------------------------
# Main
# -------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train-paper (per-subject 8-class) using baseline-like data loading.")
    ap.add_argument("--subjects", type=str, nargs="+", required=True,
                    help="Subject IDs (0..15) or 'all' to scan ROI JSONs.")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    if len(args.subjects) == 1 and args.subjects[0].lower() == "all":
        subs = sorted([int(p.stem.split('_')[1]) for p in ROI_SELECTIONS_DIR.glob("subject_*_roi_selections.json")])
    else:
        subs = [int(s) for s in args.subjects]

    results = []

    for sid in subs:
        print(f"\n=== Subject {sid} ===")
        try:
            X_trials, y_labels, sfreq, roi_names = load_subject_trials(sid)
        except Exception as e:
            print(f"[ERR] Subject {sid}: {e}")
            continue

        if X_trials.size == 0 or len(y_labels) == 0:
            print(f"[WARN] Subject {sid}: no usable 8-class trials. Skipping.")
            continue

        n_trials, n_rois, n_times = X_trials.shape
        print(f" - Trials: {n_trials} | ROI: {n_rois} | Timepoints: {n_times} | sfreq={sfreq:.1f} Hz")

        for combo in COMBOS:
            print(f"   > Combo: {combo}")
            X, feat_names = build_features(X_trials, sfreq, roi_names, combo)
            if X.shape[0] != len(y_labels):
                print(f"[ERR] Feature/trial mismatch for {combo}: X={X.shape}, y={y_labels.shape}")
                continue

            mean_acc, std_acc, mean_imp = run_cv_rf(X, y_labels, seed=RANDOM_STATE)

            # Save importances
            imp_df = pd.DataFrame({
                "subject": sid,
                "combo": combo,
                "feature": feat_names,
                "importance": mean_imp
            })
            imp_df.to_csv(out_dir / f"feat_importance__S{sid:02d}__{combo}.csv", index=False)

            # Add summary row
            results.append({
                "subject": sid,
                "combo": combo,
                "n_features": X.shape[1],
                "mean_acc_5fold": mean_acc,
                "std_acc_5fold": std_acc
            })

    # Save summary CSV
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(out_dir / "results_summary.csv", index=False)
        print(f"\nSaved summary to {out_dir/'results_summary.csv'}")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    main()
