#!/usr/bin/env python3
"""
train_ll_alone_24+50.py

Runs two line-length (LL) baseline feature sets for 24-ROI and 50-ROI
representations on ALL / COARSE / FINE (class subsets) for all subjects.

Feature sets
-----------

1. ll24:
   • 24 × line length on original broadband time series — ALL 24 ROIs

2. ll50:
   • 50 × line length on original broadband time series — ALL 50 ROIs
     (intersection across trials)

Class subsets
-------------
- all:    all 80 ImageNet classes (synset_map_en.txt)
- coarse: first 40 entries (coarse categories)
- fine:   5×8 fine-grained groups as in train_fine_grain_fixed.py

Outputs (relative to this script directory: train-paper/)
--------------------------------------------------------
  ll_baselines_features/
  ll_baselines_results/
  ll_baselines_plots/
Each feature set additionally gets its own subdirectory.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import signal
import mne

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# -------------------------
# Paths
# -------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root

ROI_DIR_24 = ROOT / "roi_selections_all_classes"
ROI_DIR_50 = ROOT / "roi_selections_all_classes_extra_roi"
STC_DIR    = ROOT / "trialwise_sources_mne"
SYNSET_PATH = ROOT / "synset_map_en.txt"

OUT_FEAT_DIR = HERE / "ll_baselines_features"
OUT_RES_DIR  = HERE / "ll_baselines_results"
OUT_PLOT_DIR = HERE / "ll_baselines_plots"
OUT_FEAT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Data constants
# -------------------------
FS = 1000
N_TIMEPOINTS = 501

ROI24_NAMES = [
    'PCC-lh','PCC-rh','Cuneus-lh','Cuneus-rh','dlPFC-lh','dlPFC-rh',
    'SuperiorFrontal-lh','SuperiorFrontal-rh','Fusiform-lh','Fusiform-rh',
    'V2-lh','V2-rh','Parahip-lh','Parahip-rh','IPL-lh','IPL-rh',
    'SPL-lh','SPL-rh','Precuneus-lh','Precuneus-rh','IT-lh','IT-rh',
    'V1-lh','V1-rh'
]

WINDOWS = {
    "task-visual": (0.080, 0.200),
    "full":        (0.000, 0.500),
}

FEATURE_SETS = ["ll24", "ll50"]
SUBSETS = ["all", "coarse", "fine"]

# -------------------------
# Fine 5×8 groups (by synset code)
# -------------------------
FINE_GROUPS: Dict[str, List[str]] = {
    "41_48_dogs": [
        "n02099601","n02099712","n02106166","n02106550",
        "n02107142","n02110185","n02111889","n02112826"
    ],
    "49_56_fish": [
        "n01443537","n01456756","n01484850","n01494475",
        "n01496331","n02630281","n02643566","n02655020"
    ],
    "57_64_fruit": [
        "n07740461","n07745940","n07749192","n07753275",
        "n07756951","n07758680","n07772935","n12144580"
    ],
    "65_72_vehicles": [
        "n02701002","n02901620","n03384352","n03690473",
        "n03790512","n03845190","n04389033","n04465666"
    ],
    "73_80_instruments": [
        "n02672831","n02992211","n03372029","n03495258",
        "n03838899","n03884397","n04249415","n04487394"
    ],
}

# -------------------------
# Small helpers
# -------------------------
def _slice_window(x: np.ndarray, fs: int, win: Tuple[float,float]) -> np.ndarray:
    s = max(0, int(round(win[0]*fs)))
    e = min(len(x), int(round(win[1]*fs)))
    if e <= s:
        s, e = 0, len(x)
    return x[s:e]

def line_length_broadband(ts: np.ndarray, fs: int, win: Tuple[float,float]) -> float:
    seg = _slice_window(ts, fs, win)
    if seg.size < 2:
        return 0.0
    # normalized by length to make comparable across windows
    return float(np.sum(np.abs(np.diff(seg))) / (len(seg) - 1))

# -------------------------
# IO helpers
# -------------------------
def parse_subjects(s: str) -> List[int]:
    s = s.strip()
    if s.lower() == 'all':
        return list(range(16))
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',') if x.strip() != '']

def _trial_key(k: str):
    try:
        return int(k.split('_')[-1])
    except Exception:
        return k

def read_synset_labels(path: Path) -> List[str]:
    """
    Returns ordered list of 80 english labels as used in JSON 'class_name'.
    Expected format like 'n01440764 tench, Tinca tinca' (we take the part after the first space).
    Falls back to the whole line if no space exists.
    """
    if not path.exists():
        raise FileNotFoundError(f"synset_map_en.txt not found: {path}")
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            lab = parts[1] if len(parts) > 1 else parts[0]
            labels.append(lab)
    return labels[:80]

# -------------------------
# ROI / STC helpers
# -------------------------
def extract_roi_timeseries(stc_base: str,
                           roi_selections: Dict[str, List[int]],
                           roi_names: List[str]) -> np.ndarray:
    """
    Read a paired STC and extract one representative time series per ROI name.

    stc_base: stored in JSON, e.g., 'sources_subject_0_trial_123'
    roi_selections: mapping ROI name -> [vertex index]
    roi_names: list of ROI names to extract; missing ones become zeros.
    """
    with mne.utils.use_log_level("ERROR"):
        stc = mne.read_source_estimate(str(STC_DIR / stc_base))
    data = stc.data  # [n_vertices, n_times]
    T = min(data.shape[1], N_TIMEPOINTS)
    roi_ts = np.zeros((len(roi_names), T), dtype=float)
    for i, rname in enumerate(roi_names):
        sel = roi_selections.get(rname, [])
        if isinstance(sel, (int, np.integer)):
            sel = [int(sel)]
        if sel:
            idx = int(sel[0])
            if 0 <= idx < data.shape[0]:
                roi_ts[i, :] = data[idx, :T]
    return roi_ts

def infer_roi50_names(rows: List[Dict]) -> List[str]:
    """
    For 50-ROI set: infer a stable ROI name list as the intersection of keys
    across all trials for a given subject/subset.
    """
    keys: Optional[set] = None
    for row in rows:
        rs = row.get("roi_selections", {})
        kset = set(rs.keys())
        keys = kset if keys is None else (keys & kset)
    if not keys:
        raise RuntimeError("No common ROI keys across trials for 50-ROI set.")
    return sorted(keys)

# -------------------------
# Trial loaders
# -------------------------
def load_trials_by_name(subject_id: int,
                        subset: str,
                        allowed_labels: List[str],
                        roi_set: int) -> List[Dict]:
    """
    Load trials for ALL / COARSE subsets using class_name-based filtering.

    roi_set: 24 or 50
    """
    roi_dir = ROI_DIR_24 if roi_set == 24 else ROI_DIR_50
    sel_path = roi_dir / f"subject_{subject_id}_roi_selections_all_classes.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"[{subset}] ROI selections not found: {sel_path}")

    with open(sel_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    allowed = set(allowed_labels)
    rows: List[Dict] = []
    kept = skipped = 0

    for k in sorted(raw.keys(), key=_trial_key):
        row = raw[k]
        cname = row.get("class_name")
        if cname in allowed:
            rows.append({
                "stc_filename": row.get("stc_filename"),
                "roi_selections": row.get("roi_selections", {}),
                "class_name": cname,
                "class_code": row.get("class_code"),
            })
            kept += 1
        else:
            skipped += 1

    print(f"  [info] subject {subject_id}: kept {kept} trials, skipped {skipped} not in subset '{subset}' (roi_set={roi_set}).")
    if kept == 0:
        uniq = sorted({raw[k].get("class_name") for k in raw.keys()})
        preview = uniq[:5]
        raise RuntimeError(
            f"[{subset}] No trials for subject {subject_id} after filtering. "
            f"First few 'class_name' values in JSON: {preview}"
        )
    return rows

def load_trials_by_codes(subject_id: int,
                         allowed_codes: List[str],
                         roi_set: int) -> List[Dict]:
    """
    Load trials for FINE 5×8 groups using class_code-based filtering.

    roi_set: 24 or 50
    """
    roi_dir = ROI_DIR_24 if roi_set == 24 else ROI_DIR_50
    sel_path = roi_dir / f"subject_{subject_id}_roi_selections_all_classes.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"[fine] ROI selections not found: {sel_path}")

    with open(sel_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    allowed = set(allowed_codes)
    rows: List[Dict] = []
    kept = skipped = 0

    for k in sorted(raw.keys(), key=_trial_key):
        row = raw[k]
        ccode = row.get("class_code")
        if ccode in allowed:
            rows.append({
                "stc_filename": row.get("stc_filename"),
                "roi_selections": row.get("roi_selections", {}),
                "class_name": row.get("class_name"),
                "class_code": ccode,
            })
            kept += 1
        else:
            skipped += 1

    print(f"    [fine] subject {subject_id}: kept {kept} trials, skipped {skipped} for codes={sorted(list(allowed))} (roi_set={roi_set}).")
    if kept == 0:
        uniq = sorted({raw[k].get("class_code") for k in raw.keys()})
        preview = uniq[:8]
        raise RuntimeError(
            f"[fine] No trials for subject {subject_id} for codes {sorted(list(allowed))}. "
            f"First few 'class_code' values in JSON: {preview}"
        )
    return rows

# -------------------------
# Feature builder
# -------------------------
def build_ll_features_for_rows(
    rows: List[Dict],
    roi_names: List[str],
    win: Tuple[float,float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build line-length features for a list of trials.

    Returns (X, y, feature_names) where y is class label (string or int).
    """
    X_list: List[List[float]] = []
    y_list: List = []

    for row in rows:
        ts = extract_roi_timeseries(row["stc_filename"], row["roi_selections"], roi_names)
        feats: List[float] = []
        for i in range(len(roi_names)):
            val = line_length_broadband(ts[i], FS, win)
            feats.append(val)

        X_list.append(feats)
        label = row.get("class_name") or row.get("class_code")
        y_list.append(label)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=object)
    feat_names = [f"{r}_LLbb" for r in roi_names]

    return X, y, feat_names

# -------------------------
# RF evaluation
# -------------------------
def run_kfold_rf(X: np.ndarray,
                 y: np.ndarray,
                 feat_names: List[str],
                 n_splits: int = 5,
                 n_trees: int = 300,
                 seed: int = 42) -> Dict:
    """
    Stratified k-fold CV with RF (gini criterion).
    Returns dict with accuracy_mean, accuracy_folds, feature_importances, ...
    """
    # Drop any all-NaN features (shouldn't happen) and impute remaining NaNs
    dfX = pd.DataFrame(X, columns=feat_names)
    dfX = dfX.loc[:, dfX.notna().any(axis=0)].copy()
    dfX = dfX.fillna(dfX.mean(numeric_only=True))
    feat_names = list(dfX.columns)
    X = dfX.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs: List[float] = []
    importances = np.zeros(X.shape[1], dtype=np.float64)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        ytr, yte = y[tr_idx], y[te_idx]

        clf = RandomForestClassifier(
            n_estimators=n_trees,
            criterion="gini",
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float(accuracy_score(yte, pred))
        accs.append(acc)
        if hasattr(clf, "feature_importances_"):
            importances += clf.feature_importances_.astype(float)
        print(f"      Fold {fold}/{n_splits}: acc={acc:.3f}")

    if len(accs) > 0:
        importances /= len(accs)

    return {
        "accuracy_mean": float(np.mean(accs) if accs else np.nan),
        "accuracy_folds": accs,
        "n_features": int(X.shape[1]),
        "feature_names": feat_names,
        "feature_importances": importances.tolist(),
    }

def plot_top20_importances(res: Dict, tag: str, out_dir: Path):
    feat_names = res["feature_names"]
    imps = np.array(res["feature_importances"], dtype=float)
    if imps.size == 0:
        return
    idx = np.argsort(imps)[::-1][:20]
    names_top = [feat_names[i] for i in idx]
    imps_top = imps[idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(names_top[::-1], imps_top[::-1])
    plt.xlabel("Mean feature importance (across folds)")
    plt.tight_layout()
    out_path = out_dir / f"top20_{tag}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return str(out_path)

# -------------------------
# Main pipeline: ALL / COARSE subsets
# -------------------------
def run_all_coarse_for_subject(
    subject_id: int,
    subset: str,
    feature_set: str,
    allowed_labels: List[str],
    win: Tuple[float,float],
    n_splits: int,
    n_trees: int,
    seed: int,
):
    assert subset in ("all", "coarse")
    assert feature_set in FEATURE_SETS

    roi_set = 24 if feature_set == "ll24" else 50

    rows = load_trials_by_name(subject_id, subset, allowed_labels, roi_set=roi_set)
    if not rows:
        raise RuntimeError(f"[{subset}/{feature_set}] no rows for subject {subject_id}")

    if roi_set == 24:
        roi_names = ROI24_NAMES
    else:
        roi_names = infer_roi50_names(rows)

    X, y, feat_names = build_ll_features_for_rows(
        rows, roi_names, win
    )

    print(f"    Subject {subject_id} [{subset}/{feature_set}] — X={X.shape}, classes={len(np.unique(y))}")

    res = run_kfold_rf(X, y, feat_names, n_splits=n_splits, n_trees=n_trees, seed=seed)

    # Save features
    feat_dir = OUT_FEAT_DIR / feature_set
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_csv = feat_dir / f"features_{feature_set}_{subset}_sub{subject_id}.csv"
    dfX = pd.DataFrame(X, columns=feat_names)
    dfX.insert(0, "label", y)
    dfX.to_csv(feat_csv, index=False)

    # Save results JSON
    res_dir = OUT_RES_DIR / feature_set
    res_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{feature_set}_{subset}_sub{subject_id}"
    plot_path = plot_top20_importances(res, tag, OUT_PLOT_DIR / feature_set)
    out_json = {
        "subject_id": subject_id,
        "subset": subset,
        "feature_set": feature_set,
        "roi_set": roi_set,
        "window": list(win),
        "n_splits": n_splits,
        "n_trees": n_trees,
        "results": res,
        "paths": {
            "features_csv": str(feat_csv),
            "top20_plot": plot_path,
        },
    }
    (res_dir / f"results_{tag}.json").write_text(json.dumps(out_json, indent=2))

    return res["accuracy_mean"]

# -------------------------
# Main pipeline: FINE 5×8 subsets
# -------------------------
def run_fine_for_subject(
    subject_id: int,
    feature_set: str,
    win: Tuple[float,float],
    n_splits: int,
    n_trees: int,
    seed: int,
):
    roi_set = 24 if feature_set == "ll24" else 50
    group_results = []

    for gname, codes in FINE_GROUPS.items():
        print(f"  [fine] subject {subject_id}, group={gname}, feature_set={feature_set}")
        try:
            rows = load_trials_by_codes(subject_id, codes, roi_set=roi_set)
        except Exception as e:
            print(f"    [fine] skip group {gname}: {e}")
            continue

        if roi_set == 24:
            roi_names = ROI24_NAMES
        else:
            roi_names = infer_roi50_names(rows)

        # For fine groups we use class_code labels for y; assign into class_name field
        for r in rows:
            r["class_name"] = r["class_code"]

        X, y, feat_names = build_ll_features_for_rows(
            rows, roi_names, win
        )

        # Map class codes to 0..7
        codes_unique = sorted(set(codes))
        code_to_int = {c: i for i, c in enumerate(codes_unique)}
        y_int = np.array([code_to_int[str(label)] for label in y], dtype=int)

        print(f"    [fine] X={X.shape}, y codes={len(np.unique(y_int))}")

        res = run_kfold_rf(X, y_int, feat_names, n_splits=n_splits, n_trees=n_trees, seed=seed)

        feat_dir = OUT_FEAT_DIR / feature_set / "fine"
        feat_dir.mkdir(parents=True, exist_ok=True)
        feat_csv = feat_dir / f"features_{feature_set}_fine_{gname}_sub{subject_id}.csv"
        dfX = pd.DataFrame(X, columns=feat_names)
        dfX.insert(0, "label_code", y_int)
        dfX.to_csv(feat_csv, index=False)

        res_dir = OUT_RES_DIR / feature_set / "fine"
        res_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{feature_set}_fine_{gname}_sub{subject_id}"
        plot_path = plot_top20_importances(res, tag, OUT_PLOT_DIR / feature_set)
        out_json = {
            "subject_id": subject_id,
            "group": gname,
            "feature_set": feature_set,
            "roi_set": roi_set,
            "window": list(win),
            "n_splits": n_splits,
            "n_trees": n_trees,
            "results": res,
            "paths": {
                "features_csv": str(feat_csv),
                "top20_plot": plot_path,
            },
        }
        (res_dir / f"results_{tag}.json").write_text(json.dumps(out_json, indent=2))

        group_results.append((gname, res["accuracy_mean"]))

    if not group_results:
        return np.nan

    mean_acc = float(np.mean([acc for _, acc in group_results]))
    return mean_acc

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Line-length baselines (24- and 50-ROI) on all/coarse/fine class subsets."
    )
    ap.add_argument("--subjects", type=str, default="all",
                    help="Subject ids, e.g. 'all', '0-15' or '0,1,2'")
    ap.add_argument("--feature-sets", type=str,
                    default="ll24,ll50",
                    help=f"Comma list from {{{','.join(FEATURE_SETS)}}}")
    ap.add_argument("--subsets", type=str, default="all,coarse,fine",
                    help="Comma list from {all,coarse,fine}")
    ap.add_argument("--window", choices=list(WINDOWS.keys()), default="full")
    ap.add_argument("--win-start", type=float, default=None)
    ap.add_argument("--win-end",   type=float, default=None)

    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-trees",  type=int, default=300)
    ap.add_argument("--seed",     type=int, default=42)

    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    fs_requested = [s.strip() for s in args.feature_sets.split(",") if s.strip() in FEATURE_SETS]
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip() in SUBSETS]

    custom_win = (args.win_start, args.win_end) if (args.win_start is not None and args.win_end is not None) else None
    win = WINDOWS[args.window] if custom_win is None else custom_win

    # Read ordered labels for all/coarse subsets
    labels80 = read_synset_labels(SYNSET_PATH)
    if len(labels80) < 80:
        print(f"[warn] synset_map_en.txt yielded only {len(labels80)} labels.")
    subset_to_labels = {
        "all":    labels80,
        "coarse": labels80[:40],
        # fine handled separately via FINE_GROUPS
    }

    print("="*72)
    print(f"Subjects: {subjects}")
    print(f"Feature sets: {fs_requested}")
    print(f"Subsets: {subsets}")
    print(f"Window={win}")
    print(f"CV: {args.n_splits}-fold RF, trees={args.n_trees}, seed={args.seed}")
    print("="*72)

    # Ensure plot subdirs exist
    for fs_name in fs_requested:
        (OUT_PLOT_DIR / fs_name).mkdir(parents=True, exist_ok=True)

    # Aggregate summary
    summary_rows = []

    for subset in subsets:
        if subset in ("all", "coarse"):
            allowed_labels = subset_to_labels[subset]
            for fs_name in fs_requested:
                print(f"\n>>> Subset={subset}, feature_set={fs_name}")
                for sid in subjects:
                    try:
                        acc = run_all_coarse_for_subject(
                            subject_id=sid,
                            subset=subset,
                            feature_set=fs_name,
                            allowed_labels=allowed_labels,
                            win=win,
                            n_splits=args.n_splits,
                            n_trees=args.n_trees,
                            seed=args.seed,
                        )
                        summary_rows.append({
                            "subject": sid,
                            "subset": subset,
                            "feature_set": fs_name,
                            "accuracy_mean": acc,
                        })
                    except Exception as e:
                        print(f"  [skip] subset={subset}, feature_set={fs_name}, subject={sid}: {e}")
        elif subset == "fine":
            for fs_name in fs_requested:
                print(f"\n>>> Subset=fine (5x8 groups), feature_set={fs_name}")
                for sid in subjects:
                    try:
                        acc = run_fine_for_subject(
                            subject_id=sid,
                            feature_set=fs_name,
                            win=win,
                            n_splits=args.n_splits,
                            n_trees=args.n_trees,
                            seed=args.seed,
                        )
                        summary_rows.append({
                            "subject": sid,
                            "subset": "fine",
                            "feature_set": fs_name,
                            "accuracy_mean": acc,
                        })
                        print(f"    [fine] subject {sid}, feature_set={fs_name}, mean acc across groups = {acc:.3f}")
                    except Exception as e:
                        print(f"  [skip] fine, feature_set={fs_name}, subject={sid}: {e}")
        else:
            print(f"[warn] unknown subset: {subset}")

    # Save overall summary
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values(["subset", "feature_set", "subject"])
        # E.g. subjects 0-15
        subj_tag = f"{subjects[0]}-{subjects[-1]}" if len(subjects) > 1 else f"{subjects[0]}"
        summary_csv = OUT_RES_DIR / f"summary_all_subjects_{subj_tag}.csv"
        df_sum.to_csv(summary_csv, index=False)
        print("\nSaved global summary ->", summary_csv)
        for subset in sorted(df_sum["subset"].unique()):
            for fs_name in fs_requested:
                sub_df = df_sum[(df_sum["subset"] == subset) & (df_sum["feature_set"] == fs_name)]
                if sub_df.empty:
                    continue
                mu = sub_df["accuracy_mean"].mean()
                sd = sub_df["accuracy_mean"].std(ddof=1)
                print(f"  {subset:6s} | {fs_name:7s} : {mu:.3f} ± {sd:.3f}  (n={len(sub_df)})")
    else:
        print("\nNo results to summarize (all runs skipped or failed).")

if __name__ == "__main__":
    main()
