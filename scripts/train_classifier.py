#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_classifier.py
===================

Classifier selection experiment for the 8-class pilot:

- Uses the same data-loading procedure and ROI order as train_paper.py
  (24-ROI core parcellation, 8-class subset).
- Builds a 24-dimensional high-gamma (70–150 Hz) band-power representation.
- Compares five candidate classifiers:
    1) Random Forest (RF)
    2) Linear SVM
    3) ℓ2-regularized multinomial Logistic Regression
    4) K-Nearest Neighbors (KNN)
    5) Ridge Classifier

All three models are evaluated on:
  - the same subjects,
  - the same trials,
  - the same 5-fold stratified CV splits,
  - the same feature representation.

For each subject and classifier, we report mean ± SD accuracy across folds.
We then summarize accuracy across subjects for each classifier.

Usage:
  cd imed-mne/train-paper
  python train_classifier.py --subjects all
  # or specific subjects:
  # python train_classifier.py --subjects 0 1 2
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import RidgeClassifier

# Import data-loading and feature-building utilities from train_paper.py
from train_paper import (
    load_subject_trials,  # X, y, sfreq, roi_names
    features_gamma,       # gamma band-power features
    ROI_ORDER_24,
    RANDOM_STATE,
)

# -------------------------------
# Config
# -------------------------------

# High-gamma band for classifier comparison (Hz)
GAMMA_BAND: Tuple[float, float] = (70.0, 150.0)

# Output directory
DEFAULT_OUT_DIR = Path("./outputs_classifier_selection")
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Classifier factory
# -------------------------------

def build_classifier_dict(random_state: int) -> Dict[str, object]:
    """
    Define the candidate classifiers with reasonable, symmetric settings.

    - Random Forest (RF): same hyperparameters as train_paper.py
    - Linear SVM: LinearSVC with L2 penalty and class_weight='balanced'
    - Logistic Regression: multinomial, L2, class_weight='balanced'
    - KNN: distance-based classifier with standardized inputs
    - Ridge: linear ridge classifier with standardized inputs

    All linear / distance-based models are wrapped in a StandardScaler pipeline.
    """
    from sklearn.ensemble import RandomForestClassifier

    clf_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )

    clf_svm = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=10000,
        )
    )

    clf_logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
    )
   

    clf_knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2,   # Euclidean distance
        )
    )

    clf_ridge = make_pipeline(
        StandardScaler(),
        RidgeClassifier(
            alpha=1.0,
            class_weight="balanced",
            random_state=random_state,
        )
    )

    return {
        "RandomForest": clf_rf,
        "LinearSVM": clf_svm,
        "LogisticRegression": clf_logreg,
        "KNN": clf_knn,
        "Ridge": clf_ridge,
    }


# -------------------------------
# CV evaluation helper
# -------------------------------

def evaluate_classifiers_for_subject(
    X: np.ndarray,
    y: np.ndarray,
    clf_dict: Dict[str, object],
    n_splits: int = 5,
    seed: int = 42,
) -> List[dict]:
    """
    Evaluate each classifier in clf_dict on the SAME CV splits for one subject.

    Parameters
    ----------
    X : array, shape (n_trials, n_features)
        Feature matrix for this subject.
    y : array-like, shape (n_trials,)
        Class labels (strings or integers).
    clf_dict : dict
        Mapping from classifier name -> sklearn estimator (unfitted).
    n_splits : int
        Number of stratified CV folds.
    seed : int
        Random seed for fold generation.

    Returns
    -------
    results : list of dict
        One entry per classifier with per-subject mean and SD accuracy and
        number of folds used.
    """
    # Encode labels for CV split stability (strings are fine, but encoding is explicit)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    # Generate splits once and reuse for all classifiers
    splits = list(skf.split(X, y_enc))

    results = []

    for clf_name, clf_proto in clf_dict.items():
        fold_accs = []

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

            # Clone classifier to avoid leakage between folds
            clf = clone(clf_proto)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            fold_accs.append(acc)

        fold_accs = np.asarray(fold_accs, dtype=float)
        results.append({
            "classifier": clf_name,
            "mean_acc_5fold": float(fold_accs.mean()),
            "std_acc_5fold": float(fold_accs.std()),
            "n_folds": int(len(fold_accs)),
        })

    return results


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classifier comparison on 8-class pilot using 24-ROI high-gamma (70–150 Hz) band power."
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        required=True,
        help="Subject IDs (0..15) or 'all' to scan ROI JSONs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for per-subject and group-level summaries.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subject list
    if len(args.subjects) == 1 and args.subjects[0].lower() == "all":
        # Discover subjects from ROI selection files, as in train_paper.py
        roi_dir = Path("../roi_selections")
        subs = sorted([
            int(p.stem.split('_')[1])
            for p in roi_dir.glob("subject_*_roi_selections.json")
        ])
    else:
        subs = [int(s) for s in args.subjects]

    print("Subjects to process:", subs)

    clf_dict = build_classifier_dict(random_state=RANDOM_STATE)

    all_rows = []

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
        print(f" - Trials: {n_trials} | ROIs: {n_rois} | Timepoints: {n_times} | sfreq = {sfreq:.1f} Hz")

        # Build 24-ROI high-gamma (70–150 Hz) band-power features
        X_feat, feat_names = features_gamma(X_trials, sfreq, roi_names, GAMMA_BAND)
        print(f" - Feature representation: 24-ROI high-γ {GAMMA_BAND[0]}–{GAMMA_BAND[1]} Hz")
        print(f"   -> Feature matrix shape: {X_feat.shape}")

        # Evaluate all classifiers for this subject
        subj_results = evaluate_classifiers_for_subject(
            X=X_feat,
            y=y_labels,
            clf_dict=clf_dict,
            n_splits=5,
            seed=RANDOM_STATE,
        )

        for r in subj_results:
            row = {
                "subject": sid,
                "n_trials": n_trials,
                "n_features": X_feat.shape[1],
                "classifier": r["classifier"],
                "mean_acc_5fold": r["mean_acc_5fold"],
                "std_acc_5fold": r["std_acc_5fold"],
                "n_folds": r["n_folds"],
            }
            all_rows.append(row)

    # Save per-subject results
    if not all_rows:
        print("\nNo results to save (no subjects with usable data).")
        return

    df = pd.DataFrame(all_rows)
    per_subject_path = out_dir / "classifier_selection_per_subject.csv"
    df.to_csv(per_subject_path, index=False)
    print(f"\nSaved per-subject results to: {per_subject_path}")

    # Group-level summary across subjects
    grouped = df.groupby("classifier")["mean_acc_5fold"].agg(
        mean_acc="mean",
        std_acc="std",
        n_subjects="count",
    ).reset_index()

    group_path = out_dir / "classifier_selection_group_summary.csv"
    grouped.to_csv(group_path, index=False)
    print(f"Saved group-level summary to: {group_path}")

    print("\n=== Group-level accuracy across subjects (8-class, 24-ROI high-γ) ===")
    for _, row in grouped.iterrows():
        name = row["classifier"]
        mean_acc = row["mean_acc"]
        std_acc = row["std_acc"]
        n_sub = int(row["n_subjects"])
        print(f"{name:18s} {mean_acc:.3f} ± {std_acc:.3f}  (n={n_sub})")


if __name__ == "__main__":
    main()
