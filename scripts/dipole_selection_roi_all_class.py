#!/usr/bin/env python
"""
dipole_selection_roi_all_class.py

Exactly the same dipole-selection pipeline as dipole_selection_roi.py,
but processes **all classes** (no filtering) and writes JSON under
./roi_selections_all_classes with the same schema.

Everything else is unchanged:
- Builds ROI ellipsoid envelopes from fsaverage (cached)
- For each trial, picks the dipole **closest to the ROI center** (1 per ROI)
- Saves one JSON per subject + a combined JSON

"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# --- MNE import (quiet) ---
try:
    import mne
    mne.set_log_level('WARNING')
    MNE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MNE import issue: {e}")
    MNE_AVAILABLE = False

from scipy.linalg import inv, eigvalsh

# -------------------------------------------------------------------------
# Config: paths & constants
# -------------------------------------------------------------------------

SUBJECT_IDS = range(16)  # subjects 0..15
STC_DIR      = "./trialwise_sources_mne"
MAPPING_DIR  = "./trial_mappings"

# Write to a separate folder so we don't overwrite your 8-class results
OUTPUT_DIR   = "./roi_selections_all_classes"

# ROI atlas mapping (same as original)
ROI_NAMES = {
    # Visual cortex & IT
    "V1": "S_calcarine", "V2": "G_oc-temp_med-Lingual", "Cuneus": "G_cuneus",
    "Fusiform": "G_oc-temp_lat-fusifor", "IT": "G_temporal_inf",
    # Parietal
    "Precuneus": "G_precuneus", "SPL": "G_parietal_sup",
    "IPL": "G_pariet_inf-Supramar",
    # Default-mode
    "PCC": "G_cingul-Post-dorsal",
    # Pre-frontal
    "dlPFC": "G_front_middle", "SuperiorFrontal": "G_front_sup",
    # Hippocampal
    "Parahip": "G_oc-temp_med-Parahip", "MedialTemporal": "G_oc-temp_med",
}

# ROI ellipsoid parameters (same as original)
KEEP_PCT    = 98
MARGIN_FRAC = 0.50
MM          = 1_000.0  # m → mm

# Cache directory (same name)
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# ROI envelopes (unchanged logic)
# -------------------------------------------------------------------------

def build_roi_envelopes():
    """Builds ROI ellipsoid envelopes using fsaverage cortical labels (cached)."""
    print("Building ROI envelopes from fsaverage...")
    cache_file = Path(CACHE_DIR) / "roi_envelopes.json"
    if cache_file.exists():
        print("Loading ROI envelopes from cache...")
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        env = {
            k: {
                'center': np.array(v['center']),
                'inv_cov': np.array(v['inv_cov']),
                'thr': v['thr']
            } for k, v in cached.items()
        }
        print(f"Loaded {len(env)} ROI envelopes from cache")
        return env

    if not MNE_AVAILABLE:
        print("Error: MNE not available and no cache found.")
        return {}

    try:
        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False).parent
        with mne.utils.use_log_level('ERROR'):
            labels = mne.read_labels_from_annot(
                'fsaverage', parc='aparc.a2009s', subjects_dir=subjects_dir, hemi='both'
            )

        env = {}
        for lab in labels:
            hemi = 'lh' if '-lh' in lab.name else 'rh'
            base = lab.name.replace('-lh', '').replace('-rh', '')
            for short, fsname in ROI_NAMES.items():
                if base != fsname:
                    continue
                v_mm = lab.pos * MM
                centre_mm = v_mm.mean(0)
                cov_mm = np.cov(v_mm.T)
                inv_cov_mm = inv(cov_mm + 1e-6 * np.eye(3))
                d_mahal = np.sqrt(np.einsum('ni,ij,nj->n', v_mm - centre_mm, inv_cov_mm, v_mm - centre_mm))
                core_thr = np.percentile(d_mahal, KEEP_PCT)
                semi_axes = np.sqrt(np.clip(eigvalsh(cov_mm), 0, None))
                margin_mah = (MARGIN_FRAC * np.median(semi_axes)) / np.median(semi_axes)
                thr = float(core_thr + margin_mah)
                roi_key = f"{short}-{hemi}"
                env[roi_key] = {'center': centre_mm, 'inv_cov': inv_cov_mm, 'thr': thr}

        # cache
        dump = {k: {'center': v['center'].tolist(),
                    'inv_cov': v['inv_cov'].tolist(),
                    'thr': v['thr']} for k, v in env.items()}
        with open(cache_file, 'w') as f:
            json.dump(dump, f, indent=2)

        print(f"Created and cached {len(env)} ROI envelopes")
        return env
    except Exception as e:
        print(f"Error building ROI envelopes: {e}")
        return {}

def is_within_roi(position_mm, roi_envelope):
    center = roi_envelope['center']
    inv_cov = roi_envelope['inv_cov']
    thr = roi_envelope['thr']
    diff = position_mm - center
    dist = np.sqrt(diff @ inv_cov @ diff)
    return dist <= thr

# -------------------------------------------------------------------------
# Selection methods (unchanged: closest-to-center)
# -------------------------------------------------------------------------

def select_closest_to_center(dipole_positions_mm, roi_envelope, n_dipoles=1):
    """N closest dipoles to ROI center (default picks 1)."""
    roi_center = roi_envelope['center']
    d = np.linalg.norm(dipole_positions_mm - roi_center, axis=1)
    idx = np.argsort(d)[:n_dipoles]
    return idx.tolist()

SELECTION_METHOD = select_closest_to_center

# -------------------------------------------------------------------------
# Helpers to load mapping & STC positions (unchanged, except no class filter)
# -------------------------------------------------------------------------

def load_trial_mapping(subject_id: int) -> pd.DataFrame:
    """Load trial mapping for a subject; **no class filtering** (all classes)."""
    mapping_file = Path(MAPPING_DIR) / f"subject_{subject_id}_trial_mapping.csv"
    if not mapping_file.exists():
        print(f"Warning: Mapping file not found: {mapping_file}")
        return pd.DataFrame()
    df = pd.read_csv(mapping_file)
    # Keep all rows (ALL classes)
    print(f"Subject {subject_id}: {len(df)} trials (ALL classes, no filtering)")
    return df

# cache source space across trials
_SOURCE_SPACE_CACHE = None

def get_source_space():
    """Get or create fsaverage ico4 source space (cached)."""
    global _SOURCE_SPACE_CACHE
    if _SOURCE_SPACE_CACHE is not None:
        return _SOURCE_SPACE_CACHE
    cache_file = Path(CACHE_DIR) / "fsaverage_ico4_src.fif"
    if cache_file.exists():
        with mne.utils.use_log_level('ERROR'):
            _SOURCE_SPACE_CACHE = mne.read_source_spaces(str(cache_file))
        return _SOURCE_SPACE_CACHE
    print("Creating source space (first time)...")
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False).parent
    with mne.utils.use_log_level('ERROR'):
        src = mne.setup_source_space('fsaverage', spacing='ico4', subjects_dir=subjects_dir, add_dist=False)
    mne.write_source_spaces(str(cache_file), src, overwrite=True)
    _SOURCE_SPACE_CACHE = src
    print("Source space cached.")
    return src

def load_stc_positions(stc_base: str) -> np.ndarray:
    """Load dipole positions (mm) for a trial from STC; caches per-trial positions."""
    if not MNE_AVAILABLE:
        print(f"Error: MNE not available, cannot load {stc_base}")
        return np.array([])
    stc_path = Path(STC_DIR) / stc_base
    cache_file = Path(CACHE_DIR) / f"{stc_base.replace('/', '_')}_positions.npy"
    if cache_file.exists():
        return np.load(cache_file)
    try:
        with mne.utils.use_log_level('ERROR'):
            stc = mne.read_source_estimate(str(stc_path))
        src = get_source_space()
        # LH + RH vertices → positions (mm)
        pos_lh = src[0]['rr'][stc.vertices[0]] * MM
        pos_rh = src[1]['rr'][stc.vertices[1]] * MM
        all_pos = np.vstack([pos_lh, pos_rh])
        np.save(cache_file, all_pos)
        return all_pos
    except Exception as e:
        print(f"Error loading STC {stc_base}: {e}")
        return np.array([])

# -------------------------------------------------------------------------
# Core per-trial / per-subject processing (unchanged)
# -------------------------------------------------------------------------

def process_trial(subject_id: int, trial_row: pd.Series, roi_envs: dict):
    trial_idx = trial_row['processing_trial_idx']
    stc_base  = trial_row['stc_filename_base']
    class_nm  = trial_row['class_name']
    class_cd  = trial_row.get('class_code', '')

    pos = load_stc_positions(stc_base)
    if pos.size == 0:
        print(f"  [W] positions missing for trial {trial_idx}")
        return None

    roi_selections, roi_counts = {}, {}
    for roi_name, env in roi_envs.items():
        sel = SELECTION_METHOD(pos, env, n_dipoles=1)
        roi_selections[roi_name] = sel
        roi_counts[roi_name] = len(sel)

    return {
        'trial_idx': int(trial_idx),
        'stc_filename': stc_base,
        'class_name': class_nm,
        'class_code': class_cd,
        'roi_selections': roi_selections,
        'roi_counts': roi_counts,
        'total_dipoles': int(pos.shape[0])
    }

def process_subject(subject_id: int, roi_envs: dict):
    print(f"\n=== Processing Subject {subject_id} (ALL classes) ===")
    df = load_trial_mapping(subject_id)
    if df.empty:
        print(f"No trials for subject {subject_id}")
        return {}

    _ = get_source_space()  # init/cache
    results = {}
    roi_stats = defaultdict(list)

    print(f"Processing {len(df)} trials...")
    for k, (_, row) in enumerate(df.iterrows(), 1):
        if k % 50 == 0 or k == 1:
            print(f"  Trial {k}/{len(df)}")
        tr = process_trial(subject_id, row, roi_envs)
        if tr is None:
            continue
        tkey = f"trial_{tr['trial_idx']}"
        results[tkey] = tr
        for rn, c in tr['roi_counts'].items():
            roi_stats[rn].append(c)

    print(f"\nROI Stats (subject {subject_id}):")
    print(f"{'ROI':<20} {'Mean':<8} {'Std':<8} {'Min':<6} {'Max':<6}")
    print("-"*52)
    for rn in sorted(roi_stats.keys()):
        arr = np.array(roi_stats[rn]) if roi_stats[rn] else np.array([0])
        print(f"{rn:<20} {arr.mean():<8.1f} {arr.std():<8.1f} {arr.min():<6} {arr.max():<6}")

    print(f"Processed {len(results)} trials for subject {subject_id}")
    return results

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting ROI dipole selection (ALL classes).")
    roi_envs = build_roi_envelopes()
    if not roi_envs:
        print("Error: ROI envelopes unavailable.")
        return

    print(f"\nROI envelopes ready for {len(roi_envs)} regions.")
    all_results = {}
    for sid in SUBJECT_IDS:
        subj_res = process_subject(sid, roi_envs)
        if not subj_res:
            continue
        out_file = Path(OUTPUT_DIR) / f"subject_{sid}_roi_selections_all_classes.json"
        with open(out_file, 'w') as f:
            json.dump(subj_res, f, indent=2)
        print(f"Saved: {out_file}")
        all_results[f"subject_{sid}"] = subj_res

    combo = Path(OUTPUT_DIR) / "all_subjects_roi_selections_all_classes.json"
    with open(combo, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results: {combo}")
    print("Done.")

# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
