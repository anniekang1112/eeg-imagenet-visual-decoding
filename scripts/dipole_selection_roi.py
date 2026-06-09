#!/usr/bin/env python

"""
dipole_selection_roi.py

Selects dipoles within predefined ROI boundaries from source estimates (.stc files).
Adapted from original dipole_selection.py to work with MNE source localization results.

This script:
1. Loads trial mappings to get class information
2. Filters for selected classes only
3. For each trial, loads the .stc file and extracts dipole positions
4. Applies ROI selection method to find dipoles within each ROI boundary
5. Saves ROI selections as JSON files (one per subject)

Output: JSON files with dipole indices for each ROI for each trial
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Import MNE with error handling
try:
    import mne
    # Set MNE logging to reduce output
    mne.set_log_level('WARNING')
    MNE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MNE import issue: {e}")
    print("Trying alternative approach...")
    MNE_AVAILABLE = False

from scipy.linalg import inv, eigvalsh

###############################################################################
# Configuration - Easy to modify
###############################################################################

# Selected classes (8 classes from your table)
SELECTED_CLASSES = [
    "African elephant, Loxodonta africana",
    "airliner", 
    "banana",
    "electric guitar",
    "folding chair", 
    "desktop computer",
    "lycaenid, lycaenid butterfly",
    "revolver, six-gun, six-shooter"
]

# Input/Output paths
SUBJECT_IDS = range(16)  # subjects 0..15
STC_DIR = "./trialwise_sources_mne"
MAPPING_DIR = "./trial_mappings"
OUTPUT_DIR = "./roi_selections"

# ROI definitions (12 regions × 2 hemispheres = 24 ROIs)
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

# ROI ellipsoid parameters
KEEP_PCT = 98            # 98-percentile core
MARGIN_FRAC = 0.50       # +50% safety-margin
MM = 1_000.0             # m → mm conversion

# Set MNE logging to reduce output
mne.set_log_level('WARNING')

###############################################################################
# ROI Boundary Functions (from original dipole_selection.py)
###############################################################################

def build_roi_envelopes():
    """
    Builds ROI ellipsoid envelopes using fsaverage cortical labels.
    Returns dictionary with ROI boundaries for dipole selection.
    """
    print("Building ROI envelopes from fsaverage...")
    
    # Check cache first
    cache_file = Path(CACHE_DIR) / "roi_envelopes.json"
    if cache_file.exists():
        print("Loading ROI envelopes from cache...")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        # Convert back to numpy arrays
        env = {}
        for roi_name, roi_data in cached_data.items():
            env[roi_name] = {
                'center': np.array(roi_data['center']),
                'inv_cov': np.array(roi_data['inv_cov']),
                'thr': roi_data['thr']
            }
        print(f"Loaded {len(env)} ROI envelopes from cache")
        return env
    
    if not MNE_AVAILABLE:
        print("Error: MNE not available and no cache found. Please run with working MNE installation first.")
        return {}
    
    try:
        # Get fsaverage subjects directory
        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False).parent
        
        # Read cortical labels
        with mne.utils.use_log_level('ERROR'):
            labels = mne.read_labels_from_annot(
                'fsaverage',
                parc='aparc.a2009s',
                subjects_dir=subjects_dir,
                hemi='both'
            )
        
        env = {}
        
        for lab in labels:
            hemi = 'lh' if '-lh' in lab.name else 'rh'
            base = lab.name.replace('-lh', '').replace('-rh', '')
            
            for short, fsname in ROI_NAMES.items():
                if base != fsname:
                    continue
                    
                # Convert positions to mm
                v_mm = lab.pos * MM
                centre_mm = v_mm.mean(0)
                cov_mm = np.cov(v_mm.T)
                inv_cov_mm = inv(cov_mm + 1e-6 * np.eye(3))
                
                # Calculate Mahalanobis distances
                d_mahal = np.sqrt(np.einsum('ni,ij,nj->n',
                                            v_mm - centre_mm,
                                            inv_cov_mm,
                                            v_mm - centre_mm))
                
                # Compute threshold
                core_thr = np.percentile(d_mahal, KEEP_PCT)
                semi_axes = np.sqrt(np.clip(eigvalsh(cov_mm), 0, None))
                margin_mah = (MARGIN_FRAC * np.median(semi_axes)) / np.median(semi_axes)
                thr = core_thr + margin_mah
                
                # Store ROI envelope
                roi_key = f"{short}-{hemi}"
                env[roi_key] = {
                    'center': centre_mm,
                    'inv_cov': inv_cov_mm,
                    'thr': float(thr)
                }
        
        # Save to cache
        cache_data = {}
        for roi_name, roi_env in env.items():
            cache_data[roi_name] = {
                'center': roi_env['center'].tolist(),
                'inv_cov': roi_env['inv_cov'].tolist(),
                'thr': roi_env['thr']
            }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Created and cached {len(env)} ROI envelopes")
        return env
        
    except Exception as e:
        print(f"Error building ROI envelopes: {e}")
        return {}

def is_within_roi(position_mm, roi_envelope):
    """
    Check if a position (in mm) is within the ROI ellipsoid boundary.
    
    Args:
        position_mm: 3D position in mm coordinates [x, y, z]
        roi_envelope: ROI boundary dict with 'center', 'inv_cov', 'thr'
    
    Returns:
        bool: True if position is within ROI boundary
    """
    center = roi_envelope['center']
    inv_cov = roi_envelope['inv_cov']
    threshold = roi_envelope['thr']
    
    # Compute Mahalanobis distance
    diff = position_mm - center
    distance = np.sqrt(diff @ inv_cov @ diff)
    
    return distance <= threshold

###############################################################################
# Dipole Selection Functions
###############################################################################

def select_dipoles_within_roi(dipole_positions_mm, roi_envelope):
    """
    Selection method 1: All dipoles within ROI ellipsoid boundary.
    
    Args:
        dipole_positions_mm: Array of dipole positions in mm (n_dipoles, 3)
        roi_envelope: ROI boundary definition
    
    Returns:
        list: Indices of dipoles within ROI boundary
    """
    selected_indices = []
    
    for idx, position in enumerate(dipole_positions_mm):
        if is_within_roi(position, roi_envelope):
            selected_indices.append(idx)
    
    return selected_indices

# Future selection methods can be added here
def select_closest_to_center(dipole_positions_mm, roi_envelope, n_dipoles=1):
    """
    Selection method 2: N closest dipoles to ROI center.
    
    Args:
        dipole_positions_mm: Array of dipole positions in mm (n_dipoles, 3)
        roi_envelope: ROI boundary definition (contains 'center')
        n_dipoles: Number of closest dipoles to select
    
    Returns:
        list: Indices of N closest dipoles
    """
    roi_center = roi_envelope['center']
    distances = np.linalg.norm(dipole_positions_mm - roi_center, axis=1)
    closest_indices = np.argsort(distances)[:n_dipoles]
    return closest_indices.tolist()

# Current selection method (easy to swap)
SELECTION_METHOD = select_closest_to_center  # Changed from select_dipoles_within_roi

# Cache directories and global variables
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Global cache for source space (to avoid recreating for each trial)
_SOURCE_SPACE_CACHE = None

###############################################################################
# Main Processing Functions
###############################################################################

def load_trial_mapping(subject_id):
    """
    Load trial mapping for a subject and filter for selected classes.
    
    Args:
        subject_id: Subject ID (0-15)
    
    Returns:
        pandas.DataFrame: Filtered trial mapping
    """
    mapping_file = Path(MAPPING_DIR) / f"subject_{subject_id}_trial_mapping.csv"
    
    if not mapping_file.exists():
        print(f"Warning: Mapping file not found: {mapping_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(mapping_file)
    
    # Filter for selected classes only
    df_filtered = df[df['class_name'].isin(SELECTED_CLASSES)]
    
    print(f"Subject {subject_id}: {len(df_filtered)} trials from {len(SELECTED_CLASSES)} selected classes")
    
    return df_filtered

def get_source_space():
    """
    Get or create the fsaverage source space (cached for efficiency).
    
    Returns:
        mne.SourceSpaces: The fsaverage ico4 source space
    """
    global _SOURCE_SPACE_CACHE
    
    if _SOURCE_SPACE_CACHE is not None:
        return _SOURCE_SPACE_CACHE
    
    # Check file cache
    cache_file = Path(CACHE_DIR) / "fsaverage_ico4_src.fif"
    
    if cache_file.exists():
        print("Loading source space from cache...")
        with mne.utils.use_log_level('ERROR'):
            _SOURCE_SPACE_CACHE = mne.read_source_spaces(str(cache_file))
        return _SOURCE_SPACE_CACHE
    
    print("Creating source space (this may take a moment)...")
    
    # Get fsaverage subjects directory
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False).parent
    
    # Create source space
    with mne.utils.use_log_level('ERROR'):
        src = mne.setup_source_space(
            'fsaverage', 
            spacing='ico4', 
            subjects_dir=subjects_dir,
            add_dist=False
        )
    
    # Cache to file
    mne.write_source_spaces(str(cache_file), src, overwrite=True)
    print("Source space cached for future use")
    
    _SOURCE_SPACE_CACHE = src
    return src

def load_stc_positions(stc_filename):
    """
    Load dipole positions from .stc file.
    
    Args:
        stc_filename: Base filename (without -lh.stc suffix)
    
    Returns:
        numpy.ndarray: Dipole positions in mm (n_dipoles, 3)
    """
    if not MNE_AVAILABLE:
        print(f"Error: MNE not available, cannot load {stc_filename}")
        return np.array([])
    
    stc_path = Path(STC_DIR) / stc_filename
    
    # Check cache first
    cache_file = Path(CACHE_DIR) / f"{stc_filename.replace('/', '_')}_positions.npy"
    if cache_file.exists():
        return np.load(cache_file)
    
    try:
        # Load source estimate (without verbose parameter)
        with mne.utils.use_log_level('ERROR'):
            stc = mne.read_source_estimate(str(stc_path))
        
        # Get cached source space
        src = get_source_space()
        
        # Extract dipole positions in mm
        positions_lh = src[0]['rr'][stc.vertices[0]] * MM  # Left hemisphere
        positions_rh = src[1]['rr'][stc.vertices[1]] * MM  # Right hemisphere
        
        # Combine both hemispheres
        all_positions = np.vstack([positions_lh, positions_rh])
        
        # Cache the positions
        np.save(cache_file, all_positions)
        
        return all_positions
        
    except Exception as e:
        print(f"Error loading {stc_filename}: {e}")
        return np.array([])

def process_trial(subject_id, trial_row, roi_envelopes):
    """
    Process one trial and return ROI dipole selections.
    
    Args:
        subject_id: Subject ID
        trial_row: Row from trial mapping DataFrame
        roi_envelopes: Dictionary of ROI boundary definitions
    
    Returns:
        dict: Trial results with ROI selections
    """
    trial_idx = trial_row['processing_trial_idx']
    stc_filename = trial_row['stc_filename_base']
    class_name = trial_row['class_name']
    
    # Load dipole positions
    dipole_positions = load_stc_positions(stc_filename)
    
    if len(dipole_positions) == 0:
        print(f"  Warning: Could not load positions for trial {trial_idx}")
        return None
    
    # Apply selection method to each ROI
    roi_selections = {}
    roi_counts = {}
    
    for roi_name, roi_env in roi_envelopes.items():
        # Use the current selection method (now closest-to-center)
        selected_indices = SELECTION_METHOD(dipole_positions, roi_env, n_dipoles=1)
        roi_selections[roi_name] = selected_indices
        roi_counts[roi_name] = len(selected_indices)
    
    # Create trial result
    trial_result = {
        'trial_idx': int(trial_idx),
        'stc_filename': stc_filename,
        'class_name': class_name,
        'class_code': trial_row['class_code'],
        'roi_selections': roi_selections,
        'roi_counts': roi_counts,
        'total_dipoles': len(dipole_positions)
    }
    
    return trial_result

def process_subject(subject_id, roi_envelopes):
    """
    Process all trials for one subject.
    
    Args:
        subject_id: Subject ID (0-15)
        roi_envelopes: Dictionary of ROI boundary definitions
    
    Returns:
        dict: Subject results
    """
    print(f"\n=== Processing Subject {subject_id} ===")
    
    # Load trial mapping
    trial_mapping = load_trial_mapping(subject_id)
    
    if trial_mapping.empty:
        print(f"No trials found for subject {subject_id}")
        return {}
    
    # Initialize source space once for this subject
    print("Initializing source space...")
    _ = get_source_space()  # This caches it
    
    # Process each trial
    subject_results = {}
    roi_stats = defaultdict(list)  # For statistics
    
    print(f"Processing {len(trial_mapping)} trials...")
    
    for idx, (_, trial_row) in enumerate(trial_mapping.iterrows()):
        if idx % 50 == 0:  # Progress every 50 trials
            print(f"  Trial {idx+1}/{len(trial_mapping)}")
            
        trial_result = process_trial(subject_id, trial_row, roi_envelopes)
        
        if trial_result is not None:
            trial_key = f"trial_{trial_result['trial_idx']}"
            subject_results[trial_key] = trial_result
            
            # Collect statistics
            for roi_name, count in trial_result['roi_counts'].items():
                roi_stats[roi_name].append(count)
    
    # Print ROI statistics
    print(f"\nROI Statistics for Subject {subject_id}:")
    print(f"{'ROI':<20} {'Mean':<8} {'Std':<8} {'Min':<6} {'Max':<6}")
    print("-" * 50)
    
    for roi_name in sorted(roi_stats.keys()):
        counts = roi_stats[roi_name]
        if counts:
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            min_count = np.min(counts)
            max_count = np.max(counts)
            print(f"{roi_name:<20} {mean_count:<8.1f} {std_count:<8.1f} {min_count:<6} {max_count:<6}")
    
    print(f"Processed {len(subject_results)} trials for subject {subject_id}")
    
    return subject_results

###############################################################################
# Main Function
###############################################################################

def main():
    """Main processing function."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting ROI dipole selection...")
    print(f"Selected classes: {len(SELECTED_CLASSES)}")
    for i, class_name in enumerate(SELECTED_CLASSES, 1):
        print(f"  {i}. {class_name}")
    
    # Build ROI envelopes
    roi_envelopes = build_roi_envelopes()
    
    if not roi_envelopes:
        print("Error: Could not create ROI envelopes")
        return
    
    print(f"\nROI envelopes created for {len(roi_envelopes)} regions:")
    for roi_name in sorted(roi_envelopes.keys()):
        print(f"  {roi_name}")
    
    # Process each subject
    all_results = {}
    
    for subject_id in SUBJECT_IDS:
        subject_results = process_subject(subject_id, roi_envelopes)
        
        if subject_results:
            # Save subject results
            output_file = Path(OUTPUT_DIR) / f"subject_{subject_id}_roi_selections.json"
            
            with open(output_file, 'w') as f:
                json.dump(subject_results, f, indent=2)
            
            print(f"Saved results: {output_file}")
            all_results[f"subject_{subject_id}"] = subject_results
    
    # Save combined results
    combined_file = Path(OUTPUT_DIR) / "all_subjects_roi_selections.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSaved combined results: {combined_file}")
    print("ROI dipole selection completed!")

###############################################################################
# Helper Functions for Analysis
###############################################################################

def load_roi_selections(subject_id, output_dir="./roi_selections"):
    """
    Helper function to load ROI selections for a subject.
    
    Args:
        subject_id: Subject ID (0-15)
        output_dir: Directory containing ROI selection JSON files
    
    Returns:
        dict: ROI selections for the subject
    """
    file_path = Path(output_dir) / f"subject_{subject_id}_roi_selections.json"
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        print(f"ROI selections file not found: {file_path}")
        return {}

def get_dipole_indices_for_roi(subject_id, trial_idx, roi_name, output_dir="./roi_selections"):
    """
    Helper function to get dipole indices for a specific ROI and trial.
    
    Args:
        subject_id: Subject ID (0-15)
        trial_idx: Trial index
        roi_name: ROI name (e.g., 'V1-lh', 'dlPFC-rh')
        output_dir: Directory containing ROI selection JSON files
    
    Returns:
        list: Dipole indices for the specified ROI and trial
    """
    roi_data = load_roi_selections(subject_id, output_dir)
    trial_key = f"trial_{trial_idx}"
    
    if trial_key in roi_data and 'roi_selections' in roi_data[trial_key]:
        return roi_data[trial_key]['roi_selections'].get(roi_name, [])
    else:
        return []

if __name__ == "__main__":
    main()
    
    # Load ROI selections for subject 0
    example_data = load_roi_selections(0)
    if example_data:
        trial_keys = list(example_data.keys())[:3]  # First 3 trials
        print(f"\nFirst 3 trials for subject 0: {trial_keys}")
        
        for trial_key in trial_keys:
            trial_data = example_data[trial_key]
            print(f"\n{trial_key}:")
            print(f"  Class: {trial_data['class_name']}")
            print(f"  Total dipoles: {trial_data['total_dipoles']}")
            print(f"  ROI counts: {trial_data['roi_counts']}")