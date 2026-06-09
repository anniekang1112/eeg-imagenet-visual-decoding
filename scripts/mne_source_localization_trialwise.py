#!/usr/bin/env python

"""
mne_source_localization_trialwise.py

use MNE source localization
This processes each 0.5s trial individually and saves
SourceEstimate objects (.stc files) for later ROI analysis.

Key changes from original:
1. Uses mne.minimum_norm.apply_inverse() instead of mne.fit_dipole()
2. Creates cortical surface source space (fsaverage)
3. Saves SourceEstimate objects instead of dipole objects
4. Each trial gets individual source time courses

Steps:
1) Loop over subjects (0..15)
2) For each subject, load ~4,000 EEG trials (0.5s each) from cleaned_subject_X.pth
3) For each trial, create Evoked, set average reference, and apply inverse solution
4) Use parallel processing for trials
5) Save each trial's SourceEstimate to disk as .stc files
6) Skip if .stc file already exists (resume capability)
"""

import os
import mne
import torch
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

# Set MNE logging level to suppress verbose output
mne.set_log_level('WARNING')

###############################################################################
# Configuration Parameters
###############################################################################
DATA_FOLDER = "./cleaned_subjects"
SUBJECT_IDS = range(16)  # subjects 0..15
SOURCE_OUT_DIR = "./trialwise_sources_mne"

# Forward modeling files - using MNE's built-in fsaverage
SUBJECTS_DIR = None  # Will be set to MNE's fsaverage location
BEM_FILE = None      # Will use MNE's built-in fsaverage BEM
SRC_FILE = None      # Will create source space dynamically
TRANS_FILE = None    # No transformation needed for fsaverage

# EEG parameters
SFREQ = 1000.0     # Sampling frequency
TMIN = 0.0         # Start time for each trial
MONTAGE_NAME = "standard_1020"

# Source localization parameters
METHOD = "dSPM"       # Source localization method
SNR = 3.0            # Signal-to-noise ratio
LAMBDA2 = 1.0 / SNR**2  # Regularization parameter (≈ 0.111)

# Use decimation or not
USE_DECIMATION = False
DECIM = 2

# 62 channels from the dataset
MY_DATASET_CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8","AF7","AF3","AF4","AF8","F1","F2","F5","F6",
    "FT7","FT8","FC5","FC3","FC1","FC2","FC4","FC6",
    "T7","C3","C1","Cz","C2","C4","C6","T8",
    "TP9","TP10","CP5","CP3","CP1","CP2","CP4","CP6",
    "P7","P3","P1","Pz","P2","P4","P6","P8",
    "PO7","PO3","POz","PO4","PO8","PO9","PO10",
    "O1","Oz","O2","O9","O10",
    "P5","P9","P10"
]

###############################################################################
# Helper: Load a subject's cleaned EEG data
###############################################################################
def load_cleaned_subject_data(subject_id: int, data_folder: str):
    """
    Loads the subject's denoised EEG data from a .pth file.
    Returns an array of shape (n_trials, n_channels, n_times).
    """
    filename = os.path.join(data_folder, f"cleaned_subject_{subject_id}.pth")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing cleaned EEG file: {filename}")
    
    print(f"Loading subject {subject_id} data from: {filename}")
    
    try:
        # Load with memory mapping to handle large files
        data_list = torch.load(filename, weights_only=False, map_location='cpu')
        print(f"Loaded {len(data_list)} trials for subject {subject_id}")
        
        # Convert to numpy arrays
        signals_np = []
        for i, item in enumerate(data_list):
            if i % 1000 == 0:  # Progress indicator for large datasets
                print(f"Processing trial {i}/{len(data_list)}")
            signals_np.append(item['eeg_data'].numpy())
        
        signals_np = np.stack(signals_np, axis=0)
        print(f"Final data shape: {signals_np.shape}")
        return signals_np
        
    except Exception as e:
        print(f"Error loading data for subject {subject_id}: {e}")
        raise

###############################################################################
# Helper: Apply MNE source localization to a single trial
###############################################################################
def apply_mne_source_localization(trial_idx: int,
                                  trial_data: np.ndarray,
                                  original_info: mne.Info,
                                  inverse_operator: mne.minimum_norm.InverseOperator,
                                  subject_id: int) -> tuple:
    """
    Applies MNE source localization to one trial's data and saves the SourceEstimate.
    
    Args:
        trial_idx: Trial index
        trial_data: EEG data for this trial (n_channels, n_times)
        original_info: MNE Info object
        inverse_operator: Precomputed inverse operator
        subject_id: Subject ID for file naming
    
    Returns:
        (trial_idx, success_flag): Tuple indicating trial index and whether processing succeeded
    """
    # Determine output file name for the SourceEstimate
    stc_fname = os.path.join(
        SOURCE_OUT_DIR,
        f"sources_subject_{subject_id}_trial_{trial_idx}"  # .stc extension added automatically
    )
    
    # Check if this trial has already been processed
    if os.path.exists(stc_fname + "-lh.stc"):  # MNE saves as -lh.stc and -rh.stc
        return (trial_idx, True)

    try:
        # Optionally decimate the data
        if USE_DECIMATION:
            decimated_data = trial_data[:, ::DECIM]
            new_sfreq = original_info["sfreq"] / DECIM
            # Create a new Info object with decimated sampling rate
            dec_info = mne.create_info(
                ch_names=original_info["ch_names"],
                sfreq=new_sfreq,
                ch_types=["eeg"] * len(original_info["ch_names"])
            )
            montage = original_info.get_montage()
            if montage is not None:
                dec_info.set_montage(montage)
            evoked = mne.EvokedArray(decimated_data, dec_info, tmin=TMIN, comment=f"Trial{trial_idx}")
        else:
            evoked = mne.EvokedArray(trial_data, original_info, tmin=TMIN, comment=f"Trial{trial_idx}")

        # Set EEG reference (average reference) - suppress output
        with mne.utils.use_log_level('ERROR'):
            evoked.set_eeg_reference(ref_channels="average", projection=True)
            evoked.apply_proj()

        # Apply the inverse solution to get source time courses - suppress output
        with mne.utils.use_log_level('ERROR'):
            stc = mne.minimum_norm.apply_inverse(
                evoked,
                inverse_operator,
                lambda2=LAMBDA2,
                method=METHOD,
                pick_ori=None,  # Use loose orientation constraint
                verbose=False   # Reduce verbosity for parallel processing
            )

        # Save the SourceEstimate with compression
        with mne.utils.use_log_level('ERROR'):
            # Convert to float32 to save space (reduces file size by ~50%)
            stc.data = stc.data.astype(np.float32)
            stc.save(stc_fname, overwrite=True)
        
        # Print completion message
        print(f"[Subject {subject_id}] Trial {trial_idx} completed -> {stc.data.shape[0]} sources, {stc.data.shape[1]} time points")

        return (trial_idx, True)

    except Exception as e:
        print(f"[Subject {subject_id}] Trial {trial_idx} -> ERROR: {str(e)}")
        return (trial_idx, False)

###############################################################################
# Helper: Create and cache the inverse operator for a subject
###############################################################################
def create_inverse_operator(original_info: mne.Info, subject_id: int):
    """
    Creates the inverse operator for source localization using MNE's built-in fsaverage.
    This includes forward solution and noise covariance computation.
    
    Args:
        original_info: MNE Info object with channel information
        subject_id: Subject ID for caching
    
    Returns:
        inverse_operator: MNE InverseOperator object
    """
    # Check if inverse operator is already cached
    inv_fname = os.path.join(SOURCE_OUT_DIR, f"inverse_op_subject_{subject_id}-inv.fif")
    
    if os.path.exists(inv_fname):
        print(f"[Subject {subject_id}] Loading cached inverse operator from {inv_fname}")
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
        return inverse_operator

    print(f"[Subject {subject_id}] Creating inverse operator using MNE's fsaverage...")

    # Get MNE's built-in fsaverage subjects directory
    fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = fsaverage_dir.parent  # Remove the extra 'fsaverage' from path
    print(f"[Subject {subject_id}] Using subjects_dir: {subjects_dir}")
    print(f"[Subject {subject_id}] fsaverage subject: {fsaverage_dir}")
    
    # Create source space (cortical surface, ico-4 spacing)
    print(f"[Subject {subject_id}] Creating ico-4 source space...")
    src = mne.setup_source_space(
        subject="fsaverage",
        spacing="ico4",
        subjects_dir=subjects_dir,
        add_dist=False,  # Don't compute distances to save time
        n_jobs=1,
        verbose=False
    )
    print(f"[Subject {subject_id}] Source space created: {len(src)} hemispheres, "
          f"{sum(s['nuse'] for s in src)} active sources")

    # Get the fsaverage BEM model (this is built into MNE)
    print(f"[Subject {subject_id}] Loading fsaverage BEM model...")
    bem = mne.read_bem_solution(
        fsaverage_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    )

    # Create forward solution
    print(f"[Subject {subject_id}] Computing forward solution...")
    fwd = mne.make_forward_solution(
        original_info,
        trans=TRANS_FILE,  # None for fsaverage
        src=src,
        bem=bem,
        meg=False,         # EEG only
        eeg=True,
        mindist=5.0,       # Minimum distance from inner skull (5mm)
        n_jobs=1,
        verbose=False
    )
    print(f"[Subject {subject_id}] Forward solution: {fwd['nsource']} sources, "
          f"{fwd['nchan']} channels")

    # Create noise covariance with proper EEG reference handling
    print(f"[Subject {subject_id}] Creating noise covariance matrix...")
    noise_cov = mne.make_ad_hoc_cov(original_info, std=1e-4)
    
    # Add EEG average reference if not already present
    if not any('Average EEG reference' in str(proj) for proj in original_info.get('projs', [])):
        try:
            eeg_ref_proj = mne.preprocessing.make_eeg_average_ref_proj(original_info)
            # Apply the reference to the noise covariance
            noise_cov = mne.cov.regularize(noise_cov, original_info, proj=True)
            print(f"[Subject {subject_id}] Added EEG average reference to noise covariance")
        except Exception as e:
            print(f"[Subject {subject_id}] Could not add EEG reference: {e}")
            # Continue without it - the warning is not critical

    # Create inverse operator
    print(f"[Subject {subject_id}] Creating inverse operator...")
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        original_info,
        fwd,
        noise_cov,
        loose=0.2,    # Loose orientation constraint (20% tangential)
        depth=0.8,    # Depth weighting exponent
        verbose=False
    )

    # Cache the inverse operator
    print(f"[Subject {subject_id}] Saving inverse operator to {inv_fname}")
    mne.minimum_norm.write_inverse_operator(inv_fname, inverse_operator)
    print(f"[Subject {subject_id}] Inverse operator created successfully!")

    return inverse_operator

###############################################################################
# Main processing loop
###############################################################################
def main():
    # Create output directory
    os.makedirs(SOURCE_OUT_DIR, exist_ok=True)

    for subject_id in SUBJECT_IDS:
        print(f"\n=== Processing Subject {subject_id} ===")
        
        # Load subject data
        try:
            data = load_cleaned_subject_data(subject_id, DATA_FOLDER)
        except FileNotFoundError as e:
            print(e)
            continue

        n_trials, n_channels, n_times = data.shape
        print(f"Subject {subject_id}: {n_trials} trials, {n_channels} channels, {n_times} time points each.")

        # Create the original Info object (used for each trial)
        original_info = mne.create_info(
            ch_names=MY_DATASET_CHANNELS,
            sfreq=SFREQ,
            ch_types=["eeg"] * n_channels
        )
        montage = mne.channels.make_standard_montage(MONTAGE_NAME)
        original_info.set_montage(montage)
        
        # Add average EEG reference projector to avoid warnings during inverse operator creation
        try:
            # Try the newer function name first
            eeg_ref_proj = mne.preprocessing.make_eeg_average_ref_proj(original_info)
            original_info['projs'].append(eeg_ref_proj)
        except AttributeError:
            # If that doesn't work, we'll handle it in the noise covariance step
            print(f"[Subject {subject_id}] Will handle EEG reference in noise covariance calculation")
            pass

        # Handle decimation if requested
        if USE_DECIMATION:
            print(f"[Subject {subject_id}] Using decimation factor {DECIM}: "
                  f"{SFREQ} Hz -> {SFREQ/DECIM} Hz, {n_times} -> {n_times//DECIM} time points")
            print(f"[Subject {subject_id}] Expected file size per subject: ~8 GB (with float32 + decimation)")
        else:
            print(f"[Subject {subject_id}] No decimation used")
            print(f"[Subject {subject_id}] Expected file size per subject: ~65 GB")

        # Create inverse operator (this is the main computational step)
        try:
            inverse_operator = create_inverse_operator(original_info, subject_id)
        except Exception as e:
            print(f"[Subject {subject_id}] Failed to create inverse operator: {e}")
            continue

        print(f"[Subject {subject_id}] Starting source localization for {n_trials} trials...")
        print(f"[Subject {subject_id}] Note: Processing sequentially due to MNE object serialization limitations")

        # Process trials sequentially (parallel processing has serialization issues with MNE objects)
        results = []
        for trial_idx in range(n_trials):
            if trial_idx % 100 == 0:  # Progress indicator every 100 trials
                print(f"[Subject {subject_id}] Processing trial {trial_idx}/{n_trials}")
            
            trial_data = data[trial_idx]
            result = apply_mne_source_localization(
                trial_idx, trial_data, original_info, inverse_operator, subject_id
            )
            results.append(result)

        # Count successful processing
        successful_trials = sum(1 for (_, success) in results if success)
        failed_trials = n_trials - successful_trials
        
        print(f"\n--- Subject {subject_id} completed: {successful_trials} successful trials, "
              f"{failed_trials} failed trials ---\n")

    print("All subjects processed with MNE source localization and resume capability.\n")


if __name__ == "__main__":
    main()