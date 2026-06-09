#!/usr/bin/env python

"""
trial_mapping_recovery.py

Creates CSV mapping files for each participant showing which processed trial indices
correspond to which class names. Uses the original cleaned data and synset mapping
to recover the trial-to-class relationships.

Output: One CSV per participant with columns:
- processing_trial_idx: Index used in .stc filename (0, 1, 2, ...)  
- class_code: Original class code (e.g., 'n02106662')
- class_name: Human-readable class name (e.g., 'German shepherd')
- stc_filename_base: Base filename for the .stc files
"""

import os
import torch
import pandas as pd
from collections import defaultdict

###############################################################################
# Configuration
###############################################################################
DATA_FOLDER = "./cleaned_subjects"
SYNSET_FILE = "./synset_map_en.txt"
OUTPUT_DIR = "./trial_mappings"
SUBJECT_IDS = range(16)  # subjects 0..15

###############################################################################
# Debug function to search for specific codes
###############################################################################
def debug_synset_search(synset_file, search_codes):
    """
    Debug function to search for specific codes in the synset file.
    """
    print(f"Searching for codes in {synset_file}:")
    
    with open(synset_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for code in search_codes:
        if code in content:
            # Find the line containing this code
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                if line.startswith(code):
                    print(f"  FOUND {code}: {line}")
                    break
            else:
                print(f"  {code} found in file but not at line start")
        else:
            print(f"  NOT FOUND: {code}")

###############################################################################
# Helper: Load synset mapping
###############################################################################
def load_synset_mapping(synset_file):
    """
    Loads the synset mapping from class codes to human-readable names.
    
    Returns:
        dict: {class_code: class_name}
    """
    synset_map = {}
    
    if not os.path.exists(synset_file):
        print(f"Warning: Synset file not found: {synset_file}")
        return synset_map
    
    print(f"Loading synset mapping from: {synset_file}")
    
    with open(synset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            original_line = line
            line = line.strip()
            
            if not line:  # Skip empty lines
                continue
                
            if '\t' in line:
                parts = line.split('\t', 1)  # Split only on first tab
                class_code = parts[0].strip()
                class_name = parts[1].strip()
                synset_map[class_code] = class_name
            elif ' ' in line:  # Try space separator as backup
                parts = line.split(' ', 1)
                class_code = parts[0].strip()
                class_name = parts[1].strip()
                synset_map[class_code] = class_name
                print(f"  Line {line_num}: Used space separator for {class_code}")
            else:
                print(f"  Line {line_num}: No separator found: '{original_line.strip()}'")
    
    print(f"Loaded {synset_map} class mappings from synset file")
    
    # Print first few entries for debugging
    print("First few synset mappings:")
    for i, (code, name) in enumerate(synset_map.items()):
        if i < 5:
            print(f"  '{code}' -> '{name}'")
        else:
            break
    
    return synset_map

###############################################################################
# Helper: Recover trial mapping for one subject
###############################################################################
def recover_subject_mapping(subject_id, data_folder, synset_map):
    """
    Recovers the trial-to-class mapping for one subject.
    
    Args:
        subject_id: Subject ID (0-15)
        data_folder: Path to cleaned subject data
        synset_map: Dictionary mapping class codes to names
    
    Returns:
        pandas.DataFrame: Trial mapping with processing order and class info
    """
    filename = os.path.join(data_folder, f"cleaned_subject_{subject_id}.pth")
    
    if not os.path.exists(filename):
        print(f"Warning: Subject {subject_id} data file not found: {filename}")
        return pd.DataFrame()
    
    print(f"Processing subject {subject_id}...")
    
    # Load the data in the same order as the processing script
    try:
        data_list = torch.load(filename, weights_only=False)
    except Exception as e:
        print(f"Error loading subject {subject_id}: {e}")
        return pd.DataFrame()
    
    print(f"  Loaded {len(data_list)} trials")
    
    # Extract mapping information
    mapping_data = []
    class_counts = defaultdict(int)
    missing_codes = set()
    
    for processing_idx, item in enumerate(data_list):
        # Extract class information from the data item
        class_code = item.get('label', 'unknown')
        if isinstance(class_code, torch.Tensor):
            class_code = class_code.item()
        
        # Convert numeric label to string if needed
        if isinstance(class_code, (int, float)):
            class_code = f"class_{int(class_code)}"
        
        # Get human-readable class name
        if class_code in synset_map:
            class_name = synset_map[class_code]
        else:
            class_name = f"Unknown ({class_code})"
            missing_codes.add(class_code)
        
        # Count occurrences of each class
        class_counts[class_name] += 1
        
        # Create mapping entry
        mapping_data.append({
            'processing_trial_idx': processing_idx,
            'class_code': class_code,
            'class_name': class_name,
            'stc_filename_base': f"sources_subject_{subject_id}_trial_{processing_idx}",
            'subject_id': subject_id
        })
    
    # Print missing codes for debugging
    if missing_codes:
        print(f"  Missing codes in synset file for subject {subject_id}:")
        for code in sorted(missing_codes):
            print(f"    {code}")
    
    # Create DataFrame
    df = pd.DataFrame(mapping_data)
    
    # Print class distribution for verification
    print(f"  Class distribution for subject {subject_id}:")
    for class_name, count in sorted(class_counts.items()):
        print(f"    {class_name}: {count} trials")
    
    return df

###############################################################################
# Main function
###############################################################################
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load synset mapping
    print("Loading synset mapping...")
    synset_map = load_synset_mapping(SYNSET_FILE)
    
    if not synset_map:
        print("Warning: No synset mapping loaded. Class names will show as 'Unknown'")
        return
    
    # Debug: Search for some of the missing codes
    missing_codes_to_search = ['n01443537', 'n02099601', 'n02099712', 'n02106166', 'n02124075']
    debug_synset_search(SYNSET_FILE, missing_codes_to_search)
    
    # Process each subject
    all_mappings = []
    
    for subject_id in SUBJECT_IDS:
        # Recover mapping for this subject
        subject_df = recover_subject_mapping(subject_id, DATA_FOLDER, synset_map)
        
        if subject_df.empty:
            print(f"Skipping subject {subject_id} (no data)")
            continue
        
        # Save individual subject mapping
        subject_output_file = os.path.join(OUTPUT_DIR, f"subject_{subject_id}_trial_mapping.csv")
        subject_df.to_csv(subject_output_file, index=False)
        print(f"  Saved mapping for subject {subject_id}: {subject_output_file}")
        
        # Add to combined mapping
        all_mappings.append(subject_df)
    
    # Create combined mapping file
    if all_mappings:
        combined_df = pd.concat(all_mappings, ignore_index=True)
        combined_output_file = os.path.join(OUTPUT_DIR, "all_subjects_trial_mapping.csv")
        combined_df.to_csv(combined_output_file, index=False)
        print(f"\nSaved combined mapping: {combined_output_file}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"  Total subjects processed: {len(all_mappings)}")
        print(f"  Total trials mapped: {len(combined_df)}")
        print(f"  Unique classes found: {len(combined_df['class_name'].unique())}")
        
        # Show class distribution across all subjects
        print(f"\nOverall class distribution:")
        class_dist = combined_df['class_name'].value_counts()
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count} trials")
    
    else:
        print("No subjects processed successfully.")

###############################################################################
# Example usage functions
###############################################################################
def get_trials_for_class(subject_id, class_name, mapping_dir="./trial_mappings"):
    """
    Helper function to get all trial indices for a specific class and subject.
    
    Args:
        subject_id: Subject ID (0-15)
        class_name: Class name (e.g., 'German shepherd')
        mapping_dir: Directory containing mapping CSV files
    
    Returns:
        list: Processing trial indices for the specified class
    """
    mapping_file = os.path.join(mapping_dir, f"subject_{subject_id}_trial_mapping.csv")
    
    if not os.path.exists(mapping_file):
        print(f"Mapping file not found: {mapping_file}")
        return []
    
    df = pd.read_csv(mapping_file)
    class_trials = df[df['class_name'] == class_name]['processing_trial_idx'].tolist()
    return class_trials

def get_stc_files_for_class(subject_id, class_name, mapping_dir="./trial_mappings"):
    """
    Helper function to get all .stc filenames for a specific class and subject.
    
    Args:
        subject_id: Subject ID (0-15)
        class_name: Class name (e.g., 'German shepherd')
        mapping_dir: Directory containing mapping CSV files
    
    Returns:
        list: Base filenames for .stc files (add -lh.stc and -rh.stc for full paths)
    """
    mapping_file = os.path.join(mapping_dir, f"subject_{subject_id}_trial_mapping.csv")
    
    if not os.path.exists(mapping_file):
        print(f"Mapping file not found: {mapping_file}")
        return []
    
    df = pd.read_csv(mapping_file)
    stc_files = df[df['class_name'] == class_name]['stc_filename_base'].tolist()
    return stc_files

if __name__ == "__main__":
    main()
    
    # Example usage:
    print("\n" + "="*50)
    print("Example Usage:")
    print("="*50)
    
    # Example 1: Get trials for a specific class
    example_subject = 0
    example_class = "German shepherd"  # Update with actual class name from your data
    
    trials = get_trials_for_class(example_subject, example_class)
    if trials:
        print(f"\nSubject {example_subject}, Class '{example_class}':")
        print(f"  Trial indices: {trials[:5]}...")  # Show first 5
        print(f"  Total trials: {len(trials)}")
        
        stc_files = get_stc_files_for_class(example_subject, example_class)
        print(f"  Example .stc files:")
        for i, base_name in enumerate(stc_files[:3]):  # Show first 3
            print(f"    {base_name}-lh.stc")
            print(f"    {base_name}-rh.stc")
    else:
        print(f"No trials found for subject {example_subject}, class '{example_class}'")