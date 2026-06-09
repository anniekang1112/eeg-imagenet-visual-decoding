# EEG-ImageNet Source Localization and Feature-Based Classification

This repository contains the analysis pipeline used for source-localized EEG decoding experiments on the EEG-ImageNet dataset.

The workflow includes:

1. Trial mapping recovery
2. Source-space ROI selection
3. Feature extraction from source-localized signals
4. Classification experiments
5. Statistical comparison of ROI configurations

---

## Repository Structure

```text
trial_mapping_recovery.py
dipole_selection_roi.py
dipole_selection_roi_all_class.py

train_paper.py
train_classifier.py

train_gamma_ll_baselines.py
train_ll_alone_24+50.py

pair_test_ll24_vs_ll50_full_auto.py
```

---

## Pipeline Overview

### 1. Trial Mapping

**trial_mapping_recovery.py**

Creates trial mapping tables linking processed trials to ImageNet class labels.

Output:

```text
trial_mappings/
```

---

### 2. ROI Selection

#### 8-Class ROI Selection

**dipole_selection_roi.py**

Generates ROI selections for the 8-class subset used in the primary feature comparison experiments.

Output:

```text
roi_selections/
```

#### All-Class ROI Selection

**dipole_selection_roi_all_class.py**

Generates ROI selections for all EEG-ImageNet classes.

Output:

```text
roi_selections_all_classes/
```

ROI selection is performed using the fsaverage source space and representative vertices selected according to ROI-center proximity.

---

### 3. Main Feature Comparison Experiments

**train_paper.py**

Runs the feature comparison experiments reported in the manuscript using the 8-class subset.

Evaluated feature groups:

* Five-band spectral power
* Gamma power (30–120 Hz)
* Gamma power (70–150 Hz)
* Line length
* Catch22
* Phase-phase coupling
* Phase-amplitude coupling
* Amplitude-amplitude coupling
* Combined coupling features

Classifier:

* Random Forest

Output:

```text
outputs_train_paper/
```

---

### 4. Classifier Selection

**train_classifier.py**

Compares multiple classifiers using the same source-localized EEG representation.

Evaluated classifiers:

* Random Forest
* Linear SVM
* Logistic Regression
* K-Nearest Neighbors
* Ridge Classifier

Output:

```text
outputs_classifier_selection/
```

---

### 5. Baseline Experiments

#### Gamma and Line-Length Baselines

**train_gamma_ll_baselines.py**

Evaluates:

* gamma24
* ll24
* gamma50
* ll50

Across:

* All-80
* Coarse-40
* Fine-40

Outputs:

```text
gamma_ll_baselines_results/
gamma_ll_baselines_features/
gamma_ll_baselines_plots/
```

#### Pure Line-Length Baselines

**train_ll_alone_24+50.py**

Evaluates:

* ll24
* ll50

Across:

* All-80
* Coarse-40
* Fine-40

Outputs:

```text
ll_baselines_results/
ll_baselines_features/
ll_baselines_plots/
```

---

### 6. Statistical Testing

**pair_test_ll24_vs_ll50_full_auto.py**

Performs paired Wilcoxon signed-rank tests comparing:

* 24-ROI line-length features
* 50-ROI line-length features

Across:

* All-80
* Coarse-40
* Fine-40

Outputs:

* Subject-level comparison tables
* Wilcoxon test statistics
* p-values

---

## Requirements

Typical dependencies include:

```bash
pip install numpy pandas scipy scikit-learn matplotlib mne pycatch22
```

---

## Dataset

Experiments were conducted using the EEG-ImageNet dataset.

Source localization was performed using MNE-Python and the fsaverage template anatomy.

---

## Citation

If you use this code, please cite the associated publication describing the source-localized EEG decoding framework.
