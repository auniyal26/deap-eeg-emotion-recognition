# Results Card v0

## Project

**A Reproducible Benchmarking Pipeline for EEG-Based Valence and Arousal Classification Using the DEAP Dataset**

## Current Stage

Dataset access, loading, and first exploratory analysis are complete.

This project is currently in the pre-baseline stage. No classification results are reported yet.

## Dataset

The project uses the DEAP preprocessed Python format.

Current loaded dataset:

- Subjects: 32
- Trials per subject: 40
- Total trials: 1280
- Data shape per subject: trials × channels × samples
- Expected subject shape: 40 × 40 × 8064
- EEG channels used: first 32 channels

## Label Construction

Binary labels are created from DEAP self-ratings using:

- low: rating <= 5
- high: rating > 5

Tasks:

- binary valence classification
- binary arousal classification

## Dataset-Level Label Distribution

Using threshold > 5:

| Task | Low Count | High Count | High-Class Ratio |
|---|---:|---:|---:|
| Valence | 1011 | 269 | 0.210 |
| Arousal | 982 | 298 | 0.233 |

## Key Finding 1: Strong Class Imbalance

Both target variables are imbalanced.

The high class represents only:

- 21.0% of valence labels
- 23.3% of arousal labels

This means accuracy alone is not an appropriate primary metric.

Primary metrics should be:

- balanced accuracy
- macro F1

## Key Finding 2: Subject-Level Imbalance Varies Strongly

High-class ratio ranges:

| Task | Minimum High Ratio | Maximum High Ratio |
|---|---:|---:|
| Valence | 0.050 | 0.525 |
| Arousal | 0.000 | 0.475 |

This suggests that subject-independent evaluation may be difficult and must be interpreted carefully.

Some subjects contribute very few or no high-class examples for a target.

## Figures Generated

- `results/figures/s01_eeg_trial0_channels.png`
- `results/figures/s01_label_distribution.png`
- `results/figures/all_subject_valence_arousal_distribution.png`
- `results/figures/subjectwise_label_balance.png`

## Baseline Approach Locked

The first modelling baseline will use EEG-only bandpower features.

Features:

- delta: 1–4 Hz
- theta: 4–8 Hz
- alpha: 8–13 Hz
- beta: 13–30 Hz
- gamma: 30–45 Hz

Expected feature size:

- 32 EEG channels × 5 frequency bands = 160 features per trial

Initial models:

- Logistic Regression
- Linear SVM
- Random Forest

## Evaluation Plan

### Protocol 1: Random Trial Split

Used only as an optimistic sanity check.

Risk:

- Trials from the same subject can appear in both train and test sets.
- This can inflate performance due to subject-specific signal patterns.

### Protocol 2: GroupKFold by Subject

Main benchmark protocol.

Goal:

- Ensure subjects in the test fold are not present in training.

This is the more meaningful generalization test.

### Protocol 3: Leave-One-Subject-Out

Later extension if runtime and class balance allow.

## Claim Boundary

This project does not claim:

- state-of-the-art performance
- clinical emotion detection
- mind-reading
- general-purpose emotion recognition

The current goal is reproducible, transparent baseline benchmarking.

## Next Step

Implement feature extraction:

- load all subject files
- extract EEG channels
- compute channel-wise bandpower features
- create valence/arousal binary labels
- save feature table
- save feature metadata