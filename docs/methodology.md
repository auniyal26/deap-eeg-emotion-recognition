# Methodology

## Study Framing

This project is designed as a reproducible benchmarking study for EEG-based valence and arousal classification using the DEAP dataset.

The goal is to establish transparent baseline results under clearly defined preprocessing, feature extraction, and evaluation settings.

## Dataset

The project uses the DEAP preprocessed Python data format.

Each subject file contains:

- physiological recordings
- trial-level self-ratings
- valence and arousal labels used for binary classification

Raw data is not committed to the repository.

## Label Construction

Valence and arousal ratings are binarized using a threshold of 5:

- low: rating <= 5
- high: rating > 5

This threshold is treated as a baseline design choice and may be revisited in sensitivity analysis.

## EEG Feature Extraction

The initial feature set uses channel-wise frequency bandpower from EEG channels.

Frequency bands:

- delta: 1–4 Hz
- theta: 4–8 Hz
- alpha: 8–13 Hz
- beta: 13–30 Hz
- gamma: 30–45 Hz

For 32 EEG channels and 5 frequency bands, this produces 160 features per trial.

## Evaluation Strategy

Two main evaluation settings are planned:

### Subject-dependent evaluation

Random train/test splits over trials.

This setting is useful as a sanity check but may overestimate performance due to subject-specific leakage.

### Subject-independent evaluation

Grouped evaluation where subjects in the test set are not present in the training set.

This setting is considered more meaningful for generalization.

## Metrics

Primary metrics:

- balanced accuracy
- macro F1

Secondary metrics:

- accuracy
- precision
- recall
- confusion matrix

## Claim Boundary

This project does not claim state-of-the-art performance or clinical validity.

The goal is transparent, reproducible baseline benchmarking.