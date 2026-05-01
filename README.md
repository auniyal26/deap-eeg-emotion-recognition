# DEAP EEG Emotion Recognition

## Working Title

**A Reproducible Benchmarking Pipeline for EEG-Based Valence and Arousal Classification Using the DEAP Dataset**

## Project Goal

This project builds a reproducible benchmarking pipeline for EEG-based binary valence and arousal classification using the DEAP dataset.

The focus is not on claiming state-of-the-art performance. Instead, the project emphasizes:

- transparent preprocessing
- reproducible feature extraction
- classical baseline models
- leakage-safe evaluation
- clear metrics and experiment logs
- honest limitations

## Claim Boundary

This project does **not** claim to perform clinical emotion detection or mind-reading.

The models are evaluated only as supervised classifiers trained on DEAP self-rating labels. Results should be interpreted as dataset-specific machine learning baselines, not as general-purpose emotion recognition systems.

## Initial Tasks

- Binary valence classification
- Binary arousal classification

The initial binary split uses:

- low: rating <= 5
- high: rating > 5

## Planned Evaluation Protocols

1. Subject-dependent/random trial split
2. Subject-independent grouped evaluation
3. Leave-one-subject-out evaluation

The grouped and leave-one-subject-out settings are treated as more meaningful tests of generalization.

## Reproducibility

All scripts are designed to save metrics, figures, and logs under the `results/` and `docs/` folders.

Raw DEAP data is not included in this repository.