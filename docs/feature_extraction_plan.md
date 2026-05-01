# Feature Extraction Plan

## Goal

Create the first reproducible EEG feature table for DEAP valence/arousal classification.

The first baseline will use simple, interpretable channel-wise bandpower features.

## Input

DEAP preprocessed Python files:

- `s01.dat` to `s32.dat`

Each subject file contains:

- `data`
- `labels`

Expected data shape:

```text
40 trials × 40 channels × 8064 samples