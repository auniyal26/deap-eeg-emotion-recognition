# Limitations

## Dataset-specific labels

DEAP labels are based on participant self-ratings after viewing music videos. The models therefore learn patterns associated with these labels, not objective emotional states.

## Binary thresholding

The initial split uses rating > 5 as the high class. This is a common baseline choice but may discard nuance from the original 1–9 rating scale.

## Subject leakage risk

Random trial-level splits may overestimate performance because trials from the same subject can appear in both training and test sets.

Subject-independent evaluation is necessary for stronger generalization claims.

## No clinical claim

This project is not a medical or clinical emotion-recognition system.

## No state-of-the-art claim

This project prioritizes reproducibility and transparent benchmarking over leaderboard-style optimization.