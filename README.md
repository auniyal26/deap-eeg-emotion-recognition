# DEAP EEG EDA

Goal:
- Inspect raw-loaded DEAP EEG trials corresponding to extreme continuous valence/arousal ratings.
- Verify whether extreme ratings show visible signal-level differences before further modelling.

Outputs:
- results/tables/raw_extreme_trial_selection.csv
- results/figures/raw_extreme_trials_eeg_snapshots.png
- results/figures/raw_extreme_trials_psd.png

Key findings:
- Several extreme-rating cases were selected from the same subject, especially s02.
- Raw-loaded EEG snapshots showed large amplitude differences and transient spikes in some extreme trials.
- Mean PSD curves showed large low-frequency power differences across selected trials.
- These differences may reflect subject-specific signal scale, artifacts, or preprocessing residue rather than affect-specific EEG structure.

Interpretation:
- Absolute bandpower features may be dominated by subject-specific amplitude/artifact effects.
- This helps explain why random-split results appeared better than subject-independent GroupKFold results.
- Before pursuing more complex affect-space models, normalized feature extraction should be tested.

Next:
- Implement log/relative/subject-normalized bandpower features.
- Rerun regression and EEG-aligned axis experiments using normalized feature tables.

## Reproducibility

All scripts are designed to save metrics, figures, and logs under the `results/` and `docs/` folders.

Raw DEAP data is not included in this repository.
