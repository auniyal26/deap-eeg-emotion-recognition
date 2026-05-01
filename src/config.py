from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEAP_PYTHON_DIR = RAW_DIR / "data_preprocessed_python"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

SUBJECT_FILE = DEAP_PYTHON_DIR / "s01.dat"
SUBJECT_FILES = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

EEG_CHANNELS = 32
VALENCE_IDX = 0
AROUSAL_IDX = 1
BINARY_THRESHOLD = 5.0