from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "data" / "sakshi_ppg_20260611T074737_len148s.csv"
PLOT_DIR = PROJECT_DIR / "plots"
REPORT_DIR = PROJECT_DIR / "reports"
DECOMP_DIR = PROJECT_DIR / "decomposition"

TIME_COLUMN = "timestamp_ms"
TARGET_COLUMN = "ir_corrected"