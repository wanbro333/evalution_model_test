"""项目级配置：路径、随机种子与业务假设。"""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = MODEL_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
PAPER_DIR = MODEL_DIR / "paper"
DB_PATH = MODEL_DIR / "credit_data.db"

RANDOM_SEED = 20200713

BUDGET_PROBLEM_1 = 8000.0
BUDGET_PROBLEM_2 = 10000.0
LOAN_MIN = 10.0
LOAN_MAX = 100.0
RATE_MIN = 0.04
RATE_MAX = 0.15

LGD = 0.60
FUNDING_COST = 0.025

GRADE_THRESHOLDS = {
    "A": (0.00, 0.10),
    "B": (0.10, 0.20),
    "C": (0.20, 0.40),
    "D": (0.40, 1.01),
}

SHOCK_FACTORS = {
    "严重负面": 1.30,
    "中度负面": 1.15,
    "其他": 1.00,
    "未知": 1.00,
    "正面/轻影响": 0.90,
}


def ensure_directories():
    for path in (OUTPUT_DIR, CHART_DIR, PAPER_DIR):
        path.mkdir(parents=True, exist_ok=True)
