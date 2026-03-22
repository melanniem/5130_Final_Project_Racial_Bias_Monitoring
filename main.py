from input_layer import input
from data_persistence import data_persistence
from model_interface import gemini_interface
from prompt_layer import prompt_standardization
import logging
import pandas as pd
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

INPUT_PATH = "input_combinations.csv"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_input_data(path):
    # Load input data using the input layer
    df = pd.read_csv(path)
    columns = [
        "resume_id", "name_id", "job_title_id",
        "name", "identity", "job_title",
        "resume_text", "job_description"
    ]
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in input data: {missing_columns}")

    logger.info(f"Input data loaded successfully with {len(df)} records from {path}.")
    logger.info(f"  Resumes:    {df['resume_id'].nunique()}")
    logger.info(f"  Names:      {df['name_id'].nunique()}")
    logger.info(f"  Jobs:       {df['job_title_id'].nunique()}")
    logger.info(f"  Identities: {df['identity'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    # Load input data
    df = load_input_data(INPUT_PATH)
    print(df.head())


