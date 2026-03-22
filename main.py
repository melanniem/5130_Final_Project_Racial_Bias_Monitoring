from input_layer import input
from data_persistence import data_persistence
from model_interface import gemini_interface
from prompt_layer import prompt_standardization
import logging
import pandas as pd
from pathlib import Path

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


if __name__ == "__main__":
    # Load input data
    df = input.run_input_layer()
    print(df.head())


