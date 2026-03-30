from input_layer import input as input_layer
from data_persistence import data_persistence
from model_interface import ollama_interface
from prompt_layer import prompt_standardization
from bias_analysis import bias_quantification
import logging
import pandas as pd
from pathlib import Path
import os
import json
import random

INPUT_PATH = "input_combinations.csv"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
RACE_GROUPS = ["White", "Black or African American", "Hispanic", "Asian or Pacific Islander", "Null Baseline"]
JOB_IDS = [0, 1, 2]

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
    RESULTS_PATH = Path("results")
    RESULTS_PATH.mkdir(exist_ok=True)

    # Ask user whether to run LLM scoring
    run_llm = input("Run LLM scoring? (y/n): ").strip().lower() == "y"
    logger.info(f"User selected {'LLM scoring' if run_llm else 'skip to bias analysis'}")

    if run_llm:
        # Input Layer
        resume_df = input_layer.run_input_layer()
        print(resume_df.head())
        logger.info(f"Generated {len(resume_df)} resume dataframe")

        # Prompt Standardization Layer
        prompt_df = prompt_standardization.run_prompt_layer()
        prompt_df = prompt_df.rename(columns={"identity": "race_group"})
        prompt_df.to_csv(RESULTS_PATH / "prompts_output.csv", index=False)
        print(prompt_df.head())
        logger.info(f"Generated {len(prompt_df)} prompts dataframe")

        # Initialize Ollama LLM
        client = ollama_interface.OllamaQwen()
        logger.info("Initialized Ollama Qwen client")

        # Data Logging and Resume Scoring
        prompt_list = prompt_df.apply(lambda row: {
            "prompt": row["prompt"],
            "name_id": row["name_id"],
            "job_title_id": row["job_title_id"],
            "race_group": row["race_group"]
        }, axis=1).tolist()


        results = client.score_batch(prompt_list[:5])
        logger.info(f"Scoring complete. {sum(1 for r in results if r['score'] is not None)} succeeded, "
                    f"{sum(1 for r in results if r['score'] is None)} failed.")

    # Bias Quantification Layer
    logger.info("Starting bias quantification analysis...")
    quantifier = bias_quantification.BiasQuantification(
        data_path="results",
        input_file="llm_outputs.csv",
        output_dir="results",
        threshold=60.0,
    )
    quantifier.run_bias_quantification_layer()
    logger.info("Bias quantification complete.")