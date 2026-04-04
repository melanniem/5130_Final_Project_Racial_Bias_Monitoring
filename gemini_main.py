from input_layer import input as input_layer
from data_persistence import data_persistence
from model_interface import ollama_interface, gemini_interface
from prompt_layer import prompt_standardization
from bias_analysis import bias_quantification
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import random
from collections import defaultdict

INPUT_PATH = "input_combinations.csv"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
RACE_GROUPS = ["White", "Black or African American", "Hispanic", "Asian or Pacific Islander", "Null Baseline"]
JOB_IDS = [0, 1, 2]
BATCH_SIZE = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "running.log"),
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
        prompt_df = prompt_standardization.run_prompt_layer(n_baseline=BATCH_SIZE//4)
        prompt_df = prompt_df.rename(columns={"identity": "race_group"})
        prompt_df.to_csv(RESULTS_PATH / "prompts_output.csv", index=False)
        logger.info(f"Generated {len(prompt_df)} prompts dataframe")

        prompt_list = prompt_df.apply(lambda row: {
            "prompt": row["prompt"],
            "name_id": row["name_id"],
            "job_title_id": row["job_title_id"],
            "race_group": row["race_group"]
        }, axis=1).tolist()


        # Balanced Sample for Testing
        grouped = defaultdict(list)
        for item in prompt_list:
            grouped[item["race_group"]].append(item)

        N_PER_JOB = 2
        balanced_list = []
        for group in RACE_GROUPS:
            group_items = grouped[group]
            by_job = defaultdict(list)
            for item in group_items:
                by_job[item["job_title_id"]].append(item)
            for job_id in JOB_IDS:
                balanced_list.extend(by_job[job_id][:N_PER_JOB])

        balanced_df = pd.DataFrame(balanced_list)
        balanced_df.to_csv(RESULTS_PATH / "balanced_prompts.csv", index=False)
        logger.info(f"Saved {len(balanced_df)} to sample_prompts.csv")

        # Initialize Gemini LLM
        load_dotenv()
        client = gemini_interface.Gemini(api_key=os.environ['GEMINI_API_KEY'], cost_limit=5.00)
        logger.info("Initialized Gemini client")

        results = client.score_batch(balanced_list) # Running test sample

        # logger.info(f"Scoring {len(prompt_list)} prompts via Gemini...")
        # results = client.score_batch(prompt_list) # Running full pipeline

        logger.info(f"Scoring complete. {sum(1 for r in results if r['score'] is not None)} succeeded, "
                    f"{sum(1 for r in results if r['score'] is None)} failed.")

        # Log API usage
        usage = client.get_usage_summary()
        logger.info(f"API Usage - Calls: {usage['total_api_calls']}, "
                    f"Input tokens: {usage['total_input_tokens']}, "
                    f"Output tokens: {usage['total_output_tokens']}")
        usage_df = pd.DataFrame([usage])
        usage_df.to_csv(RESULTS_PATH / "api_usage.csv", index=False)

    # Bias Quantification Layer
    llm_output_file = RESULTS_PATH / "llm_outputs.csv"
    if not llm_output_file.exists():
        logger.error(f"File not found: {llm_output_file}. Run LLM scoring first.")
        exit(1)

    scored_df = pd.read_csv(llm_output_file)
    scored_count = scored_df["score"].notna().sum()
    total_count = len(scored_df)
    logger.info(f"Found {scored_count}/{total_count} scored rows in llm_outputs.csv")

    if scored_count == 0:
        logger.error("No scored results found. Cannot run bias analysis.")
        exit(1)

    logger.info("Starting bias quantification analysis...")
    quantifier = bias_quantification.BiasQuantification(
        data_path="results",
        input_file="llm_outputs.csv",
        output_dir="results",
        threshold=60.0,
    )
    quantifier.run_bias_quantification_layer()
    logger.info("Bias quantification complete.")