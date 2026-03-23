from input_layer import input as input_layer
from data_persistence import data_persistence
#from model_interface import gemini_interface
from model_interface import ollama_interface
from prompt_layer import prompt_standardization
from bias_analysis import bias_quantification
import logging
import pandas as pd
from pathlib import Path
#from dotenv import load_dotenv
import os
import json
import random

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

        # Initialize Gemini
        # load_dotenv()
        # client = gemini_interface.Gemini(api_key=os.environ['GEMINI_API_KEY'])

        # Initialize Ollama LLM
        client = ollama_interface.OllamaQwen()
        logger.info("Initialized Ollama Qwen client")

        # Data Logging and Resume Scoring
        prompt_list = prompt_df.apply(lambda row: {
            "prompt": row["prompt"],
            "resume_id": row["resume_id"],
            "name_id": row["name_id"],
            "job_title_id": row["job_title_id"],
            "race_group": row["race_group"]
        }, axis=1).tolist()

        persistence = data_persistence.DataPersistence(
            DATA_PATH=RESULTS_PATH,
            input_path="prompts_output.csv",
            output_path="llm_outputs.csv"
        )

        logger.info(f"Scoring {len(prompt_list)} prompts via LLM...")
        # results = client.score_batch(prompt_list) # For running whole pipeline
        #results = client.score_batch(prompt_list[:30]) # Test few prompts
        #job_prompts = [p for p in prompt_list if p["job_title_id"] == 0]
        #results = client.score_batch(random.sample(job_prompts, 50))  # Test random prompts

        per_group = 10
        job_id = 1
        sampled = []
        for race in ["White", "Black or African American", "Hispanic", "Asian or Pacific Islander"]:
            race_prompts = [p for p in prompt_list if p["race_group"] == race and str(p["job_title_id"]) == str(job_id)]
            sampled.extend(random.sample(race_prompts, per_group))
        random.shuffle(sampled)
        results = client.score_batch(sampled)

        logger.info(f"Scoring complete. {sum(1 for r in results if r['score'] is not None)} succeeded, "
                    f"{sum(1 for r in results if r['score'] is None)} failed.")

        persistence.append_batch(results)
        logger.info(f"Saved results via DataPersistence to {persistence.output_path}")
    else:
        logger.info("Skipping LLM scoring, using existing results from results/llm_outputs.csv")

    # Bias Quantification Layer
    logger.info("Starting bias quantification analysis...")
    quantifier = bias_quantification.BiasQuantification(
        data_path="results",
        input_file="llm_outputs.csv",
        output_dir="results",
        threshold=75.0,
    )
    quantifier.run_bias_quantification_layer()
    logger.info("Bias quantification complete.")