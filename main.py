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
RACE_GROUPS = ["White", "Black or African American", "Hispanic", "Asian or Pacific Islander"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def sample_balanced(prompt_list, job_id, per_group):
    """
    Sample exactly per_group prompts per race for the given job_id.
    """
    sampled = []
    for race in RACE_GROUPS:
        race_prompts = [
            p for p in prompt_list
            if p["race_group"] == race and str(p["job_title_id"]) == str(job_id)
        ]
        if len(race_prompts) < per_group:
            logger.warning(f"Only {len(race_prompts)} prompts available for {race} "
                           f"(job {job_id}), need {per_group}")
        sampled.extend(random.sample(race_prompts, min(per_group, len(race_prompts))))
    random.shuffle(sampled)
    return sampled
 

def score_with_retries(client, prompt_pool, per_group, job_id, max_retries=3):
    """
    Score prompts and retry failures until every race group has exactly
    per_group successful scores, or retries are exhausted.
 
    Returns only the successful results which are guaranteed balanced if enough
    source prompts exist and the LLM doesn't fail persistently.
    """
    # Track successful results per group and which prompt keys have been used
    successful = {race: [] for race in RACE_GROUPS}
    used_keys = set()  # (resume_id, name_id) pairs already attempted
 
    # Build per-race prompt pools (all available prompts for this job)
    race_pools = {}
    for race in RACE_GROUPS:
        race_pools[race] = [
            p for p in prompt_pool
            if p["race_group"] == race and str(p["job_title_id"]) == str(job_id)
        ]
        random.shuffle(race_pools[race])
 
    for attempt in range(1, max_retries + 1):
        # Figure out how many more successes each group needs
        needed = {
            race: per_group - len(successful[race])
            for race in RACE_GROUPS
        }
 
        # If every group is full, we're done
        if all(n <= 0 for n in needed.values()):
            break
 
        # Build this attempt's batch from unused prompts
        batch = []
        for race in RACE_GROUPS:
            if needed[race] <= 0:
                continue
            available = [
                p for p in race_pools[race]
                if (p["resume_id"], p["name_id"]) not in used_keys
            ]
            pick = available[:needed[race]]
            batch.extend(pick)
            for p in pick:
                used_keys.add((p["resume_id"], p["name_id"]))
 
        if not batch:
            logger.warning(f"Attempt {attempt}: no unused prompts left to retry")
            break
 
        random.shuffle(batch)
        logger.info(f"Attempt {attempt}: scoring {len(batch)} prompts "
                    f"(need {sum(n for n in needed.values() if n > 0)} more successes)")
 
        results = client.score_batch(batch)
 
        # Sort successes into the right group bucket
        for r in results:
            race = r["race_group"]
            if r["score"] is not None and len(successful[race]) < per_group:
                successful[race].append(r)
 
    # Log final counts
    for race in RACE_GROUPS:
        count = len(successful[race])
        status = "OK" if count == per_group else "SHORT"
        logger.info(f"  {race}: {count}/{per_group} [{status}]")
 
    # Flatten into a single list
    all_results = []
    for race in RACE_GROUPS:
        all_results.extend(successful[race])
    return all_results

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

        per_group = 50
        job_id = 0
        results = score_with_retries(
            client, prompt_list, per_group, job_id, max_retries=3
        )
        logger.info(f"Final balanced set: {len(results)} results")
        persistence.reset_scores()
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