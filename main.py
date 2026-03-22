# from input_layer import input as input_layer
# from data_persistence import data_persistence
# from model_interface import gemini_interface
# from prompt_layer import prompt_standardization
# from bias_analysis import bias_quantification
# import logging
# import pandas as pd
# from pathlib import Path
# from dotenv import load_dotenv
# import os
# import json
#
# INPUT_PATH = "input_combinations.csv"
# LOG_DIR = Path("logs")
# LOG_DIR.mkdir(exist_ok=True)
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_DIR / "pipeline.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
#
#
# if __name__ == "__main__":
#     # Load resume data
#     # == Input Layer ==
#     resume_df = input_layer.run_input_layer()
#     print(resume_df.head())
#     logger.info(f"Generated {len(resume_df)} resume dataframe")
#
#     # Load prompt data
#     # == Prompt Standardization Layer ==
#     prompt_df = prompt_standardization.run_prompt_layer()
#     print(prompt_df.head())
#     print(prompt_df.columns.tolist())
#     logger.info(f"Generated {len(prompt_df)} prompts dataframe")
#
#     # # Run model interface
#     # load_dotenv() # Load data variables
#     #
#     # # Initialize LLM
#     # client = gemini_interface.Gemini(api_key=os.environ['GEMINI_API_KEY'])
#     #
#     # # == Data Logging and Resume Scoring ==
#     # prompt_list = prompt_df.apply(lambda row: {
#     #     "prompt": row["prompt"],
#     #     "resume_id": row["resume_id"],
#     #     "name_id": row["name_id"],
#     #     "job_title_id": row["job_title_id"],
#     #     "race_group": row["identity"]
#     # }, axis=1).tolist()
#     # RESULTS_PATH = Path("results")
#     # RESULTS_PATH.mkdir(exist_ok=True)
#     #
#     # persistence = data_persistence.DataPersistence(
#     #     DATA_PATH=RESULTS_PATH,
#     #     input_path="prompts_output.csv",
#     #     output_path="llm_outputs.csv"
#     # )
#     #
#     # logger.info(f"Scoring {len(prompt_list)} prompts via Gemini...")
#     # # results = client.score_batch(prompt_list) # For running whole pipeline
#     # results = client.score_batch(prompt_list[:5]) # Test few prompts
#     #
#     # logger.info(f"Scoring complete. {sum(1 for r in results if r['score'] is not None)} succeeded, "
#     #             f"{sum(1 for r in results if r['score'] is None)} failed.")
#     # # Save results
#     # persistence.append_batch(results)
#     # logger.info(f"Saved results via DataPersistence to {persistence.output_path}")
#
#     # # == Bias Quantification Layer ==
#     #
#     # quantifier = bias_quantification.BiasQuantification(
#     # data_path="./data",
#     # input_file="llm_outputs.csv",
#     # output_dir="./evaluation_outputs",
#     # threshold=75.0,
#     # )
#     #
#     # quantifier.run_bias_quantification_layer()
#

from input_layer import input as input_layer
from data_persistence import data_persistence
from model_interface import gemini_interface
from prompt_layer import prompt_standardization
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import json

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
    # Load resume data
    resume_df = input_layer.run_input_layer()
    print(resume_df.head())
    logger.info(f"Generated {len(resume_df)} resume dataframe")

    # Load prompt data
    RESULTS_PATH = Path("results")
    RESULTS_PATH.mkdir(exist_ok=True)
    prompt_df = prompt_standardization.run_prompt_layer()
    prompt_df = prompt_df.rename(columns={"identity": "race_group"})
    prompt_df.to_csv(RESULTS_PATH / "prompts_output.csv", index=False)
    print(prompt_df.head())
    logger.info(f"Generated {len(prompt_df)} prompts dataframe")

    # Run model interface
    load_dotenv() # Load data variables

    # Initialize LLM
    client = gemini_interface.Gemini(api_key=os.environ['GEMINI_API_KEY'])

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

    logger.info(f"Scoring {len(prompt_list)} prompts via Gemini...")
    # results = client.score_batch(prompt_list) # For running whole pipeline
    results = client.score_batch(prompt_list[:2]) # Test few prompts
    for r in results:
        print(r)

    logger.info(f"Scoring complete. {sum(1 for r in results if r['score'] is not None)} succeeded, "
                f"{sum(1 for r in results if r['score'] is None)} failed.")
    # Save results
    persistence.append_batch(results)
    logger.info(f"Saved results via DataPersistence to {persistence.output_path}")
