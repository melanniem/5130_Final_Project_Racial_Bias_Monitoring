import sys

import pandas as pd
import logging
import os
from datetime import datetime
from pathlib import Path

Path("./logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/running.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataPersistence:
    def __init__(self, DATA_PATH: Path, input_path="prompts_output.csv", output_path="llm_outputs.csv"):
        self.input_path = os.path.join(DATA_PATH, input_path)
        self.output_path = os.path.join(DATA_PATH, output_path)
        self._append_count = 0

        if os.path.exists(self.output_path):
            self.df = pd.read_csv(self.output_path)
            logger.info(f"Loaded existing file llm_outputs from {self.output_path} ({len(self.df)} rows)")
        elif os.path.exists(self.input_path):
            self.df = pd.read_csv(self.input_path)
            for col in ["model", "temperature", "score", "rationale", "raw_response", "timestamp"]:
                if col not in self.df.columns:
                    self.df[col] = None
            # Cast to correct types
            self.df["score"] = pd.to_numeric(self.df["score"], errors="coerce")
            self.df["temperature"] = pd.to_numeric(self.df["temperature"], errors="coerce")
            self.df["model"] = self.df["model"].astype(object)
            self.df["rationale"] = self.df["rationale"].astype(object)
            self.df["raw_response"] = self.df["raw_response"].astype(object)
        else:
            self.df = pd.DataFrame(columns=[
                "name_id", "job_title_id", "race_group", "model", "temperature", "score", "rationale", "raw_response", "timestamp"
            ])
            logger.info("No input file found, creating new file llm_outputs")

    def append_result(self, result: dict, save_every: int = 20):
        """
        Appends result to llm_outputs dataframe
        :param result:
        :return:
        """
        name_id = result.get("name_id")
        job_title_id = result.get("job_title_id")
        race_group = result.get("race_group")

        match = self.df[
            (self.df["name_id"] == name_id)
            & (self.df["job_title_id"] == job_title_id)
            & (self.df["race_group"] == race_group)
            ]

        if match.empty:
            logger.warning(
                f"No matching row found for name_id={name_id}, job_title_id={job_title_id}, race_group={race_group}")
            return

        try:
            idx = match.index[0]
            for col in ["model", "temperature", "score", "rationale", "raw_response"]:
                self.df.at[idx, col] = result.get(col)
            self.df.at[idx, "timestamp"] = result.get("timestamp", datetime.now())

            self._append_count += 1
            if self._append_count % save_every == 0:
                self.save()
                logger.info(f"Auto saved at {len(self.df)} rows")
            logger.info(
                f"Updated row for name_id={name_id}, job_title_id={job_title_id}, race_group={race_group}")

        except Exception as e:
            logger.error(f"Error updating resume for name_id: {name_id} in llm_outputs: {e}")

    def reset_scores(self):
        """
        Clear all prior scores so only the current run's results are kept.
        """
        for col in ["model", "temperature", "score", "rationale", "raw_response", "timestamp"]:
            self.df[col] = None
        logger.info("Reset all scores for new run")
    
    def append_batch(self, results: list):
        """
        Append a list of results to llm_outputs dataframe
        :param result:
        :return:
        """
        for result in results:
            self.append_result(result)
        self.save()
        logger.info(f"Batch complete - total rows saved: {len(self.df)}")

    def save(self):
        """
        Saves results to csv
        :return:
        """
        dir_name = os.path.dirname(self.output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)



