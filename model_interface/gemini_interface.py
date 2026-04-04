from google import generativeai as genai
import json
import time
from datetime import datetime
from data_persistence import data_persistence
from pathlib import Path
import logging
import sys
import random

RESULTS_PATH = Path("results")
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

class Gemini:
    def __init__(self, api_key, model="models/gemini-2.5-flash", temperature=0, cost_limit=5.0):
        genai.configure(api_key=api_key)
        self.temperature = temperature
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        self.total_api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.cost_limit = cost_limit

        # Pricing per 1M tokens (update if model changes)
        self.pricing = {
            "models/gemini-2.5-flash": {"input": 0.15, "output": 0.60},
            "models/gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "models/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        }

    def _update_cost(self, input_tokens, output_tokens):
        rates = self.pricing.get(self.model_name, {"input": 0.15, "output": 0.60})
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        self.total_cost += cost
        return cost

    def _check_cost_limit(self):
        if self.total_cost >= self.cost_limit:
            logger.warning(f"Cost limit reached: ${self.total_cost:.4f} >= ${self.cost_limit:.2f}. Stopping.")
            return True
        return False

    # Raw API Call
    def call_model(self, prompt):
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )
        self.total_api_calls += 1
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
        self._update_cost(input_tokens, output_tokens)
        return response.text.strip()

    # Return Usage
    def get_usage_summary(self):
        return {
            "model": self.model_name,
            "total_api_calls": self.total_api_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "cost_limit_usd": self.cost_limit,
        }

    # Retry Logic to handle a failed API call
    def score_resume(self, prompt, resume_id=None, race_group=None, name_id=None, job_title_id=None, retries=3):
        for attempt in range(retries):
            try:
                text = self.call_model(prompt)
                # Strip markdown json fences if present
                text = text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)
                logger.info(f"200 OK | name_id={name_id}, job_title_id={job_title_id}")

                return {
                    "resume_id": resume_id,
                    "race_group": race_group,
                    "name_id": name_id,
                    "job_title_id": job_title_id,
                    "model": self.model,
                    "temperature": self.temperature,
                    "score": parsed.get("score"),
                    "rationale": parsed.get("rationale"),
                    "raw_response": parsed.get("raw_response"),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                error_str = str(e).lower()
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str or "resource exhausted" in error_str:
                    status = "429 Rate Limited"
                elif "400" in error_str or "invalid" in error_str:
                    status = "400 Bad Request"
                elif "401" in error_str or "unauthorized" in error_str:
                    status = "401 Unauthorized"
                elif "403" in error_str or "permission" in error_str:
                    status = "403 Forbidden"
                elif "500" in error_str:
                    status = "500 Server Error"
                elif "503" in error_str:
                    status = "503 Unavailable"
                else:
                    status = f"Error: {type(e).__name__}"

                logger.warning(f"{status} | name_id={name_id}, attempt {attempt + 1}/{retries}")

                if attempt == retries - 1:
                    return {
                        "race_group": race_group,
                        "name_id": name_id,
                        "job_title_id": job_title_id,
                        "model": self.model,
                        "temperature": self.temperature,
                        "score": None,
                        "rationale": None,
                        "raw_response": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                if "429" in error_str or "resource exhausted" in error_str:
                    wait = 2 ** attempt + random.uniform(0, 1)
                else:
                    wait = 1
                time.sleep(wait)

    def score_batch(self, prompt_list, save_every=1):
        persistence = data_persistence.DataPersistence(
            DATA_PATH=RESULTS_PATH,
            input_path="prompts_output.csv",
            output_path="llm_outputs.csv"
        )
        results = []
        batch = []
        for n, item in enumerate(prompt_list):
            if self._check_cost_limit():
                logger.warning(f"Stopped at prompt {n}/{len(prompt_list)}. "
                               f"Total cost: ${self.total_cost:.4f}")
                break

            result = self.score_resume(
                prompt=item["prompt"],
                race_group=item["race_group"],
                name_id=item.get("name_id"),
                job_title_id=item.get("job_title_id"),
            )
            results.append(result)
            if result["score"] is not None:
                batch.append(result)

            if len(batch) >= save_every:
                persistence.append_batch(batch)
                logger.info(f"Scored {n + 1}/{len(prompt_list)}")
                batch = []  # clear after saving

            time.sleep(2)

        # save any remaining results
        if batch:
            persistence.append_batch(batch)
        return results