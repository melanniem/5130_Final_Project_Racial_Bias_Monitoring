from ollama import chat
import json
import time
from datetime import datetime
from data_persistence import data_persistence
from pathlib import Path
import logging
import sys

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

class OllamaQwen:
    def __init__(self, model="qwen2.5:7b", temperature=0):
        self.model = model
        self.temperature = temperature

    # Raw API Call
    def call_model(self, prompt):
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response.message.content.strip()

    # Retry Logic to handle a failed API call
    def score_resume(self, prompt, resume_id=None, race_group=None, name_id=None, job_title_id=None, retries=3):
        for attempt in range(retries):
            try:
                text = self.call_model(prompt)
                # Strip markdown json fences if present
                text = text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)

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
                is_rate_limit = "429" in error_str or "too many requests" in error_str or "resource exhausted" in error_str
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
                if is_rate_limit:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(f"Rate limited. Attempt {attempt + 1}/{retries}. Waiting {wait:.1f}s...")
                else:
                    wait = 1
                    logger.warning(f"Attempt {attempt + 1}/{retries} failed for name_id={name_id}: {e}")
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

            time.sleep(0.002)

        # save any remaining results
        if batch:
            persistence.append_batch(batch)
        return results
