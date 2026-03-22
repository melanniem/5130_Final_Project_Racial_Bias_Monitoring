from ollama import chat
import json
import time
from datetime import datetime


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
                if attempt == retries - 1:
                    return {
                        "resume_id": resume_id,
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
                time.sleep(2)

    def score_batch(self, prompt_list):
        results = []
        for item in prompt_list:
            result = self.score_resume(
                prompt=item["prompt"],
                resume_id=item["resume_id"],
                race_group=item["race_group"],
                name_id=item.get("name_id"),
                job_title_id=item.get("job_title_id"),
            )
            results.append(result)
            time.sleep(0.5)

        return results
