import google.generativeai as genai
import json
import time
from datetime import datetime

class Gemini:
    def __init__(self, api_key, model="gemini-1.5-pro", temperature=0):
        genai.configure(api_key=api_key, model=model)
        self.temperature = temperature
        self.model = genai.GenerativeModel(model)


    #Raw API Call
    def call_model(self, prompt):

        response = (self.model.generate(
            prompt,
            generation_config={
                "temperature":self.temperature
            }
        }
        return response.text.strip()

    #Retry Logic to handle a failed API call
    def score_resume(self, prompt, resume_id=None, race_group=None, retries=3):

        for attempt in range(retries):

            try:

                text = self.call_model(prompt)

                parsed = json.loads(text)

                return {
                    "resume_id":resume_id,
                    "race_group":race_group,
                    "model": self.model_name,
                    "temperature":self.temperature,
                    "score": parsed["score"]
                    "rationale": parsed["rationale"]
                    "raw_response": parsed["raw_response"]
                    "timestamp": datetime.now()
                }
            except Exception as e:
                if attempt == retries - 1:
                    return {
                        "resume_id": resume_id,
                        "race_group": race_group,
                        "model": self.model_name,
                        "score": None
                        "rationale": None
                        "raw_response": str(e)
                        "timestamp": datetime.now()
                    }
                time.sleep(2)



    def score_batch(self, prompt_list):

        results = []

        for item in prompt_list:
            result = self.score_resume(
                prompt=item["prompt"],
                resume_id=item["resume_id"],
                race_group=item["race_group"]
            )

            results.append(result)

        return results