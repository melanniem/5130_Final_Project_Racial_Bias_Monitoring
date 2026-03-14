## CS5130 Project - Quantifying Racial Bias in Automated Resume Screening: An Audit Study of Google's Gemini API ##
from abc import ABC, abstractmethod
class ModelInterface(ABC):

    @abstractmethod
    def score_resume(self, prompt: str) -> dict:
        pass
