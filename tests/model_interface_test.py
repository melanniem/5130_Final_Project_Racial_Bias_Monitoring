"""
Unit tests for gemini_interface.py and ollama_interface.py (Model Interface Layer)

All tests mock call_model, so no real LLM (here: Ollama or Gemini) is needed.
Covers: JSON parsing, markdown fence stripping, retry logic, failure fallback,
required output keys, and score_batch accumulation.
"""

import json
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Stub out the google.generativeai dependency before any project imports.
# The real library fails on Python 3.14 due to a broken _cffi_backend build!
def _make_genai_stub():
    genai = types.ModuleType("google.generativeai")
    genai.configure = MagicMock()
    genai.GenerativeModel = MagicMock(return_value=MagicMock())
    return genai

_google = types.ModuleType("google")
_genai_stub = _make_genai_stub() # Inject stub for test cases
sys.modules.setdefault("google", _google)
sys.modules.setdefault("google.generativeai", _genai_stub)
_google.generativeai = _genai_stub

from model_interface.ollama_interface import OllamaQwen  # noqa: E402
from model_interface.gemini_interface import Gemini       # noqa: E402
# noqa and E402 tells linter to ignore warnings for import lines not being at the top of the file

# === Helpers ===
VALID_RESPONSE = json.dumps({"score": 78, "rationale": "Strong candidate", "raw_response": None})

REQUIRED_KEYS = {"race_group", "name_id", "job_title_id", "model",
                 "temperature", "score", "rationale", "timestamp"}

def make_ollama():
    return OllamaQwen(model="qwen2.5:7b", temperature=0)

def make_gemini():
    """
    Build a Gemini instance without hitting the real API.
    """
    with patch("model_interface.gemini_interface.genai"):
        g = Gemini(api_key="fake-key", model="models/gemini-2.5-flash", temperature=0)
    return g

# Parametrize shared behaviour across both classes
@pytest.fixture(params=["ollama", "gemini"])
def model(request):
    if request.param == "ollama":
        return make_ollama()
    return make_gemini()

# === HAPPY PATH: Test function "score_resume" ===

class TestScoreResumeSuccess:
    def test_returns_score_and_rationale(self, model):
        with patch.object(model, "call_model", return_value=VALID_RESPONSE):
            result = model.score_resume("some prompt", race_group="Black", name_id=1, job_title_id=0)
        assert result["score"] == 78
        assert result["rationale"] == "Strong candidate"

    def test_required_keys_present(self, model):
        with patch.object(model, "call_model", return_value=VALID_RESPONSE):
            result = model.score_resume("prompt")
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_metadata_passed_through(self, model):
        with patch.object(model, "call_model", return_value=VALID_RESPONSE):
            result = model.score_resume(
                "prompt", resume_id=5, race_group="Asian", name_id=3, job_title_id=2
            )
        assert result["resume_id"] == 5
        assert result["race_group"] == "Asian"
        assert result["name_id"] == 3
        assert result["job_title_id"] == 2

    def test_timestamp_is_iso_format(self, model):
        with patch.object(model, "call_model", return_value=VALID_RESPONSE):
            result = model.score_resume("prompt")
        # Should not raise
        datetime.fromisoformat(result["timestamp"])

    def test_strips_markdown_json_fence(self, model):
        fenced = f"```json\n{VALID_RESPONSE}\n```"
        with patch.object(model, "call_model", return_value=fenced):
            result = model.score_resume("prompt")
        assert result["score"] == 78

    def test_strips_plain_code_fence(self, model):
        fenced = f"```\n{VALID_RESPONSE}\n```"
        with patch.object(model, "call_model", return_value=fenced):
            result = model.score_resume("prompt")
        assert result["score"] == 78

    def test_call_model_called_once_on_success(self, model):
        with patch.object(model, "call_model", return_value=VALID_RESPONSE) as mock_call:
            model.score_resume("prompt")
        mock_call.assert_called_once()

# === RETRY & FAILURE BEHAVIOUR: Test function "score_resume" ===

class TestScoreResumeRetries:
    def test_retries_on_bad_json(self, model):
        """
        Should retry when call_model returns invalid JSON, then succeed.
        """
        with patch.object(
            model, "call_model",
            side_effect=["not json", "still bad", VALID_RESPONSE]
        ):
            result = model.score_resume("prompt", retries=3)
        assert result["score"] == 78

    def test_returns_none_score_after_all_retries_fail(self, model):
        """Exhausting all retries should return a result dict with score=None."""
        with patch.object(model, "call_model", return_value="invalid json"):
            result = model.score_resume("prompt", retries=3)
        assert result["score"] is None
        assert result["rationale"] is None

    def test_failure_result_has_required_keys(self, model):
        with patch.object(model, "call_model", return_value="bad"):
            result = model.score_resume("prompt", retries=1)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_raw_response_contains_error_on_failure(self, model):
        """raw_response should hold the exception string when all retries fail."""
        with patch.object(model, "call_model", side_effect=Exception("timeout")):
            result = model.score_resume("prompt", retries=1)
        assert "timeout" in result["raw_response"]

    def test_call_model_called_retries_times_on_failure(self, model):
        with patch.object(model, "call_model", return_value="bad") as mock_call:
            model.score_resume("prompt", retries=3)
        assert mock_call.call_count == 3

    def test_succeeds_on_second_attempt(self, model):
        with patch.object(
            model, "call_model",
            side_effect=["bad json", VALID_RESPONSE]
        ) as mock_call:
            result = model.score_resume("prompt", retries=3)
        assert result["score"] == 78
        assert mock_call.call_count == 2

# === Test function "score_batch" ===

PROMPT_LIST = [
    {"prompt": "p1", "race_group": "Black",    "name_id": 0, "job_title_id": 0},
    {"prompt": "p2", "race_group": "White",    "name_id": 1, "job_title_id": 1},
    {"prompt": "p3", "race_group": "Hispanic", "name_id": 2, "job_title_id": 0},
]

class TestScoreBatch:
    def _mock_persistence(self):
        mock = MagicMock()
        mock.append_batch = MagicMock()
        return mock

    def test_returns_one_result_per_prompt(self, model):
        with patch.object(model, "call_model", return_value=VALID_RESPONSE), \
             patch("model_interface.ollama_interface.data_persistence.DataPersistence" if hasattr(model, 'temperature') and isinstance(model, OllamaQwen) else
                   "model_interface.gemini_interface.data_persistence.DataPersistence",
                   return_value=self._mock_persistence()):
            results = model.score_batch(PROMPT_LIST)
        assert len(results) == len(PROMPT_LIST)

    def test_all_results_have_scores_on_success(self, model):
        with patch.object(model, "call_model", return_value=VALID_RESPONSE), \
             patch("model_interface.ollama_interface.data_persistence.DataPersistence" if isinstance(model, OllamaQwen) else
                   "model_interface.gemini_interface.data_persistence.DataPersistence",
                   return_value=self._mock_persistence()):
            results = model.score_batch(PROMPT_LIST)
        assert all(r["score"] == 78 for r in results)

    def test_failed_scores_excluded_from_persistence(self, model):
        """
        Results with score=None should not be passed to append_batch.
        """
        mock_pers = self._mock_persistence()
        with patch.object(model, "call_model", return_value="bad json"), \
             patch("model_interface.ollama_interface.data_persistence.DataPersistence" if isinstance(model, OllamaQwen) else
                   "model_interface.gemini_interface.data_persistence.DataPersistence",
                   return_value=mock_pers):
            model.score_batch(PROMPT_LIST)
        # append_batch should never be called since all scores are None
        mock_pers.append_batch.assert_not_called()

    def test_empty_prompt_list_returns_empty(self, model):
        mock_pers = self._mock_persistence()
        with patch.object(model, "call_model", return_value=VALID_RESPONSE), \
             patch("model_interface.ollama_interface.data_persistence.DataPersistence" if isinstance(model, OllamaQwen) else
                   "model_interface.gemini_interface.data_persistence.DataPersistence",
                   return_value=mock_pers):
            results = model.score_batch([])
        assert results == []