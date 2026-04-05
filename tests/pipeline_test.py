import pandas as pd
from pathlib import Path
from unittest.mock import patch

from input_layer.input import sample_names, build_combinations, JOB_DESCRIPTIONS
from prompt_layer.prompt_standardization import build_prompt, verify_prompt
from model_interface.gemini_interface import Gemini
from model_interface.ollama_interface import OllamaQwen

MOCK_RESPONSE = '{"score": 75, "rationale": "Strong candidate with relevant experience."}'

SYNTHETIC_RESUME = {
    "personal_info": {"email": "test@test.com", "phone": "555-0000"},
    "experience": [
        {
            "company": "Test Corp",
            "title": "Software Engineer",
            "dates": {"start": "2020-01-01", "end": "2023-01-01"},
            "responsibilities": ["Built scalable systems"],
        }
    ],
    "education": [],
    "skills": {},
}


def test_full_pipeline_synthetic():
    """
    Validate layers 1-3 using a small synthetic fixture dataset:
    4 names (1 per identity group) x 1 job = 4 rows
    """
    # Synthetic names DataFrame: 1 name per identity group
    synthetic_names_df = pd.DataFrame([
        {"name": "Alice Smith",   "first": "Alice",  "last": "Smith",   "identity": "white",    "mean.correct": 0.90},
        {"name": "Carlos Lopez",  "first": "Carlos", "last": "Lopez",   "identity": "hispanic", "mean.correct": 0.85},
        {"name": "Diana Johnson", "first": "Diana",  "last": "Johnson", "identity": "black",    "mean.correct": 0.80},
        {"name": "Wei Chen",      "first": "Wei",    "last": "Chen",    "identity": "asian",    "mean.correct": 0.88},
    ])

    # Use a single job to keep the fixture small
    test_jobs = {"Software Engineer": JOB_DESCRIPTIONS["Software Engineer"]}

    # Layer 1: Input
    names_sampled = sample_names(synthetic_names_df, names_per_group=1)
    df = build_combinations(SYNTHETIC_RESUME, names_sampled, test_jobs)

    assert not df.empty, "Combinations dataframe is empty"
    assert len(df) == 4, f"Expected 4 rows (4 names x 1 job), got {len(df)}"

    # Layer 2: Prompt - build prompts directly (run_prompt_layer reads from a CSV file)
    df['prompt'] = df.apply(
        lambda row: build_prompt(row['resume_text'], row['job_title'], row['job_description']),
        axis=1
    )

    expected_cols = ['name_id', 'job_title_id', 'name', 'identity', 'job_title', 'prompt']
    for col in expected_cols:
        assert col in df.columns, f"{col} missing from prompt output"
        assert df[col].notna().all(), f"{col} contains null values"

    for p in df['prompt']:
        assert "score" in p
        assert "rationale" in p
        assert "RESUME" in p
        assert "JOB DESCRIPTION" in p

    assert verify_prompt(df) is True

    # Layer 3: Model Interface - mock call_model so no real API calls are made
    with patch.object(Gemini, 'call_model', return_value=MOCK_RESPONSE), \
         patch.object(OllamaQwen, 'call_model', return_value=MOCK_RESPONSE):

        gemini = Gemini(api_key="mock_key")
        ollama_model = OllamaQwen()

        df['gemini_score'] = df.apply(
            lambda row: gemini.score_resume(
                row['prompt'],
                resume_id=row.get('name_id'),
                race_group=row.get('identity'),
                name_id=row.get('name_id'),
                job_title_id=row.get('job_title_id'),
            ).get('score'), axis=1
        )
        df['ollama_score'] = df.apply(
            lambda row: ollama_model.score_resume(
                row['prompt'],
                resume_id=row.get('name_id'),
                race_group=row.get('identity'),
                name_id=row.get('name_id'),
                job_title_id=row.get('job_title_id'),
            ).get('score'), axis=1
        )

    assert df['gemini_score'].notna().all()
    assert df['ollama_score'].notna().all()

    print("End-to-end synthetic pipeline test passed!")
