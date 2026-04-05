import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from input_layer import input
from prompt_layer import prompt_standardization as prompt
from model_interface.gemini_interface import Gemini
from model_interface.ollama_interface import OllamaQwen
from bias_analysis.bias_quantification import BiasQuantification
from data_persistence import data_persistence

MOCK_RESPONSE = '{"score": 75, "rationale": "Strong candidate with relevant experience."}'


def test_full_pipeline_synthetic():
    """
    Validate the full pipeline using a small synthetic fixture dataset:
    5 names × 1 job × 2 models = 10 prompts
    """
    # Base path for saving outputs
    base = Path(__file__).resolve().parent.parent

    # Synthetic fixture dataset
    synthetic_names = [
        {"name": "Alice", "identity": "white"},
        {"name": "Bob", "identity": "white"},
        {"name": "Carlos", "identity": "hispanic"},
        {"name": "Diana", "identity": "black"},
        {"name": "Eve", "identity": "asian"},
    ]

    synthetic_resumes = [
        {"resume_id": 1, "content": "Experienced software engineer."}
    ]

    synthetic_jobs = ["Software Engineer"]

    # Sample 5 names
    names_sampled = input.sample_names(synthetic_names, 5)

    # Build input combinations: 5 names × 1 job × 2 models = 10 prompts
    df = input.build_combinations(
        resumes=synthetic_resumes,
        names=names_sampled,
        job_descriptions=synthetic_jobs,
        num_models=2
    )

    # Basic checks
    assert not df.empty, "Combinations dataframe is empty"
    assert len(df) == 5 * 1 * 2, "Expected 10 rows for synthetic dataset"

    df.to_csv(base / "input_combinations_synthetic.csv", index=False)

    # Run prompt layer
    df_prompts = prompt.run_prompt_layer(df)
    assert not df_prompts.empty, "Prompt layer returned empty dataframe"
    assert 'prompt' in df_prompts.columns, "Prompt column missing"

    # Check all expected columns
    expected_cols = ['resume_id', 'name_id', 'job_title_id', 'name', 'identity', 'job_title', 'prompt']
    for col in expected_cols:
        assert col in df_prompts.columns, f"{col} missing from prompt output"
        assert df_prompts[col].notna().all(), f"{col} contains null values"

    # Check prompt content
    for p in df_prompts['prompt']:
        assert "score" in p
        assert "rationale" in p
        assert "RESUME" in p
        assert "JOB DESCRIPTION" in p

    # Verify prompt layer passes custom verification
    assert prompt.verify_prompt(df_prompts) is True

    # Run Model Interface Layer — mock call_model so no real API calls are made
    with patch.object(Gemini, 'call_model', return_value=MOCK_RESPONSE), \
         patch.object(OllamaQwen, 'call_model', return_value=MOCK_RESPONSE):

        gemini = Gemini(api_key="mock_key")
        ollama_model = OllamaQwen()

        df_prompts['gemini_score'] = df_prompts.apply(
            lambda row: gemini.score_resume(
                row['prompt'],
                resume_id=row.get('resume_id'),
                race_group=row.get('identity'),
                name_id=row.get('name_id'),
                job_title_id=row.get('job_title_id'),
            ).get('score'), axis=1
        )
        df_prompts['ollama_score'] = df_prompts.apply(
            lambda row: ollama_model.score_resume(
                row['prompt'],
                resume_id=row.get('resume_id'),
                race_group=row.get('identity'),
                name_id=row.get('name_id'),
                job_title_id=row.get('job_title_id'),
            ).get('score'), axis=1
        )

    # Validate scores exist
    assert df_prompts['gemini_score'].notna().all()
    assert df_prompts['ollama_score'].notna().all()

    # Build combined output in BiasQuantification format and run bias analysis
    combined = pd.concat([
        df_prompts[['resume_id', 'name_id', 'job_title_id', 'identity']].assign(
            score=df_prompts['gemini_score'], model='gemini'
        ),
        df_prompts[['resume_id', 'name_id', 'job_title_id', 'identity']].assign(
            score=df_prompts['ollama_score'], model='ollama'
        ),
    ]).rename(columns={'identity': 'race_group'})

    with tempfile.TemporaryDirectory() as tmpdir:
        combined.to_csv(Path(tmpdir) / "llm_outputs.csv", index=False)
        bq = BiasQuantification(
            data_path=tmpdir,
            input_file="llm_outputs.csv",
            output_dir=str(Path(tmpdir) / "results"),
        )
        bias_results = bq.mean_score_difference()
        assert bias_results is not None
        assert not bias_results.empty

    results_path = base / "results_synthetic.csv"
    combined.to_csv(results_path, index=False)
    assert results_path.exists()

    print("End-to-end synthetic pipeline test passed!")
