from pathlib import Path
from input_layer import input
from prompt_layer import prompt_standardization as prompt
from google import genai
from ollama import chat
from bias_analysis import bias_quantification
from data_persistence import data_persistence
import main


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

    # Run Model Interface Layer
    df_prompts['gemini_score'] = df_prompts['prompt'].apply(lambda p: genai.mock_score(p))
    # Generate scores for Ollama
    df_prompts['ollama_score'] = df_prompts['prompt'].apply(lambda p: ollama_interface.mock_score(p))

    # Validate scores exist
    assert df_prompts['gemini_score'].notna().all()
    assert df_prompts['ollama_score'].notna().all()

    # Run bias analysis
    bias_results = bias_quantification.run_bias_analysis(df_prompts)
    assert not bias_results.empty
    assert all(col in bias_results.columns for col in ['identity', 'gemini_score', 'ollama_score', 'disparity'])

    results_path = base / "results_synthetic.csv"
    bias_results.to_csv(results_path, index=False)

    assert results_path.exists()

    print("End-to-end synthetic pipeline test passed!")
