from pathlib import Path
from input_layer.input import load_resumes
from input_layer import input
from prompt_layer import prompt_standardization as prompt
import main

def test_full_pipeline():
    base = Path(__file__).resolve().parent.parent

    names_path = base / "data" / "racial_markers.csv"
    resumes_path = base / "data" / "master_resumes.jsonl"

    names = input.load_names(names_path)
    resumes = input.load_resumes(resumes_path)

    names_sampled = input.sample_names(names, 2)
    resumes_sampled = input.sample_resumes(resumes, 3, seed=42)


    df = input.build_combinations(resumes_sampled, names_sampled, input.JOB_DESCRIPTIONS)

    assert not df.empty
    assert len(df) == 3 * len(names_sampled) * len(input.JOB_DESCRIPTIONS)
    
    df.to_csv(base / "input_combinations.csv", index=False)

    df = prompt.run_prompt_layer()
    assert not df.empty
    assert 'prompt' in df.columns
    for col in ['resume_id', 'name_id', 'job_title_id', 'name', 'identity', 'job_title', 'prompt']:
        assert col in df.columns
        assert df[col].notna().all(), f"{col} contains null values"
    assert all("score" in p for p in df['prompt'])
    assert all("rationale" in p for p in df['prompt'])
    assert all("RESUME" in p for p in df['prompt'])
    assert all("JOB DESCRIPTION" in p for p in df['prompt'])
    assert prompt.verify_prompt(df) == True

def test_data_distribution():
  df = main.run_pipeline()

  counts = df["identity"].value_counts()
  assert counts.min() > 0
  assert counts == 4