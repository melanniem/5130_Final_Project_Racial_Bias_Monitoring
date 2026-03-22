from pathlib import Path
from input_layer.input import load_resumes
from input_layer import input

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

