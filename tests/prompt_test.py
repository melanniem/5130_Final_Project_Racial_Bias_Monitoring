import pandas as pd
import pytest
from prompt_layer.prompt_standardization import (
    build_prompt,
    build_null_baseline_prompts,
    verify_prompt,
)

# Fake dataframe  
def make_fake_df():
    """Return a minimal DataFrame that mirrors input_combinations.csv"""
    return pd.DataFrame([
        {
            "name_id":      0,
            "name":         "Wei Li",
            "first":        "Wei",
            "last":         "Li",
            "identity":     "Asian or Pacific Islander",
            "mean_correct": 0.91,
            "job_title_id": 0,
            "job_title":    "Software Engineer",
            "resume_text":  "Name: Wei Li\nEmail: wei.li@email.com\nSkills: Python, Java",
            "job_description": "We are looking for a Software Engineer.",
        },
        {
            "name_id":      1,
            "name":         "Emily Clark",
            "first":        "Emily",
            "last":         "Clark",
            "identity":     "White",
            "mean_correct": 0.88,
            "job_title_id": 0,
            "job_title":    "Software Engineer",
            "resume_text":  "Name: Emily Clark\nEmail: emily.clark@email.com\nSkills: Python, Java",
            "job_description": "We are looking for a Software Engineer.",
        },
        {
            "name_id":      2,
            "name":         "James Smith",
            "first":        "James",
            "last":         "Smith",
            "identity":     "White",
            "mean_correct": 0.85,
            "job_title_id": 1,
            "job_title":    "Data Scientist",
            "resume_text":  "Name: James Smith\nEmail: james.smith@email.com\nSkills: Python, R",
            "job_description": "We are looking for a Data Scientist.",
        },
    ])

# test prompt
def test_prompt_contains_job_title():
    """Job title should appear in the generated prompt"""
    prompt = build_prompt("Name: Wei Li\nSkills: Python", "Software Engineer", "Looking for a Software Engineer.")
    assert "Software Engineer" in prompt

def test_prompt_contains_resume_text():
    """Resume text should appear in the generated prompt"""
    resume = "Name: Wei Li\nSkills: Python"
    prompt = build_prompt(resume, "Software Engineer", "Looking for a Software Engineer.")
    assert resume in prompt

def test_prompt_contains_json_instruction():
    """Prompt must instruct the model to return a JSON with score and rationale"""
    prompt = build_prompt("Name: Wei Li", "Software Engineer", "Looking for a Software Engineer.")
    assert '"score"' in prompt
    assert '"rationale"' in prompt

# test null baseline
def test_baseline_replaces_name():
    """Every resume text in the baseline should have the name be Applicant, not the original name"""
    df = make_fake_df()
    baseline = build_null_baseline_prompts(df, n = 2)
    for _, row in baseline.iterrows():
        assert "Applicant" in row["resume_text"]
        assert row["name"] not in row["resume_text"] or row["name"] == "Applicant"

def test_null_baseline_identity_tag():
    """identity column must be Null Baseline for every baseline row"""
    df = make_fake_df()
    baseline = build_null_baseline_prompts(df, n = 2)
    assert (baseline["identity"] == "Null Baseline").all()

def test_null_baseline_name_id_format():
    """name_id values should follow the pattern null_0, null_1, ..."""
    df = make_fake_df()
    baseline = build_null_baseline_prompts(df, n = 2)
    expected = [f"null_{i}" for i in range(2)]
    assert list(baseline["name_id"]) == expected

def test_null_baseline_row_count():
    """Number of baseline rows should equal n"""
    df = make_fake_df()
    baseline = build_null_baseline_prompts(df, n = 2)
    assert len(baseline) == 2

# test when prompt not matches template
def test_job_description_fails():
    """Consistency check should fail when prompts differ beyond the name"""
    df = make_fake_df()
    df["prompt"] = df.apply(
        lambda row: build_prompt(row["resume_text"], row["job_title"], row["job_description"]),
        axis=1,)
    df.loc[0, "prompt"] = df.loc[0, "prompt"] + " EXTRA TEXT"
    result = verify_prompt(df)
    assert result is False