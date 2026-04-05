import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import os
from data_persistence.data_persistence import DataPersistence

# Make fake prompt output
def make_fake_prompts(tmp_path):
    """
    Create a fake prompts_output.csv
    Mirrors the columns that DataPersistence expects from Layer 2 output
    """
    df = pd.DataFrame([
        {
            "name_id":      0,
            "name":         "Wei Li",
            "first":        "Wei",
            "last":         "Li",
            "identity":     "Asian or Pacific Islander",
            "race_group":   "Asian or Pacific Islander",
            "mean_correct": 0.91,
            "job_title_id": 0,
            "job_title":    "Software Engineer",
            "prompt":       "Please evaluate this resume...",
        },
        {
            "name_id":      1,
            "name":         "Emily Clark",
            "first":        "Emily",
            "last":         "Clark",
            "identity":     "White",
            "race_group":   "White",
            "mean_correct": 0.88,
            "job_title_id": 0,
            "job_title":    "Software Engineer",
            "prompt":       "Please evaluate this resume...",
        },
    ])
    path = tmp_path / "prompts_output.csv"
    df.to_csv(path, index=False)
    return path

# make fake result from LLM
def make_fake_result(name_id = 0, job_title_id = 0, race_group="Asian or Pacific Islander"):
    """Return a fake LLM result dict"""
    return {
        "name_id":      name_id,
        "job_title_id": job_title_id,
        "race_group":   race_group,
        "model":        "qwen2.5:7b",
        "temperature":  0.0,
        "score":        85,
        "rationale":    "Strong technical background.",
        "raw_response": '{"score": 85, "rationale": "Strong technical background."}',
        "timestamp":    datetime.now(),
    }

# test crash recovery
def test_init_loads_output(tmp_path):
    """
    If llm_outputs.csv already exists, DataPersistence should load it
    instead of starting fresh. This is the crash recovery behaviour
    """
    make_fake_prompts(tmp_path)
    existing = pd.DataFrame([{
        "name_id": 0, "job_title_id": 0, "race_group": "Asian or Pacific Islander",
        "model": "qwen2.5:7b", "temperature": 0.0, "score": 72,
        "rationale": "Good fit.", "raw_response": "{}", "timestamp": datetime.now(),
    }])
    output_path = tmp_path / "llm_outputs.csv"
    existing.to_csv(output_path, index=False)
    dp = DataPersistence(DATA_PATH = tmp_path)
    assert len(dp.df) == 1
    assert dp.df.iloc[0]["score"] == 72

# test load prompts
def test_init_from_prompts(tmp_path):
    """
    If only prompts_output.csv exists, DataPersistence
    should load it and add the output columns as None
    """
    make_fake_prompts(tmp_path)
    dp = DataPersistence(DATA_PATH = tmp_path)
    assert "score" in dp.df.columns
    assert "rationale" in dp.df.columns
    assert "raw_response" in dp.df.columns
    assert "timestamp" in dp.df.columns
    assert dp.df["score"].isna().all()

# test no crash when no files
def test_init_no_files(tmp_path):
    """
    If neither file exists, DataPersistence should create an empty DataFrame
    with the expected columns.
    """
    dp = DataPersistence(DATA_PATH = tmp_path)
    assert dp.df.empty
    assert "name_id" in dp.df.columns
    assert "score" in dp.df.columns

# test match info
def test_append_updates_row(tmp_path):
    """
    When a result matches an existing row, it should update 
    that row's score and rationale instead of adding a new row
    """
    make_fake_prompts(tmp_path)
    dp = DataPersistence(DATA_PATH = tmp_path)
    original_len = len(dp.df)
    result = make_fake_result(name_id = 0, job_title_id = 0, race_group = "Asian or Pacific Islander")
    dp.append_result(result)
    assert len(dp.df) == original_len
    match = dp.df[
        (dp.df["name_id"].astype(str) == "0") &
        (dp.df["job_title_id"].astype(str) == "0")
    ]
    assert match.iloc[0]["score"] == 85

# test not match info
def test_append_new_rows(tmp_path):
    """
    When a result has information that doesn't exist
    in the DataFrame, it should insert a new row
    """
    make_fake_prompts(tmp_path)
    dp = DataPersistence(DATA_PATH = tmp_path)
    original_len = len(dp.df)
    result = make_fake_result(name_id = 999, job_title_id = 999, race_group = "White")
    dp.append_result(result)
    assert len(dp.df) == original_len + 1

# test reset
def test_reset_scores(tmp_path):
    """After reset_scores(), all columns should be None"""
    make_fake_prompts(tmp_path)
    dp = DataPersistence(DATA_PATH = tmp_path)
    dp.append_result(make_fake_result())
    dp.reset_scores()
    for col in ["model", "temperature", "score", "rationale", "raw_response", "timestamp"]:
        assert dp.df[col].isna().all(), f"{col} should be None after reset"

# check the all results on the csv file
def test_append_batch(tmp_path):
    """append_batch() should process all results and write the CSV to disk"""
    make_fake_prompts(tmp_path)
    dp = DataPersistence(DATA_PATH = tmp_path)
    results = [
        make_fake_result(name_id = 0, job_title_id = 0, race_group = "Asian or Pacific Islander"),
        make_fake_result(name_id = 1, job_title_id = 0, race_group = "White"),
    ]
    dp.append_batch(results)
    output_path = tmp_path / "llm_outputs.csv"
    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert len(saved) == 2

# check if the csv file is exist
def test_saves(tmp_path):
    """save() should write the current DataFrame to the output CSV path"""
    make_fake_prompts(tmp_path)
    dp = DataPersistence(DATA_PATH = tmp_path)
    dp.save()
    output_path = tmp_path / "llm_outputs.csv"
    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert len(saved) == len(dp.df)