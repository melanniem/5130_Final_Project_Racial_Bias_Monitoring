import json
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from input_layer.input import (
    JOB_DESCRIPTIONS,
    build_combinations,
    fix_dates,
    format_resume,
    load_resumes,
    sample_names,
)

"""
Unit tests for input.py

Covers functions: fix_dates, format_resume, sample_names, build_combinations, load_resumes
All tests use inline fixtures (no real data required)
"""

# Creates name dataframe and a minimal resume for test purposes

def make_names_df(n_per_group=3):
    """
    Minimal names DataFrame with 4 identity groups.
    """
    groups = ["Black", "Hispanic", "Asian", "White"]
    rows = []
    for i, group in enumerate(groups):
        for j in range(n_per_group):
            rows.append({
                "name": f"{group}Name{j}",
                "first": f"{group}First{j}",
                "last": f"Last{j}",
                "identity": group,
                "mean.correct": round(0.5 + j * 0.1, 2),
            })
    return pd.DataFrame(rows)


def make_minimal_resume():
    """
    Resume with one job and one education entry, no date conflicts.
    """
    return {
        "personal_info": {
            "email": "test@email.com",
            "phone": "555-0100",
            "linkedin": "linkedin.com/in/test",
            "github": "github.com/test",
        },
        "experience": [
            {
                "company": "Acme Corp",
                "title": "Engineer",
                "level": "Mid",
                "employment_type": "Full-time",
                "responsibilities": ["Built things"],
                "dates": {"start": "2020-01-01", "end": "2022-01-01", "duration": "2 yr"},
                "company_info": {"industry": "Tech", "size": "500"},
                "technical_environment": {
                    "technologies": ["Python"],
                    "methodologies": ["Agile"],
                    "tools": ["Git"],
                    "operating_systems": ["Linux"],
                    "databases": ["Postgres"],
                },
            }
        ],
        "education": [
            {
                "degree": {"level": "Bachelor", "field": "CS", "major": "Software"},
                "institution": {"name": "State U", "location": "CA", "accreditation": "ABET"},
                "dates": {"start": "2016-01-01", "expected_graduation": "2019-12-01"},
                "achievements": {"gpa": 3.8, "honors": "Cum Laude", "relevant_coursework": ["Algorithms"]},
            }
        ],
        "skills": {
            "technical": {
                "languages": [{"name": "Python", "level": "Expert"}]
            },
            "languages": [{"name": "English", "level": "Native"}],
        },
        "projects": [],
        "achievements": [],
    }

# === Test function "fix_dates" ===
# Sanity-tests function that fixes potnetial time overlaps of CV milestones

class TestFixDates:
    def test_no_overlap_unchanged(self):
        """
        Non-overlapping jobs should not have their dates changed.
        """
        resume = {
            "experience": [
                {"dates": {"start": "2018-01-01", "end": "2019-01-01"}, "title": "A"},
                {"dates": {"start": "2020-01-01", "end": "2021-01-01"}, "title": "B"},
            ],
            "education": [],
        }
        result = fix_dates(resume)
        assert result["experience"][0]["dates"]["start"] == "2018-01-01"
        assert result["experience"][1]["dates"]["start"] == "2020-01-01"

    def test_overlapping_jobs_resolved(self):
        """
        Overlapping experience dates should be pushed forward.
        """
        resume = {
            "experience": [
                {"dates": {"start": "2018-01-01", "end": "2020-06-01"}, "title": "A"},
                {"dates": {"start": "2019-01-01", "end": "2021-01-01"}, "title": "B"},
            ],
            "education": [],
        }
        result = fix_dates(resume)
        exp = result["experience"]
        end_prev = datetime.strptime(exp[0]["dates"]["end"], "%Y-%m-%d")
        start_next = datetime.strptime(exp[1]["dates"]["start"], "%Y-%m-%d")
        assert start_next >= end_prev, "Second job start should be >= first job end"

    def test_education_graduation_after_job_fixed(self):
        """
        Graduation date that falls after job start should be moved before it.
        """
        resume = {
            "experience": [
                {"dates": {"start": "2018-06-01", "end": "2020-01-01"}, "title": "Dev"},
            ],
            "education": [
                {"dates": {"start": "2015-01-01", "expected_graduation": "2019-05-01"}}
            ],
        }
        result = fix_dates(resume)
        grad_str = result["education"][0]["dates"]["expected_graduation"]
        grad = datetime.strptime(grad_str, "%Y-%m-%d")
        job_start = datetime.strptime("2018-06-01", "%Y-%m-%d")
        assert grad < job_start, "Graduation should be before first job start"

    def test_graduation_already_before_job_unchanged(self):
        """
        Graduation already before first job should not be modified.
        """
        resume = {
            "experience": [
                {"dates": {"start": "2020-01-01", "end": "2022-01-01"}, "title": "Dev"},
            ],
            "education": [
                {"dates": {"start": "2015-01-01", "expected_graduation": "2018-12-01"}}
            ],
        }
        result = fix_dates(resume)
        assert result["education"][0]["dates"]["expected_graduation"] == "2018-12-01"

    def test_empty_experience_and_education(self):
        """
        Resume with no experience or education should not raise.
        """
        resume = {"experience": [], "education": []}
        result = fix_dates(resume)
        assert result["experience"] == []
        assert result["education"] == []

    def test_does_not_mutate_input(self):
        """
        fix_dates should return a deep copy and not modify the original.
        """
        resume = {
            "experience": [
                {"dates": {"start": "2018-01-01", "end": "2020-06-01"}, "title": "A"},
                {"dates": {"start": "2019-01-01", "end": "2021-01-01"}, "title": "B"},
            ],
            "education": [],
        }
        original_start = resume["experience"][1]["dates"]["start"]
        fix_dates(resume)
        assert resume["experience"][1]["dates"]["start"] == original_start

    def test_unknown_dates_ignored(self):
        """
        Entries with 'Unknown' or missing dates should be skipped gracefully.
        """
        resume = {
            "experience": [
                {"dates": {"start": "Unknown", "end": "Unknown"}, "title": "A"},
            ],
            "education": [],
        }
        result = fix_dates(resume)
        # Entry is dropped because _start is None
        assert result["experience"] == []

    def test_temp_start_key_removed(self):
        """
        The internal _start key must not appear in the returned dict.
        """
        resume = {
            "experience": [
                {"dates": {"start": "2020-01-01", "end": "2021-01-01"}, "title": "A"},
            ],
            "education": [],
        }
        result = fix_dates(resume)
        for exp in result["experience"]:
            assert "_start" not in exp

# == Test function "format_resume" ==
# Verify resume builder: checks for sound personal information (email, GitHub, LinkedIn) 
# and no "Unknown" data appears in output

class TestFormatResume:
    def test_name_appears_at_top(self):
        resume = make_minimal_resume()
        output = format_resume(resume, "Jane Smith")
        assert output.startswith("Name: Jane Smith")

    def test_email_slug_uses_full_name(self):
        resume = make_minimal_resume()
        output = format_resume(resume, "Jane Smith")
        assert "jane.smith@email.com" in output

    def test_linkedin_slug_uses_full_name(self):
        resume = make_minimal_resume()
        output = format_resume(resume, "Jane Smith")
        assert "linkedin.com/in/janesmith" in output

    def test_github_slug_uses_full_name(self):
        resume = make_minimal_resume()
        output = format_resume(resume, "Jane Smith")
        assert "github.com/janesmith" in output

    def test_name_changes_between_calls(self):
        """
        Same resume with two different names should differ only by name-derived fields.
        """
        resume = make_minimal_resume()
        out_a = format_resume(resume, "Alice Johnson")
        out_b = format_resume(resume, "Bob Williams")
        assert "Alice Johnson" in out_a
        assert "Bob Williams" in out_b
        assert "Alice Johnson" not in out_b
        assert "Bob Williams" not in out_a

    def test_work_experience_section_present(self):
        resume = make_minimal_resume()
        output = format_resume(resume, "Test Person")
        assert "WORK EXPERIENCE" in output
        assert "Acme Corp" in output
        assert "Built things" in output

    def test_education_section_present(self):
        resume = make_minimal_resume()
        output = format_resume(resume, "Test Person")
        assert "EDUCATION" in output
        assert "State U" in output
        assert "3.8" in output

    def test_unknown_fields_excluded(self):
        """
        Fields set to 'Unknown' should not appear in the output.
        """
        resume = make_minimal_resume()
        resume["experience"][0]["level"] = "Unknown"
        output = format_resume(resume, "Test Person")
        assert "Level: Unknown" not in output

    def test_empty_resume_returns_name_line(self):
        """
        Minimal resume with only personal_info should still return something.
        """
        resume = {"personal_info": {"email": "x@x.com"}, "experience": [], "education": []}
        output = format_resume(resume, "Solo Person")
        assert "Name: Solo Person" in output

    def test_returns_string(self):
        output = format_resume(make_minimal_resume(), "A B")
        assert isinstance(output, str)
        assert len(output) > 0

# === Test function "sample_names" ===
# Confirms right number of names are returned per group,
# checks picking highest-rated names (not random ones)

class TestSampleNames:
    def test_correct_count_per_group(self):
        names_df = make_names_df(n_per_group=5)
        result = sample_names(names_df, names_per_group=3)
        counts = result["identity"].value_counts()
        for group in names_df["identity"].unique():
            assert counts[group] == 3

    def test_selects_highest_mean_correct(self):
        """
        Top-n by mean.correct should be selected.
        """
        names_df = make_names_df(n_per_group=5)
        result = sample_names(names_df, names_per_group=2)
        for group, group_df in result.groupby("identity"):
            min_selected = group_df["mean.correct"].min()
            full_group = names_df[names_df["identity"] == group]
            # The 2 selected should have the 2 highest mean.correct values
            top2 = full_group.nlargest(2, "mean.correct")["mean.correct"].min()
            assert min_selected >= top2

    def test_returns_dataframe(self):
        names_df = make_names_df()
        result = sample_names(names_df, names_per_group=2)
        assert isinstance(result, pd.DataFrame)

    def test_all_four_groups_present(self):
        names_df = make_names_df(n_per_group=5)
        result = sample_names(names_df, names_per_group=3)
        assert set(result["identity"].unique()) == {"Black", "Hispanic", "Asian", "White"}

# == Test function "build_combinations" ==
# Checks row count, even identity representation, 
# no missing/null columns, correct name inference and
# interleaving names (alternation of ethical groups)

class TestBuildCombinations:
    def test_row_count(self):
        """
        Should produce names × jobs rows.
        """
        names_df = make_names_df(n_per_group=2)
        resume = make_minimal_resume()
        jobs = {"Job A": "desc a", "Job B": "desc b"}
        result = build_combinations(resume, names_df, jobs)
        expected = len(names_df["identity"].unique()) * 2 * 2  # 4 groups × 2 names × 2 jobs
        assert len(result) == expected

    def test_all_identities_present(self):
        names_df = make_names_df(n_per_group=2)
        result = build_combinations(make_minimal_resume(), names_df, JOB_DESCRIPTIONS)
        assert set(result["identity"].unique()) == {"Black", "Hispanic", "Asian", "White"}

    def test_balanced_identity_counts(self):
        """
        Each identity group should have the same number of rows.
        """
        names_df = make_names_df(n_per_group=2)
        result = build_combinations(make_minimal_resume(), names_df, JOB_DESCRIPTIONS)
        counts = result["identity"].value_counts()
        assert counts.nunique() == 1, "All identity groups should have equal row counts"

    def test_required_columns_present(self):
        names_df = make_names_df(n_per_group=2)
        result = build_combinations(make_minimal_resume(), names_df, JOB_DESCRIPTIONS)
        for col in ["name_id", "job_title_id", "name", "first", "last",
                    "identity", "mean_correct", "job_title", "resume_text", "job_description"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_resume_text_contains_name(self):
        """
        Each row's resume_text should contain the corresponding name.
        """
        names_df = make_names_df(n_per_group=1)
        result = build_combinations(make_minimal_resume(), names_df, JOB_DESCRIPTIONS)
        for _, row in result.iterrows():
            assert row["name"] in row["resume_text"]

    def test_no_null_values(self):
        names_df = make_names_df(n_per_group=2)
        result = build_combinations(make_minimal_resume(), names_df, JOB_DESCRIPTIONS)
        assert result.notna().all().all(), "No column should contain NaN values"

    def test_interleaving_alternates_groups(self):
        """
        Rows should alternate identity groups rather than being clustered.
        """
        names_df = make_names_df(n_per_group=2)
        jobs = {"Job A": "desc"}
        result = build_combinations(make_minimal_resume(), names_df, jobs)
        identities = result["identity"].tolist()
        # With interleaving, no two consecutive rows should have the same identity
        consecutive_same = sum(1 for a, b in zip(identities, identities[1:]) if a == b)
        assert consecutive_same == 0, "Interleaved names should alternate identity groups"

# === Test function "load_resumes" ===
# Tests three formats (JSON array, single JSON object, JSONL with one object per line)
# load correctly and missing file throws error

class TestLoadResumes:
    def test_loads_json_array(self, tmp_path):
        resumes = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        p = tmp_path / "resumes.json"
        p.write_text(json.dumps(resumes))
        result = load_resumes(str(p))
        assert len(result) == 2
        assert result[0]["name"] == "Alice"

    def test_loads_single_json_object(self, tmp_path):
        resume = {"id": 1, "name": "Alice"}
        p = tmp_path / "resume.json"
        p.write_text(json.dumps(resume))
        result = load_resumes(str(p))
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    def test_loads_jsonl(self, tmp_path):
        lines = [{"id": 1}, {"id": 2}, {"id": 3}]
        p = tmp_path / "resumes.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in lines))
        result = load_resumes(str(p))
        assert len(result) == 3

    def test_returns_list(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"a": 1}]))
        result = load_resumes(str(p))
        assert isinstance(result, list)

    def test_raises_on_missing_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_resumes("/nonexistent/path/resumes.json")
