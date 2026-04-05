import os
import pandas as pd
import pytest
from bias_analysis.bias_quantification import BiasQuantification

def sample_data(tmp_path):
    data = pd.DataFrame({
        "score": [80, 85, 90, 70, 75, 60],
        "race_group": ["A", "A", "A", "B", "B", "B"],
        "rationale": [
            "strong leadership skills",
            "excellent communication",
            "high technical ability",
            "average performance",
            "needs improvement",
            "limited experience"
        ]
    })

    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def bq(tmp_path):
    sample_data(tmp_path)
    output_dir = tmp_path / "output"

    return BiasQuantification(
        data_path=tmp_path,
        input_file="test.csv",
        output_dir=output_dir,
        threshold=75
    )

def test_mean_score_difference(bq):
    result = bq.mean_score_difference()

    assert "mean" in result.columns
    assert len(result) == 2  # two groups

    mean_A = result.loc["A", "mean"]
    mean_B = result.loc["B", "mean"]

    assert mean_A > mean_B  # A should score higher

def test_welch_t_test(bq):
    result = bq.welch_t_test()

    assert not result.empty
    assert "p_value" in result.columns

    p_val = result["p_value"].iloc[0]
    assert 0 <= p_val <= 1

def test_cohens_d(bq):
    result = bq.cohens_d()

    assert "cohens_d" in result.columns

    d = result["cohens_d"].iloc[0]
    assert isinstance(d, float)

def test_disparity_ratio(bq):
    result = bq.disparity_ratio()

    assert "dir" in result.columns

    # Majority group should have DIR = 1
    majority_row = result.loc[result["selection_rate"].idxmax()]
    assert majority_row["dir"] == 1.0

def test_compute_pmi(bq):
    result = bq.compute_pmi(min_count=1)

    assert not result.empty
    assert "term" in result.columns

def test_embedding_analysis(bq):
    result = bq.embedding_analysis()

    #assert result is not None

import os

def test_output_files_created(bq):
    bq.run_bias_quantification_layer()

    expected_files = [
        "descriptive_stats.csv",
        "welch_tests.csv",
        "cohens_d.csv",
        "disparity_ratios.csv",
        "full_results.csv"
    ]

    for f in expected_files:
        assert os.path.exists(os.path.join(bq.output_dir, f))


def test_single_group(tmp_path):
    df = pd.DataFrame({
        "score": [80, 85],
        "race_group": ["A", "A"],
        "rationale": ["good", "great"]
    })

    file_path = tmp_path / "single.csv"
    df.to_csv(file_path, index=False)

    bq = BiasQuantification(tmp_path, "single.csv", tmp_path)

    result = bq.mean_score_difference()
    assert result.empty
