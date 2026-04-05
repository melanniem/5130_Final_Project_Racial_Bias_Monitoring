# Final Project

## Overview
This project investigates whether Large Language Models (LLMs) exhibit racial bias in automated resume screening by replacing the candidate's name as a racial proxy across otherwise identical resumes, and analyzing score disparities across racial groups.

The pipeline is designed as a **five-layer modular system**:
- Input Layer
- Prompt Standardization Layer
- Model Interface layer
- Data Persistence Layer
- Bias Quantification Layer

---

## Repository Structure
## Project Structure

```
Racial_Bias_Monitering/
├── EDA.ipynb
├── README.md
├── __init__.py
├── bias_analysis/
│   └── bias_quantification.py
├── data/
│   ├── master_resumes.jsonl
│   ├── racial_markers.csv
│   └── resume1.json
├── data_persistence/
│   └── data_persistence.py
├── gemini_main.py
├── input_combinations.csv
├── input_layer/
│   └── input.py
├── logs/
│   ├── pipeline.log
│   └── running.log
├── main.py
├── model_interface/
│   ├── gemini_interface.py
│   └── ollama_interface.py
├── power_analysis.py
├── prompt_layer/
│   └── prompt_standardization.py
├── prompts_output.csv
├── requirements.txt
├── results/
│   ├── cohens_d.csv
│   ├── cohens_d.png
│   ├── descriptive_stats.csv
│   ├── disparity_ratios.csv
│   ├── disparity_ratios.png
│   ├── embedding_pca.png
│   ├── embedding_similarity.png
│   ├── full_results.csv
│   ├── llm_outputs.csv
│   ├── pmi_proxy_markers.csv
│   ├── pmi_proxy_markers.png
│   ├── prompts_output.csv
│   ├── score_distributions.png
│   ├── welch_pvalues.png
│   └── welch_tests.csv
└── tests/
    ├── __init__.py
    ├── bias_quantification_test.py
    ├── input_test.py
    ├── model_interface_test.py
    ├── pipeline_test.py
    └── prompt_test.py
```

---

## Prerequisites

Before setting up the project, install the following:

**Anaconda or Miniconda** — [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/)

**Git**
- macOS: `xcode-select --install`
- Windows/Linux: [git-scm.com](https://git-scm.com)

**Ollama** *(local LLM pipeline only)* — [ollama.com](https://ollama.com)

After installing Ollama, pull the required model:
```
ollama pull qwen2.5:7
```

## Installation
**1. Clone the repository**
```
git clone <your-repo-url>
cd <repo-folder>
```

**2. Create the environment**
```
conda env create -f environment.yml
conda activate bias-audit
```
This installs Python 3.12 and all required dependencies automatically.

**3. Configure your API key**

```
cp .env.example .env
```

Open `.env` and replace the placeholder:
```
GEMINI_API_KEY=your_actual_key_here
```
**Note:** Only required for ```gemini_main.py```. The Ollama pipeline does not need an API key.

**4. Add data files**
Use `data/resume1.json` or add your own resume data.

**5. Verify the setup**
Install test dependencies and run the test suite:
```
pip install pytest pytest-anyio
pytest tests/ -v
```
All unit tests should pass. Integration tests skip automatically if data files are missing.

## Running the Pipeline
Gemini (cloud):
```
python gemini_main.py
```
Ollama (local):
```
python main.py
```
Both pipelines prompt you at startup:
```
Run LLM scoring? (y/n):
```
`y` - full pipeline: input generation → prompt building → LLM scoring → bias analysis

`n` - skip to bias analysis only (requires results/llm_outputs.csv from a prior run)

---

## Results
All outputs are saved to results/ after a full run:

| File | Contents |
|------|----------|
| `llm_outputs.csv` | Raw LLM scores per resume / name / job |
| `full_results.csv` | Combined results across all runs |
| `descriptive_stats.csv` | Mean scores by racial group |
| `welch_tests.csv` | Statistical significance tests |
| `cohens_d.csv` | Effect sizes |
| `disparity_ratios.csv` | EEOC four-fifths rule analysis |
| `*.png` | Visualizations (boxplots, heatmaps, PCA) |

Logs are written to `logs/pipeline.log`.

---

## Troubleshooting
| Problem | Fix |
|---------|-----|
| `conda: command not found` | Restart terminal after installing Anaconda |
| `GEMINI_API_KEY not set` | Check `.env` exists and has the correct key |
| `ollama: connection refused` | Open the Ollama app or run `ollama serve` |
| `data/racial_markers.csv not found` | Place data files in the `data/` folder |
| Tests fail on import | Ensure `conda activate bias-audit` was run first |

---

## Scalability Estimate
The baseline experimental design generates:
1 resume × 50 names × 5 groups (4 racial + 1 null baseline) × 3 jobs = 750 prompts

---

## Technical Stack

| Component | Technology                                            |
|---|-------------------------------------------------------|
| Language | Python 3.10+                                          |
| Data manipulation | pandas                                                |
| LLM API | Ollama (`qwen2.5:7b`) & Google Gemini ('gemini-2.5') |
| Resume dataset | resume1.jsonl (single resume)                         |
| Name dataset | Crabtree et al. (2023) validated racial name dataset  |
| Statistical analysis | scipy, numpy                                          |
| Embeddings | sklearn (TF-IDF, PCA)                                 |
| Output format | CSV                                                   |

---

## Layer Documentation

### Layer 1 — Input Layer

**Purpose:** Loads resume and name data, injects racial-marker names into resume, and generates resume x names x jobs combinations for evaluation.

**Tech Stack:**
| Component | Technology |
|---|---|
| Data processing | `pandas` |
| Resume parsing | `json` |
| Date handling | `python-dateutil`, `datetime` |
| Sampling | `random` |

**Input:**
- `data/racial_markers.csv` — columns: `name`, `first`, `last`, `identity`, `mean.correct`
- `data/resume1.jsonl` — one JSON resume

**Output:** 
| Column | Description |
|---|---|
| `resume_id` | Index of the sampled resume |
| `name_id` | Unique ID for each name (used for downstream mapping) |
| `job_title_id` | Unique ID for each job title |
| `name` | Full injected candidate name |
| `first` | First name |
| `last` | Last name |
| `identity` | Racial group label |
| `mean_correct` | Name recognition rate from validation study |
| `job_title` | Job title being evaluated |
| `resume_text` | Formatted resume text with injected name |
| `job_description` | Full job description text |

**Error Handling:**
Missing or malformed fields are silently skipped, and date parsing failures are handled gracefully to ensure the pipeline continues processing remaining records.

---

### Layer 2 — Prompt Standardization

**Purpose:** Converts resume x names x jobs combination into a standardized evaluation prompt, and verifies that prompts for the same resume-job pair differ only by candidate name.

**Prompt format:**

Each prompt instructs the LLM to act as an expert HR recruiter and evaluate the resume for a specific job. The LLM is told to respond only with a JSON object:

```json
{
  "score": <integer 0–100>,
  "rationale": "<one to two sentences, max 100 words>"
}
```

**Tech Stack:**
| Component | Technology |
|---|---|
| Data processing | `pandas` |
| Sampling | `random` |

**Input:** csv (output of Layer 1)

**Output:** `prompts_output.csv` with columns:

| Column | Description |
|---|---|
| `resume_id` | Resume index (for downstream mapping) |
| `name_id` | Name index (for downstream mapping) |
| `name` | Full name |
| `first` | First name |
| `last` | Last name |
| `identity` | Racial group label |
| `mean_correct` | Name recognition rate |
| `job_title_id` | Job title index |
| `job_title` | Job title string |
| `prompt` | Full formatted prompt string ready for LLM input |

**Error Handling:**
Consistency check failures are flagged with the specific resume_id and job_title for review, without stopping the pipeline.

---

### Layer 3 — Model Interface

**Purpose:**
Sends each standardized prompt to the local Ollama model and returns a structured response containing a numeric score and rationale.

**Tech Stack:**

| Component        | Technology                                    |
|------------------|-----------------------------------------------|
| Local LLM        | Ollama (`qwen2.5:7b`)                         |
| Cloud LLM        | Gemini(`gemini-2.5` via google-generativeai ) |
| Response parsing | `json`                                        |
| Retry delay      | `time`, `datetime`                            |

**Input:** Prompt strings from `prompts_output.csv`

**Output:** Score (0–100), rationale text, and metadata per prompt

**Error Handling:**
API call failures are retried automatically, unresolvable errors return a null score and are logged without crashing the pipeline.

---

### Layer 4 — Data Persistence

**Purpose:**
Stores all model outputs alongside their metadata into a structured CSV file, supporting incremental saving and crash recovery.

**Tech Stack:**

| Component | Technology |
|---|---|
| Data processing | `pandas` |
| Logging | `logging` |
| File path management | `pathlib`, `os` |
| Timestamp | `datetime` |

**Input:** Scored results from Layer 3

**Output:** `llm_outputs.csv`

| Column | Description |
|---|---|
| `resume_id` | Resume index |
| `name_id` | Name index |
| `job_title_id` | Job title index |
| `race_group` | Racial group label |
| `model` | Model name/version |
| `temperature` | Sampling temperature |
| `score` | LLM-assigned score (0–100) |
| `rationale` | LLM-generated rationale text |
| `raw_response` | Raw model output (for debugging) |
| `timestamp` | Time of model call |

**Error Handling:**
Unmatched or failed record updates are logged as warnings without interrupting the pipeline. Progress is saved incrementally to support crash recovery.

---

### Layer 5 — Bias Quantification

**Purpose:**
Analyzes collected scores and rationales using statistical tests and text-based methods to detect and quantify racial bias. Includes a power analysis to determine the minimum sample size required to detect a statistically meaningful effect.

**Tech Stack:**

| Component | Technology |
|---|---|
| Statistical tests | `scipy.stats` |
| Power analysis | `statsmodels` |
| Embeddings & PCA | `sklearn` |
| Visualization | `matplotlib`, `seaborn` |
| Data processing | `numpy`, `pandas` |
| Word frequency | `collections.Counter` |

**Input:** `llm_outputs.csv` from Layer 4

**Output:** Statistical results (CSV) and visualizations (PNG) saved to output directory, consolidated into `full_results.csv`

| Method | Output file |
|---|---|
| Mean score differences | `descriptive_stats.csv`, `score_distributions.png` |
| Welch's t-test | `welch_tests.csv`, `welch_pvalues.png` |
| Cohen's d | `cohens_d.csv` |
| Disparity ratio | `disparity_ratios.csv` |
| PMI proxy markers | `pmi_proxy_markers.csv`, `pmi_proxy_markers.png` |
| Embedding analysis | `embedding_similarity.png`, `embedding_pca.png` |
| Combined results | `full_results.csv` |

**Error Handling:**
Each analysis method checks for sufficient data before running and skips if the minimum requirements are not met.



---

## Testing

Tests are organized to match the five-layer pipeline structure and cover both individual components (unit tests) and full end-to-end data flow (integration tests).

### Unit Tests

Each layer has a corresponding unit test module in `tests/`:

| Module                        | What it tests                                                                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| `input_test.py`               | Name injection, resume formatting, missing field handling, date parsing edge cases              |
| `prompt_test.py`              | Prompt template rendering, consistency checks (same resume-job pair differs only by name)       |
| `model_interface_test.py`     | Mock API calls for both Ollama and Gemini, JSON parsing, retry logic on failure                 |
| `data_persistence_test.py`    | CSV write/append correctness, crash recovery, null score logging                                |
| `bias_quantification_test.py` | Statistical test outputs (Welch's t, Cohen's d, disparity ratio), edge cases with small samples |
| `pipeline_test.py`            | validate the full pipeline using a small synthetic fixture dataset (5 names × 1 job × 2 models = 10 prompts):                                                                                                |

Run unit tests:
```bash
pytest tests/unit/
```

Run integration tests:
```bash
pytest tests/integration/
```


---


# Racial Markers Dataset

**Prepared by:** Yutong Zhang  
**Date:** February 21, 2026  

## Source

**Citation:**
Crabtree, C., Kim, J.Y., Gaddis, S.M., Holbein, J.B., Guage, C., & Marx, W.W. (2023).  
*Validated names for experimental studies on race and ethnicity.*  

**Scientific Data**, 10(1), 130.  
https://doi.org/10.1038/s41597-023-01947-0

**Data Repository:**
- [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LP4EAR)
- [GitHub](https://github.com/jaeyk/validated_names)

## Original Dataset

**Total:** 600 validated names (tested with 4,026 survey respondents)

**Distribution by Race:**

| Race | Count | Note |
|------|-------|------|
| White | 100 | Included |
| Black or African American | 100 | Included |
| Hispanic | 100 | Included |
| Asian or Pacific Islander | 100 | Pure Asian only |
| White Asian | 200 | Excluded |

> **Note:** "White Asian" names (English first names with Asian last names) were excluded from our analysis to maintain clear racial distinctions.

## Selection Criteria

### Threshold
- Recognition rate: > 65%
- Names correctly identified by more than 65% of survey respondents

### Process
1. Filter names by recognition rate threshold
2. Sort names by recognition rate (highest to lowest) within each racial category
3. Exclude "White Asian" category

### Rationale
- Ensures reliable racial signaling
- Balances quality with sufficient sample size across all groups
- Maintains clear racial distinctions for experimental validity

## Final Selected Dataset

### Summary Statistics

**Total:** 341 validated names  

### Distribution by Race

| Race | Count | Avg Recognition | Recognition Range |
|------|-------|-----------------|-------------------|
| White | 98 | 82.0% | 67.2% - 93.1% |
| Black or African American | 57 | 74.1% | 65.5% - 89.1% |
| Hispanic | 98 | 78.3% | 65.6% - 87.5% |
| Asian or Pacific Islander | 88 | 77.9% | 65.3% - 91.0% |

## File Information

**Filename:** `racial_markers.csv`  
**Dimensions:** 341 rows × 6 columns

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `first` | string | First name |
| `last` | string | Last name |
| `name` | string | Full name (first + last) |
| `identity` | string | Racial category |
| `w.asian` | binary | White Asian indicator (0 for all selected names) |
| `mean.correct` | float | Recognition rate (0-1 scale) |
