# Final Project

## Overview
This project investigates whether Large Language Models (LLMs) exhibit racial bias in automated resume screening by replacing the candidate's name as racail proxies across otherwise identical resumes, and analyzing score disparities across racial groups.

The pipeline is designed as a **five-layer modular system**:
- Input Layer
- Prompt Standardization Layer
- Model Interface layer
- Data Persistence Layer
- Bias Quantification Layer

---

## Technical Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas |
| LLM API | Ollama (`qwen2.5:7b`) & Ollama (...) |
| Resume dataset | HuggingFace (`datasetmaster/resumes`, 4,817 resumes) |
| Name dataset | Crabtree et al. (2023) validated racial name dataset |
| Statistical analysis | scipy, numpy |
| Embeddings | sentence-transformers |
| Output format | CSV |

---

## Layer Documentation

### Layer 1 — Input Layer

**Purpose:** Loads resume and name data, injects racial-marker names into resumes, and generates all resume x name x job combinations for evaluation.

**Tech Stack:**

| Component | Technology |
|---|---|
| Data processing | `pandas` |
| Resume parsing | `json` |
| Date handling | `python-dateutil`, `datetime` |
| Sampling | `random` |

**Input:**
- `data/racial_markers.csv` — columns: `name`, `first`, `last`, `identity`, `mean.correct`
- `data/master_resumes.jsonl` — one JSON resume object per line

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
Missing or malformed fields are silently skipped and date parsing failures are handled gracefully to ensure the pipeline continues processing remaining records.

---

### Layer 2 — Prompt Standardization

**Purpose:** Converts each resume x name x job combination into a standardized evaluation prompt, and verifies that prompts for the same resume-job pair differ only by candidate name.

**Prompt format:**

Each prompt instructs the LLM to act as an expert HR recruiter and evaluate the resume for a specific job. The LLM is told to respond only with a JSON object:

```json
{
  "score": <integer 0–100>,
  "rationale": "<one to two sentences, max 100 words>"
}
```

This structured format ensures consistent, machine-parseable output across all API calls.

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
| Component | Technology |
|---|---|
| Local LLM | Ollama (`qwen2.5:7b`) |
| Response parsing | `json` |
| Retry delay | `time`, `datetime` |

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

## Scalability Estimate

Baseline:
```
50 resumes × without name × 3 job titles = 150 prompts
```
Test:

```
50 resumes × 228 names (57 per group × 4 groups) × 3 job titles = 34,200 prompts
```

Full dataset:

```
4,817 resumes × 228 names × 3 job titles = 3,295,632 prompts
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
