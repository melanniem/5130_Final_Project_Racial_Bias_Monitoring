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
