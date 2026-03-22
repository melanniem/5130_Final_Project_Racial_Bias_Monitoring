import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from itertools import combinations
from collections import Counter
from numpy.linalg import norm

"""
Bias Quantification
======================
Reads output from Data Persistence layer and runs:
  1. Mean score differences between race groups
  2. Welch's t-test -> Is the gap statistically real?
  3. Cohen's d -> Is the gap practically meaningful?
  4. Disparity ratio -> Does it violate EEOC's four-fifths rule?
  5. PMI proxy marker detection -> What words are driving group differences?
  6. Embedding-based word association -> Find semantic patterns in Rationale
  7. Visual graphs for all of the above
"""

class BiasQuantification:
    """
    Loads scored resume data and provides methods for each bias
    detection technique: statistical tests, disparity analysis,
    proxy marker detection and embedding-based association.
    """

    def __init__(self, data_path, input_file="llm_outputs.csv", threshold=75.0):

        filepath = os.path.join(data_path, input_file)
        self.df = pd.read_csv(filepath)
        self.df["score"] = pd.to_numeric(self.df["score"], errors="coerce") # Converts all possible entries to float, otherwise replaces with NaN
        self.df = self.df.dropna(subset=["score", "race_group"]) # Drops rows that either miss score or race

        self.scores = self.df["score"].values
        self.groups = self.df["race_group"].values
        self.unique_groups = sorted(np.unique(self.groups))
        self.threshold = threshold

        print(f"Loaded {len(self.df)} rows | Groups: {self.unique_groups}")

    # 1. Mean Score Difference

    def mean_score_difference(self, scores_a, scores_b):
        """
        Raw difference in mean scores between two groups A and B.
        """
        mu_a = np.mean(scores_a)
        mu_b = np.mean(scores_b)
        return {"mu_a": mu_a, "mu_b": mu_b, "delta_mu": mu_a - mu_b}
    
    # 2. Welch's t-test
 
    def welch_t_test(self, scores_a, scores_b):
        """
        Calculates whether the score gap is statistically real.
        """
        t_stat, p_val = stats.ttest_ind(scores_a, scores_b, equal_var=False)
        return {"t_statistic": t_stat, "p_value": p_val}

    # 3. Cohen's d

    def cohens_d(self, scores_a, scores_b):
        """
        Calculates effect size to see if gap practically meaningful
        
        d = (mu_A - mu_B) / pooled_std
        """
        n_a, n_b = len(scores_a), len(scores_b)
        # Pooled std. combines the variance from two groups into one (weighted by their sample sizes)
        pooled_std = np.sqrt(
            ((n_a - 1) * scores_a.var(ddof=1) + (n_b - 1) * scores_b.var(ddof=1)) # ddof= 1 computes sample variance
            / (n_a + n_b - 2)
        )
        d = (scores_a.mean() - scores_b.mean()) / pooled_std

        # Gap interpretation scale by Jacob Cohen
        abs_d = abs(d)
        if abs_d < 0.2:
            size = "negligible"
        elif abs_d < 0.5:
            size = "small"
        elif abs_d < 0.8:
            size = "medium"
        else:
            size = "large"

        return {"cohens_d": d, "interpretation": size}

    # 4. Disparity Ratio

    def disparity_ratio(self):
        """
        Disparity ratio using EEOC four-fifths rule
        """
        selection_rates = {}
        for g, group_df in self.df.groupby("race_group"): # Looks at scores per racial group
            selection_rates[g] = (group_df["score"] >= self.threshold).mean() # Calculates selection rate per group using threshold as cutoff

        majority = max(selection_rates, key=selection_rates.get) 
        # Finds which group has the highest selection rate and uses it as comparison group for four-fifth rule

        results = {"majority": majority, "majority_sr": selection_rates[majority], "groups": {}}
        for g, sr in selection_rates.items():
            if g == majority:
                continue # Skip majority group
            # Create output for results per group
            dir_val = sr / selection_rates[majority]
            results["groups"][g] = {
                "selection_rate": sr,
                "dir": dir_val, # Disparity ratio
                "adverse_impact": dir_val < 0.8, # Four-fifth ration flagging adverse impact on racial group
            }

        return results

    # 5. PMI (Pointwise Mutual Information) Proxy Markers

    def compute_pmi(self, min_count=5): # Ignores terms that appear less than five times
        """
        Pointwise Mutual Information Computation: 
        Which Rationale words are disproportionately linked to which groups?
        """
        rationales = self.df["rationale"].fillna("").tolist() # Converts Rationales to list, fills NA with empty string (so code does not crash)
        group_list = self.df["race_group"].tolist() # Converts racial group name to list
        N = len(rationales)

        group_counts = Counter(group_list) # Counts how many Rationales belong to each race group
        term_group_counts = Counter() # Will track how many times a specific word appears with a specific group
        term_counts = Counter() # Will track how many times a word appears across all groups

        for doc, group in zip(rationales, group_list): # Loops through every rationale and corresponding group at the same time
            terms = set(str(doc).lower().split()) # Standardizes terms
            for term in terms: # Counts word appearance per group and overall
                term_group_counts[(term, group)] += 1
                term_counts[term] += 1

        rows = []
        # Loops through every term-group pair and skips if less than 5 appearances
        for (term, group), count in term_group_counts.items():
            if count < min_count:
                continue
            # Three probabilities for PMI formula and PMI formula:
            p_tg = count / N # Probability that this term and this group appear together
            p_t = term_counts[term] / N # Probability that this term appears at all
            p_g = group_counts[group] / N # Probability of this group
            pmi = np.log2(p_tg / (p_t * p_g)) # PMI formula
            rows.append({"term": term, "group": group, "pmi": round(pmi, 3), "count": count})

        return pd.DataFrame(rows).sort_values("pmi", ascending=False).reset_index(drop=True)

    # 6. Embedding Analysis

    def compute_embeddings(self):
        """
        Builds TF-IDF embeddings of rationale text and group centroids.
        """
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=3) # 
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["rationale"].fillna(""))
        self.feature_names = self.vectorizer.get_feature_names_out()

        self.centroids = {}
        for g in self.unique_groups:
            mask = self.groups == g
            self.centroids[g] = self.tfidf_matrix[mask].toarray().mean(axis=0)

    def embedding_direction(self, group_a, group_b, top_k=5):
        """
        Finds terms most associated with each group via embedding direction vector.
        """
        direction = self.centroids[group_a] - self.centroids[group_b]
        term_scores = sorted(zip(self.feature_names, direction), key=lambda x: x[1], reverse=True)

        toward_a = [(t, round(s, 3)) for t, s in term_scores[:top_k] if s > 0]
        toward_b = [(t, round(abs(s), 3)) for t, s in term_scores[-top_k:] if s < 0]
        return toward_a, toward_b

    def centroid_similarity(self):
        """
        Finds cosine similarity between all group rationale centroids.
        """
        n = len(self.unique_groups)
        matrix = pd.DataFrame(np.zeros((n, n)), index=self.unique_groups, columns=self.unique_groups)
        for g_a in self.unique_groups:
            for g_b in self.unique_groups:
                a, b = self.centroids[g_a], self.centroids[g_b]
                matrix.loc[g_a, g_b] = round(np.dot(a, b) / (norm(a) * norm(b)), 4)
        return matrix
    

# 7. Visuals (Graphs)






