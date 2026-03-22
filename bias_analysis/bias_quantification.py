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
"""

class BiasQuantification:
    """
    Loads scored resume data and provides methods for each bias
    detection technique: statistical tests, disparity analysis,
    proxy marker detection and embedding-based association. 
    Each method prints results, saves a graph, and returns the data.
    """

    def __init__(self, data_path, input_file="llm_outputs.csv", output_dir="./evaluation_outputs", threshold=75.0):
        self.output_dir = output_dir
        self.threshold = threshold
        os.makedirs(self.output_dir, exist_ok=True) # Create output directory

        # Assign llm_output.csv data to variables and clean data
        self.df = pd.read_csv(os.path.join(data_path, input_file))
        self.df["score"] = pd.to_numeric(self.df["score"], errors="coerce")
        self.df = self.df.dropna(subset=["score", "race_group"])
 
        self.scores = self.df["score"].values
        self.groups = self.df["race_group"].values
        self.unique_groups = sorted(np.unique(self.groups))
 
        print(f"Loaded {len(self.df)} rows | Groups: {self.unique_groups}\n")
 
    def _save_fig(self, filename):
        """
        Helper to save and close a figure.
        """
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filename}\n")
 
    def _get_group_scores(self, group):
        """
        Helper to get scores for a single group.
        """
        return self.scores[self.groups == group]

    # 1. Mean Score Differences

    def mean_score_difference(self, scores_a, scores_b):
        """
        Raw difference in mean scores between two groups A and B.
        """
        print("=" * 40)
        print("1. MEAN SCORE DIFFERENCES")
        print("=" * 40)

        summary = self.df.groupby("race_group")["score"].agg(["count", "mean", "std", "median"]).round(2)
        print(summary, "\n")

        for g_a, g_b in combinations(self.unique_groups, 2): # For each group combination
            delta = self._get_group_scores(g_a).mean() - self._get_group_scores(g_b).mean() # Calculate mean differences
            print(f"  {g_a} vs {g_b}: Δμ = {delta:+.2f}") # Print delta mu
        print()
 
        # Graph: Boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.df, x="race_group", y="score", hue="race_group",
                    order=self.unique_groups, palette="Set2", legend=False, ax=ax)
        ax.axhline(y=self.threshold, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({self.threshold})")
        ax.set_title("Score Distribution by Race Group")
        ax.legend()
        self._save_fig("score_distributions.png")
 
        summary.to_csv(os.path.join(self.output_dir, "descriptive_stats.csv")) # Write stats to csv
        return summary
    
    # 2. Welch's t-test
 
    def welch_t_test(self, scores_a, scores_b):
        """
        Calculates whether the score gap is statistically real.
        """
        print("=" * 40)
        print("2. WELCH'S T-TEST")
        print("=" * 40)

        rows = [] # Save stats for each row
        for g_a, g_b in combinations(self.unique_groups, 2): # For each group combination
            t_stat, p_val = stats.ttest_ind(
                self._get_group_scores(g_a), self._get_group_scores(g_b), equal_var=False
            )
            sig = "Yes" if p_val < 0.05 else "No" # Is gap significant?
            print(f"  {g_a} vs. {g_b}: t={t_stat:+.3f}, p={p_val:.4f} - significant: {sig}")
            rows.append({"pair": f"{g_a} vs. {g_b}", "t_stat": t_stat, "p_value": p_val})
 
        results_df = pd.DataFrame(rows)
        print()
 
        # Graph: p-value Heatmap
        n = len(self.unique_groups)
        matrix = pd.DataFrame(np.ones((n, n)), index=self.unique_groups, columns=self.unique_groups)
        for g_a, g_b in combinations(self.unique_groups, 2):
            _, p = stats.ttest_ind(self._get_group_scores(g_a), self._get_group_scores(g_b), equal_var=False)
            matrix.loc[g_a, g_b] = p
            matrix.loc[g_b, g_a] = p
 
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt=".4f", cmap="RdYlGn", vmin=0, vmax=0.1, linewidths=1, ax=ax)
        ax.set_title("Welch's t-test p-values (red = significant)")
        self._save_fig("welch_pvalues.png")
 
        results_df.to_csv(os.path.join(self.output_dir, "welch_tests.csv"), index=False)
        return results_df

    # 3. Cohen's d

    def cohens_d(self, scores_a, scores_b):
        """
        Calculates effect size to see if gap practically meaningful
        
        d = (mu_A - mu_B) / pooled_std
        """
        print("=" * 40)
        print("3. COHEN'S d")
        print("=" * 40)
 
        rows = []
        for g_a, g_b in combinations(self.unique_groups, 2):
            sa, sb = self._get_group_scores(g_a), self._get_group_scores(g_b)
            n_a, n_b = len(sa), len(sb)
            # Pooled std. combines the variance from two groups into one (weighted by their sample sizes)
            pooled_std = np.sqrt(((n_a - 1) * sa.var(ddof=1) + (n_b - 1) * sb.var(ddof=1)) / (n_a + n_b - 2)) # ddof= 1 computes sample variance
            d = (sa.mean() - sb.mean()) / pooled_std
            # Gap interpretation scale by Jacob Cohen
            abs_d = abs(d)
            size = "negligible" if abs_d < 0.2 else "small" if abs_d < 0.5 else "medium" if abs_d < 0.8 else "large"
 
            print(f"  {g_a} vs {g_b}: d={d:+.3f} ({size})")
            rows.append({"pair": f"{g_a} vs {g_b}", "cohens_d": d, "effect_size": size})
 
        results_df = pd.DataFrame(rows)
        print()
 
        # Graph: Horizontal bar chart
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(results_df["pair"], results_df["cohens_d"], color="#fc8d62")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.axvline(x=0.2, color="gray", linestyle=":", alpha=0.5, label="small (0.2)")
        ax.axvline(x=-0.2, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Cohen's d")
        ax.set_title("Effect Size — Pairwise Comparisons")
        ax.legend()
        self._save_fig("cohens_d.png")
 
        results_df.to_csv(os.path.join(self.output_dir, "cohens_d.csv"), index=False)
        return results_df

    # 4. Disparity Ratio

    def disparity_ratio(self):
        """
        Disparity ratio using EEOC four-fifths rule
        """
        print("=" * 40)
        print(f"4. DISPARITY RATIO (threshold = {self.threshold})")
        print("=" * 40)
 
        # Compute selection rate per group
        selection_rates = {}
        for g in self.unique_groups: 
            g_scores = self._get_group_scores(g)
            selection_rates[g] = (g_scores >= self.threshold).sum() / len(g_scores)
 
        majority = max(selection_rates, key=selection_rates.get)
        print(f"Majority group: {majority} (SR = {selection_rates[majority]:.3f})")
 
        rows = []
        for g, sr in selection_rates.items():
            dir_val = sr / selection_rates[majority] # Disparity ratio calculation
            flag = " <- ADVERSE IMPACT" if dir_val < 0.8 and g != majority else "" # Four-fifth ration flagging adverse impact on racial group
            if g != majority:
                print(f"  {g}: SR={sr:.3f}, DIR={dir_val:.3f}{flag}")
            rows.append({"group": g, "selection_rate": sr, "dir": dir_val, "adverse_impact": dir_val < 0.8})
 
        results_df = pd.DataFrame(rows)
        print()
 
        # Graph: Disparity ratios
        non_majority = results_df[results_df["group"] != majority]
        colors = ["#e74c3c" if ai else "#2ecc71" for ai in non_majority["adverse_impact"]]
 
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(non_majority["group"], non_majority["dir"], color=colors)
        ax.axhline(y=0.8, color="red", linestyle="--", linewidth=2, label="Four-fifths threshold (0.8)")
        ax.set_ylabel("Disparity Ratio")
        ax.set_title("Disparity Ratio vs. Majority Group")
        ax.set_ylim(0, 1.2)
        ax.legend()
        self._save_fig("disparity_ratios.png")
 
        results_df.to_csv(os.path.join(self.output_dir, "disparity_ratios.csv"), index=False)
        return results_df

    # 5. PMI (Pointwise Mutual Information) Proxy Markers

    def compute_pmi(self, min_count=5): # Ignores terms that appear less than five times
        """
        Pointwise Mutual Information Computation: 
        Which Rationale words are disproportionately linked to which groups?
        """
        print("=" * 40)
        print("5. PMI PROXY MARKERS")
        print("=" * 40)
 
        rationales = self.df["rationale"].fillna("").tolist() # Converts Rationales to list, fills NA with empty string (so code does not crash)
        group_list = self.df["race_group"].tolist() # Converts racial group name to list
        N = len(rationales)
 
        group_counts = Counter(group_list) # Counts how many Rationales belong to each race group
        term_group_counts = Counter() # Will track how many times a specific word appears with a specific group
        term_counts = Counter() # Will track how many times a word appears across all groups
 
        for doc, group in zip(rationales, group_list): # Loops through every rationale and corresponding group at the same time
            for term in set(str(doc).lower().split()): # Standardizes terms and counts word appearance per group and overall
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
 
        pmi_df = pd.DataFrame(rows).sort_values("pmi", ascending=False).reset_index(drop=True)
        print(pmi_df.head(20).to_string(index=False))
        print()
 
        # Graph: Top 10 terms per group
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for i, g in enumerate(self.unique_groups):
            ax = axes.flatten()[i]
            group_pmi = pmi_df[pmi_df["group"] == g].head(10)
            if not group_pmi.empty:
                ax.barh(group_pmi["term"], group_pmi["pmi"])
                ax.invert_yaxis()
            ax.set_title(f"Top Proxy Markers → {g}")
            ax.set_xlabel("PMI")
        self._save_fig("pmi_proxy_markers.png")
 
        pmi_df.to_csv(os.path.join(self.output_dir, "pmi_proxy_markers.csv"), index=False)
        return pmi_df

    # 6. Embedding Analysis
    
    def embedding_analysis(self, top_k=5):
        """
        TF-IDF embeddings -> group centroids -> direction vectors + PCA + Cosine similarity
        """
        print("=" * 40)
        print("6. EMBEDDING ANALYSIS")
        print("=" * 40)
 
        # Build TF-IDF embeddings of rationale text and group centroids
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=3) # Create TF-IDF vectorizer
        tfidf_matrix = vectorizer.fit_transform(self.df["rationale"].fillna("")) # Run vectorizer on every Rationale
        feature_names = vectorizer.get_feature_names_out() # Grab actual words that match each column
 
        centroids = {}
        for g in self.unique_groups: # For each race group average all Rationale vectors per group to a single vector (centroid)
            centroids[g] = tfidf_matrix[self.groups == g].toarray().mean(axis=0)
 
        # Direction vectors: Find terms most associated with each group via embedding direction vector
        print("\n  Terms most associated with each group:\n")
        for g_a, g_b in combinations(self.unique_groups, 2):
            direction = centroids[g_a] - centroids[g_b] # Direction vector between two groups
            ranked = sorted(zip(feature_names, direction), key=lambda x: x[1], reverse=True) # Pairs word with its direction and sorts scores
 
            toward_a = [(t, round(s, 3)) for t, s in ranked[:top_k] if s > 0] # Grab top k words towards group A and returns magnitude
            toward_b = [(t, round(abs(s), 3)) for t, s in ranked[-top_k:] if s < 0] # Grab top k words towards group B and returns magnitude
 
            print(f"  {g_a} vs {g_b}:")
            print(f" -> {g_a}: {', '.join(f'{t} ({s})' for t, s in toward_a)}")
            print(f" -> {g_b}: {', '.join(f'{t} ({s})' for t, s in toward_b)}")
            print()
 
        # Graph: PCA scatter plot
        pca = PCA(n_components=2)
        coords = pca.fit_transform(tfidf_matrix.toarray())
 
        fig, ax = plt.subplots(figsize=(10, 7))
        for g in self.unique_groups:
            mask = self.groups == g
            ax.scatter(coords[mask, 0], coords[mask, 1], label=g, alpha=0.5, s=30)
            ax.scatter(coords[mask, 0].mean(), coords[mask, 1].mean(),
                       marker="X", s=200, edgecolors="black", linewidths=1.5, zorder=5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("Rationale Embeddings - PCA by Race Group (X = centroid)")
        ax.legend()
        self._save_fig("embedding_pca.png")
 
        # Cosine Similarity: Find cosine similarity between all group rationale centroids
        # Graph: Cosine similarity heatmap
        n = len(self.unique_groups)
        sim_matrix = pd.DataFrame(np.zeros((n, n)), index=self.unique_groups, columns=self.unique_groups) # Create empty grid
        for g_a in self.unique_groups: # Compute cosine similarity between every group pairs centroids
            for g_b in self.unique_groups: # For each race group
                a, b = centroids[g_a], centroids[g_b]
                sim_matrix.loc[g_a, g_b] = round(np.dot(a, b) / (norm(a) * norm(b)), 4)
 
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(sim_matrix.astype(float), annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=sim_matrix.values.min() * 0.9, vmax=1.0, linewidths=1, ax=ax)
        ax.set_title("Cosine Similarity Between Group Centroids")
        self._save_fig("embedding_similarity.png")
 
        return sim_matrix
    
    # RUN ALL

    def run_bias_quantification_layer(self):
        """
        Runs full analysis in order
        """
        self.mean_score_differences()
        self.welch_t_test()
        self.cohens_d()
        self.disparity_ratio()
        self.compute_pmi()
        self.embedding_analysis()
        print("=" * 40)
        print(f"DONE - All outputs in: {self.output_dir}/")
        print("=" * 40)