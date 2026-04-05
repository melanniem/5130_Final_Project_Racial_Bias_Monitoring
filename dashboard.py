import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="LLM Racial Bias Audit Dashboard",
    page_icon="🔍",
    layout="wide"
)

PALETTE = {
    'Asian or Pacific Islander': '#4C72B0',
    'Black or African American': '#DD8452',
    'Hispanic': '#55A868',
    'Null Baseline': '#aaaaaa',
    'White': '#C44E52'
}

PAGES = [
    "Score Distributions",
    "Pairwise Similarity",
    "Score by Job × Race",
    "PMI Proxy Markers",
    "Statistical Tests"
]

# Sidebar
st.sidebar.title("Bias Audit")
st.sidebar.markdown("**CS5130 · Racial Bias in LLM Resume Scoring**")
st.sidebar.divider()

llm_path = st.sidebar.text_input("Path to llm_outputs.csv", value="results/llm_outputs.csv")
pmi_path = st.sidebar.text_input("Path to pmi_proxy_markers.csv", value="results/pmi_proxy_markers.csv")
welch_path = st.sidebar.text_input("welch_tests.csv", value="results/welch_tests.csv")
disparity_path = st.sidebar.text_input("disparity_ratios.csv", value="results/disparity_ratios.csv")
cohens_path = st.sidebar.text_input("cohens_d.csv", value="results/cohens_d.csv")
desc_path = st.sidebar.text_input("descriptive_stats.csv", value="results/descriptive_stats.csv")

st.sidebar.divider()
page = st.sidebar.radio("Section", PAGES, key="nav")

# Load Data
@st.cache_data
def load_llm(path):
    df = pd.read_csv(path)
    df.drop(columns=[c for c in ['raw_response', 'mean_correct', 'temperature'] if c in df.columns], inplace=True)
    return df

@st.cache_data
def load_pmi(path):
    return pd.read_csv(path)

def try_load(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

try:
    llm_df = load_llm(llm_path)
except FileNotFoundError:
    st.error(f"Could not find `{llm_path}`. Update the path in the sidebar.")
    st.stop()

try:
    pmi_df = load_pmi(pmi_path)
    pmi_loaded = True
except FileNotFoundError:
    pmi_loaded = False

# Adds filter on race group and job_title on all pages except PMI
if page not in ("PMI Proxy Markers", "Statistical Tests"):
    st.sidebar.divider()
    all_groups = sorted(llm_df['race_group'].dropna().unique())
    selected_groups = st.sidebar.multiselect("Race Groups", all_groups, default=all_groups)
    all_jobs = sorted(llm_df['job_title'].dropna().unique())
    selected_jobs = st.sidebar.multiselect("Job Titles", all_jobs, default=all_jobs)
    filtered = llm_df[
        llm_df['race_group'].isin(selected_groups) &
        llm_df['job_title'].isin(selected_jobs)
    ]
else:
    filtered = llm_df
    selected_groups = sorted(llm_df['race_group'].dropna().unique())



# PAGE 1 — Score Distributions

if page == "Score Distributions":
    st.title("Score Distributions by Race Group")

    tab1, tab2, tab3 = st.tabs(["Subplots by Group", "Violin Plot", "Statistical Metrics"])

    with tab1:
        groups = selected_groups
        n = len(groups)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        axes = axes.flatten()
        for i, (ax, group) in enumerate(zip(axes, groups)):
            subset = filtered[filtered['race_group'] == group]['score']
            ax.hist(subset, bins=20, color=PALETTE.get(group, '#555'), alpha=0.85, edgecolor='white')
            ax.set_title(group, fontsize=10, wrap=True)
            ax.set_xlabel("Score", fontsize=9)
            ax.set_ylabel("Frequency" if i % 3 == 0 else "", fontsize=9)
            ax.tick_params(labelsize=8)
        # hide unused subplots if fewer than 6 groups
        for j in range(len(groups), 6):
            axes[j].set_visible(False)
        fig.suptitle("Score Distribution by Race Group", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = sorted(selected_groups)
        palette_list = [PALETTE.get(g, '#333') for g in order]
        sns.violinplot(
            data=filtered, x='race_group', y='score',
            order=order, palette=palette_list,
            inner='quartile', ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)
        ax.set_title("Score Distribution by Race Group (Violin)")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.dataframe(
            filtered.groupby('race_group')['score'].describe().round(2),
            use_container_width=True
        )


# PAGE 2 — Pairwise Similarity Heatmap

elif page == "Pairwise Similarity":
    st.title("Pairwise Mean Score Difference")
    st.caption("Closer to 0 = more similar scoring. Red = large gap = potential bias signal.")

    mean_scores = filtered.groupby('race_group')['score'].mean()
    groups_order = sorted(mean_scores.index)
    mean_scores = mean_scores[groups_order]
    vals = mean_scores.values
    diff_matrix = np.abs(vals[:, None] - vals[None, :])
    diff_df = pd.DataFrame(diff_matrix, index=groups_order, columns=groups_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        diff_df, annot=True, fmt='.2f', cmap='RdYlGn_r',
        ax=ax, linewidths=0.5, annot_kws={"size": 11}
    )
    ax.set_title("Pairwise Mean Score Difference by Race Group", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

# PAGE 3 — Score by Job x Race

elif page == "Score by Job × Race":
    st.title("Score by Job Title × Race Group")

    tab1, tab2 = st.tabs(["Mean Score", "Median Score"])

    for tab, agg_fn, label in [(tab1, 'mean', 'Mean'), (tab2, 'median', 'Median')]:
        with tab:
            pivot = filtered.groupby(['job_title', 'race_group'])['score'].agg(agg_fn).unstack()
            winner = pivot.copy()
            winner['Top Race'] = pivot.idxmax(axis=1)

            st.subheader(f"{label} Score per Job × Race")
            st.dataframe(pivot.round(2), use_container_width=True)

            st.subheader("Which race scores highest per job?")
            top_counts = winner['Top Race'].value_counts().reset_index()
            top_counts.columns = ['Race Group', 'Jobs Won']

            for _, row in top_counts.iterrows():
                color = PALETTE.get(row['Race Group'], '#555')
                jobs = int(row['Jobs Won'])
                st.markdown(
                    f"<span style='color:{color}; font-size:16px; font-weight:bold'>● {row['Race Group']}</span>"
                    f"&nbsp; scored highest in &nbsp;<span style='font-size:18px; font-weight:bold'>{jobs}</span>"
                    f"&nbsp; out of {len(pivot)} job title(s)",
                    unsafe_allow_html=True
                )

            # Mean score bar chart per group
            st.markdown("#### Mean Score by Race Group")
            mean_by_group = filtered.groupby('race_group')['score'].mean().sort_values()
            fig_m, ax_m = plt.subplots(figsize=(8, max(2, len(mean_by_group) * 0.7)))
            fig_m.patch.set_facecolor('white')
            ax_m.set_facecolor('white')
            bars = ax_m.barh(
                mean_by_group.index, mean_by_group.values,
                color=[PALETTE.get(g, '#555') for g in mean_by_group.index],
                edgecolor='none', height=0.5
            )
            for bar in bars:
                ax_m.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                          f"{bar.get_width():.1f}", va='center', fontsize=10, fontweight='bold')
            ax_m.set_xlabel("Mean Score", fontsize=9)
            ax_m.set_xlim(0, mean_by_group.max() + 8)
            ax_m.spines['top'].set_visible(False)
            ax_m.spines['right'].set_visible(False)
            ax_m.spines['left'].set_color('#cccccc')
            ax_m.spines['bottom'].set_color('#cccccc')
            ax_m.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig_m)

            st.subheader("Heatmap")
            fig2, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.4)
            ax.set_title(f"{label} Score by Job and Race Group")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)


# PAGE 4 — PMI Proxy Markers

elif page == "PMI Proxy Markers":
    st.title("PMI Proxy Markers — Top Terms by Race Group")
    st.caption("PMI (Pointwise Mutual Information) measures how strongly a term is associated with a race group in LLM rationales.")

    if not pmi_loaded:
        st.error(f"Could not find `{pmi_path}`. Update the path in the sidebar.")
        st.stop()

    groups_pmi = sorted(pmi_df['group'].unique())
    BAR_COLORS = {
        'Asian or Pacific Islander': '#4C72B0',
        'Black or African American': '#DD8452',
        'Hispanic':                  '#55A868',
        'Null Baseline':             '#8C8C8C',
        'White':                     '#C44E52'
    }

    top_n_bar = st.slider("Top N terms", 5, 20, 10)

    st.subheader("Top PMI Terms per Race Group")
    fig_bar, axes_bar = plt.subplots(2, 3, figsize=(18, 11))
    fig_bar.patch.set_facecolor('white')
    axes_bar = axes_bar.flatten()

    for i, group in enumerate(groups_pmi):
        ax = axes_bar[i]
        gdf = pmi_df[pmi_df['group'] == group].nlargest(top_n_bar, 'pmi')
        color = BAR_COLORS.get(group, '#555')
        bars = ax.barh(gdf['term'][::-1], gdf['pmi'][::-1], color=color, edgecolor='none', height=0.65)
        for bar in bars:
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2f}", va='center', fontsize=7.5, color='#333')
        ax.set_facecolor('white')
        ax.set_title(f"Top Proxy Markers → {group}", fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel("PMI", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')

    for j in range(len(groups_pmi), 6):
        axes_bar[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig_bar)

    st.divider()
    st.subheader("PMI Table by Race Group")

    row1_groups = groups_pmi[:3]
    row2_groups = groups_pmi[3:]

    for row_groups in [row1_groups, row2_groups]:
        cols = st.columns(3)
        for col, group in zip(cols, row_groups):
            color = BAR_COLORS.get(group, '#555')
            gdf = pmi_df[pmi_df['group'] == group].nlargest(top_n_bar, 'pmi')[['term', 'pmi', 'count']].reset_index(drop=True)
            with col:
                st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:15px'>● {group}</span>", unsafe_allow_html=True)
                st.dataframe(gdf, use_container_width=True)


# PAGE 5 — Statistical Tests
elif page == "Statistical Tests":
    st.title("Statistical Bias Analysis")
    st.caption(
        "Welch's t-tests, Cohen's d effect sizes, descriptive stats, and adverse impact ratios across race groups.")

    welch_df = try_load(welch_path)
    disparity_df = try_load(disparity_path)
    cohens_df = try_load(cohens_path)
    desc_df = try_load(desc_path)

    # Descriptive Stats
    st.subheader(" Descriptive Statistics")
    if desc_df is not None:
        desc_df = desc_df.set_index('race_group')
        fig_d, ax_d = plt.subplots(figsize=(10, 4))
        fig_d.patch.set_facecolor('white')
        ax_d.set_facecolor('white')
        x = np.arange(len(desc_df))
        bars = ax_d.bar(x, desc_df['mean'],
                        color=[PALETTE.get(g, '#555') for g in desc_df.index],
                        edgecolor='none', width=0.5, zorder=3)
        ax_d.errorbar(x, desc_df['mean'], yerr=desc_df['std'],
                      fmt='none', color='#333', capsize=5, linewidth=1.5, zorder=4)
        for bar, val in zip(bars, desc_df['mean']):
            ax_d.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + desc_df['std'].max() + 0.5,
                      f"{val:.1f}", ha='center', fontsize=9, fontweight='bold')
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(desc_df.index, rotation=20, ha='right', fontsize=9)
        ax_d.set_ylabel("Mean Score", fontsize=9)
        ax_d.set_title("Mean Score ± Std Dev by Race Group", fontsize=12)
        ax_d.set_ylim(0, desc_df['mean'].max() + desc_df['std'].max() + 10)
        ax_d.spines['top'].set_visible(False)
        ax_d.spines['right'].set_visible(False)
        ax_d.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        plt.tight_layout()
        st.pyplot(fig_d)
        st.dataframe(desc_df.round(2), use_container_width=True)
    else:
        st.warning(f"Could not load `{desc_path}`")

    st.divider()

    # Welch + Cohen's d
    st.subheader("Welch's t-test  ×  Cohen's d")
    st.caption("p < 0.05 = statistically significant. Cohen's d: negligible < 0.2 · small 0.2–0.5 · medium 0.5–0.8")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**p-value Heatmap** (green = not significant · red = significant)")
        if welch_df is not None:
            groups_w = sorted(set(
                [p.split(' vs. ')[0] for p in welch_df['pair']] +
                [p.split(' vs. ')[1] for p in welch_df['pair']]
            ))
            p_matrix = pd.DataFrame(np.nan, index=groups_w, columns=groups_w)
            for _, row in welch_df.iterrows():
                a, b = row['pair'].split(' vs. ')
                p_matrix.loc[a, b] = row['p_value']
                p_matrix.loc[b, a] = row['p_value']
            p_arr = p_matrix.to_numpy(copy=True)
            np.fill_diagonal(p_arr, 1.0)
            p_matrix = pd.DataFrame(p_arr, index=p_matrix.index, columns=p_matrix.columns)

            fig_w, ax_w = plt.subplots(figsize=(6, 5))
            sns.heatmap(p_matrix.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                        ax=ax_w, linewidths=0.5, vmin=0, vmax=0.1, annot_kws={"size": 8})
            for i in range(len(groups_w)):
                ax_w.text(i + 0.5, i + 0.5, '—', ha='center', va='center', fontsize=10, color='gray')
            ax_w.set_xticklabels(ax_w.get_xticklabels(), rotation=30, ha='right', fontsize=7)
            ax_w.set_yticklabels(ax_w.get_yticklabels(), rotation=0, fontsize=7)
            ax_w.set_title("p-values (Welch's t-test)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_w)

            sig = welch_df.copy()
            sig['verdict'] = sig['p_value'].apply(lambda p: '✅ Significant' if p < 0.05 else '— Not significant')
            st.dataframe(sig[['pair', 'p_value', 'verdict']].round({'p_value': 4}),
                         use_container_width=True, hide_index=True)
        else:
            st.warning(f"Could not load `{welch_path}`")

    with col2:
        st.markdown("**Cohen's d Heatmap** (effect size magnitude)")
        if cohens_df is not None:
            groups_c = sorted(set(
                [p.split(' vs ')[0] for p in cohens_df['pair']] +
                [p.split(' vs ')[1] for p in cohens_df['pair']]
            ))
            d_matrix = pd.DataFrame(np.nan, index=groups_c, columns=groups_c)
            for _, row in cohens_df.iterrows():
                a, b = row['pair'].split(' vs ')
                d_matrix.loc[a, b] = abs(row['cohens_d'])
                d_matrix.loc[b, a] = abs(row['cohens_d'])
            d_arr = d_matrix.to_numpy(copy=True)
            np.fill_diagonal(d_arr, 0.0)
            d_matrix = pd.DataFrame(d_arr, index=d_matrix.index, columns=d_matrix.columns)

            fig_c, ax_c = plt.subplots(figsize=(6, 5))
            sns.heatmap(d_matrix.astype(float), annot=True, fmt='.2f', cmap='YlOrRd',
                        ax=ax_c, linewidths=0.5, vmin=0, vmax=0.5, annot_kws={"size": 8})
            ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=30, ha='right', fontsize=7)
            ax_c.set_yticklabels(ax_c.get_yticklabels(), rotation=0, fontsize=7)
            ax_c.set_title("|Cohen's d| (effect size magnitude)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_c)

            st.dataframe(cohens_df[['pair', 'cohens_d', 'effect_size']].round({'cohens_d': 3}),
                         use_container_width=True, hide_index=True)
        else:
            st.warning(f"Could not load `{cohens_path}`")

    st.divider()

    # Adverse Impact
    st.subheader("Adverse Impact — Disparate Impact Ratio (DIR)")
    st.caption("DIR < 0.80 flags adverse impact under the 4/5ths rule.")

    if disparity_df is not None:
        cols = st.columns(len(disparity_df))
        for col, (_, row) in zip(cols, disparity_df.iterrows()):
            color = PALETTE.get(row['group'], '#555')
            flag = " Adverse Impact" if row['adverse_impact'] else "✅ No Flag"
            with col:
                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding:8px 12px; border-radius:4px; background:#fafafa'>"
                    f"<b style='color:{color}'>{row['group']}</b><br>"
                    f"Selection Rate: <b>{row['selection_rate']:.2f}</b><br>"
                    f"DIR: <b>{row['dir']:.2f}</b><br>"
                    f"{flag}</div>",
                    unsafe_allow_html=True
                )

        st.markdown("")
        disp = disparity_df.sort_values('dir')
        fig_dir, ax_dir = plt.subplots(figsize=(8, 3.5))
        fig_dir.patch.set_facecolor('white')
        ax_dir.set_facecolor('white')
        bar_colors = ['#e74c3c' if ai else PALETTE.get(g, '#555')
                      for g, ai in zip(disp['group'], disp['adverse_impact'])]
        bars = ax_dir.barh(disp['group'], disp['dir'], color=bar_colors, edgecolor='none', height=0.5)
        ax_dir.axvline(0.8, color='#e74c3c', linestyle='--', linewidth=1.5, label='4/5ths threshold (0.80)')
        for bar in bars:
            ax_dir.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.2f}", va='center', fontsize=9, fontweight='bold')
        ax_dir.set_xlabel("Disparate Impact Ratio", fontsize=9)
        ax_dir.set_xlim(0, 1.15)
        ax_dir.legend(fontsize=8)
        ax_dir.spines['top'].set_visible(False)
        ax_dir.spines['right'].set_visible(False)
        ax_dir.tick_params(labelsize=9)
        ax_dir.set_title("Disparate Impact Ratio by Race Group", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_dir)
    else:
        st.warning(f"Could not load `{disparity_path}`")