import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, kruskal
import plotly.graph_objects as go
import io


st.set_page_config(page_title="Milk Composition Overview", layout="wide")
st.title(' Milk Composition')

hmo = pd.read_excel("/Users/kspann/Desktop/Oxford/Cleaned Data/HMO/hmo_data_oxford.xlsx")

# Define categories
n_colostrum = hmo[hmo["PP_day_num"].between(1, 5, inclusive="both")].shape[0]
n_mature = (hmo["PP_day_num"] == 6).sum()


# Create 3 equal-width columns for layout (you can change the number)
col1, col2, col3, col4 = st.columns(4)

# Show the unique participant count as a card
with col1:
    st.metric(
        label="Unique Participants",
        value=int(hmo['Participant ID'].nunique())
   )  
with col2:
    st.metric(
        label="Total Milk Samples",
        value=int(hmo.shape[0])
   )  
with col3:
    st.metric("Colostrum Samples (Day 1-5)", int(n_colostrum))
with col4:
    st.metric("Mature Milk Samples (Day 6)", int(n_mature))

st.divider()



# ---- Identify columns ----
# Required identifiers (best effort)
# ---- Identify columns ----
id_cols = [c for c in ["Sample Name", "Participant ID", "PP day", "secretor_status", "PP_day_num"] if c in hmo.columns]

# HMO columns with µg/mL
hmo_cols = [c for c in hmo.columns if "(µg/mL)" in c]
has_sum_ug = "SUM (µg/mL)" in hmo.columns


# Canonical HMO order (exact Excel column order)
hmo_order = [c for c in hmo.columns if "(µg/mL)" in c]
if not hmo_order:
    st.error("No HMO (µg/mL) columns found.")
    st.stop()


if len(hmo_cols) == 0:
    st.error("Could not find any columns containing '(µg/mL)'. Please check your file.")
    st.stop()

# ---- Sidebar controls ----
st.sidebar.header("Filters")

# Secretor filter
if "secretor_status" in hmo.columns:
    secretor_map = {1: "Secretor", 0: "Non-secretor"}
    secretor_choice = st.sidebar.multiselect(
        "Secretor status",
        options=["Secretor", "Non-secretor"],
        default=["Secretor", "Non-secretor"]
    )
else:
    secretor_choice = None

# PP day filter (use PP_day_num consistently; fall back to PP day if needed)
pp_col = "PP_day_num" if "PP_day_num" in hmo.columns else ("PP day" if "PP day" in hmo.columns else None)
if pp_col is not None:
    pp_numeric_all = pd.to_numeric(hmo[pp_col], errors="coerce")
    pp_min, pp_max = int(np.nanmin(pp_numeric_all)), int(np.nanmax(pp_numeric_all))
    pp_range = st.sidebar.slider("Postpartum day range", min_value=pp_min, max_value=pp_max, value=(pp_min, pp_max))
else:
    pp_range = None

# Sidebar: Heatmap options
include_sum = False
has_sum_ug = "SUM (µg/mL)" in hmo.columns
if has_sum_ug:
    include_sum = st.sidebar.toggle("Include 'SUM (µg/mL)'", value=False)

# Keep Excel order; optionally drop SUM while preserving order
available_hmos = [c for c in hmo_order if (include_sum or c != "SUM (µg/mL)")]

selected_hmos = st.sidebar.multiselect(
    "Select HMOs (µg/mL)",
    options=available_hmos,        # preserves Excel order
    default=available_hmos         # select ALL by default
)
if not selected_hmos:
    st.warning("Please select at least one HMO column.")
    st.stop()

norm = st.sidebar.selectbox(
    "Normalization",
    ["Log10(x+1)", "Row Z-score", "Column Z-score", "Column Min-Max (0–1)"],
    index=1
)

agg_level = st.sidebar.radio(
    "Rows represent",
    ["Sample Name (each sample)", "Participant ID (aggregate)"],
    index=0
)
agg_func = st.sidebar.selectbox("Aggregation (when aggregating by Participant)", ["mean", "median"], index=0)

cluster = st.sidebar.toggle("Cluster rows/columns", value=True)
cluster_rows = st.sidebar.toggle("Cluster rows", value=True, disabled=(not cluster))
cluster_cols = st.sidebar.toggle("Cluster columns", value=False, disabled=(not cluster))

fig_w = st.sidebar.slider("Figure width", 6, 22, 12)
fig_h = st.sidebar.slider("Figure height", 4, 20, 8)

# ---- Apply filters ----
df_filt = hmo.copy()

if secretor_choice is not None:
    labels = df_filt["secretor_status"].map(secretor_map) if df_filt["secretor_status"].dtype != "O" else df_filt["secretor_status"]
    df_filt = df_filt[labels.isin(secretor_choice)]

if pp_range is not None:
    pp_numeric = pd.to_numeric(df_filt[pp_col], errors="coerce")
    df_filt = df_filt[(pp_numeric >= pp_range[0]) & (pp_numeric <= pp_range[1])]


# ---- Build matrix ----
show_cols = selected_hmos
if len(show_cols) == 0:
    st.warning("Please select at least one HMO column to display.")
    st.stop()

if agg_level.startswith("Sample Name") and "Sample Name" in df_filt.columns:
    index_col = "Sample Name"
elif agg_level.startswith("Participant ID") and "Participant ID" in df_filt.columns:
    index_col = "Participant ID"
else:
    index_col = None

mat_df = df_filt[[c for c in [index_col] + show_cols if c is not None]].copy()

if index_col == "Participant ID":
    mat_df = (mat_df.groupby("Participant ID", as_index=True)[show_cols]
                     .agg("mean" if agg_func == "mean" else "median", numeric_only=True))
else:
    mat_df = mat_df.set_index(index_col) if index_col else mat_df.set_index(df_filt.index)

# ---- Normalization / transforms ----
X = mat_df.copy()
if norm == "Log10(x+1)":
    X = np.log10(X + 1.0)
elif norm == "Row Z-score":
    X = X.apply(lambda r: (r - r.mean()) / (r.std(ddof=0) if r.std(ddof=0) != 0 else 1), axis=1)
elif norm == "Column Z-score":
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
elif norm == "Column Min-Max (0–1)":
    X = (X - X.min()) / (X.max() - X.min()).replace(0, 1)

# ---- Render heatmap (flipped orientation) ----
st.subheader("Heatmap")

# Clean matrix
X = X.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=0).dropna(how="all", axis=1)

if X.shape[0] == 0 or X.shape[1] == 0:
    st.warning("No data to plot after filters/selection. Try broadening your filters or selecting more HMOs.")
    st.stop()

# Transpose so HMOs are on Y-axis and samples on X-axis
X_flipped = X.T

# Impute for clustering
X_imp = X_flipped.apply(lambda c: c.fillna(c.median()), axis=0)

# Determine if clustering is feasible
eff_row_cluster = cluster and cluster_rows and (X_imp.shape[0] >= 2)
eff_col_cluster = cluster and cluster_cols and (X_imp.shape[1] >= 2)

if cluster and not (eff_row_cluster or eff_col_cluster):
    st.info("Clustering disabled automatically (need ≥2 rows and/or ≥2 columns). Rendering plain heatmap instead.")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(X_flipped, cmap="viridis", ax=ax)
    ax.set_xlabel(index_col or "Samples")
    ax.set_ylabel("HMOs (µg/mL)" if norm == "None" else f"HMOs (transformed: {norm})")
    st.pyplot(fig, clear_figure=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download heatmap (PNG)", data=buf.getvalue(), file_name="hmo_heatmap_flipped.png", mime="image/png")
    plt.close(fig)
else:
    g = sns.clustermap(
        X_imp,
        method="average",
        metric="euclidean",
        col_cluster=eff_col_cluster,
        row_cluster=eff_row_cluster,
        cmap="viridis",
        figsize=(fig_w, fig_h)
    )
    # The clustermap labels stay consistent (rows = HMOs, columns = samples)
    st.pyplot(g.fig, clear_figure=True)

    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download heatmap (PNG)", data=buf.getvalue(), file_name="hmo_heatmap_flipped.png", mime="image/png")
    plt.close(g.fig)

