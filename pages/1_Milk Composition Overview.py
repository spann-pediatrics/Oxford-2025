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
import altair as alt


st.set_page_config(page_title="Milk Composition Overview", layout="wide")
st.title(' Milk Composition')

hmo = pd.read_excel("/Users/kspann/Desktop/Oxford/Cleaned Data/HMO/hmo_data_oxford.xlsx")

# Define categories
n_colostrum = hmo[hmo["PP_day_num"].between(1, 4, inclusive="both")].shape[0]
n_transition = (hmo["PP_day_num"] == 5).sum()
n_mature = (hmo["PP_day_num"] == 6).sum()


# Create 3 equal-width columns for layout (you can change the number)
col1, col2, col3, col4, col5 = st.columns(5)

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
    st.metric("Colostrum Samples (Day 1-4)", int(n_colostrum))
with col4:
    st.metric("Transitional Milk Samples (Day 5)", int(n_transition))
with col5:
    st.metric("Mature Milk Samples (Day 6)", int(n_mature))

st.divider()



########## HMO Graphs ##################

# ---- Identify columns ----

ID_COL = "Sample Name"        # each bar = one milk sample
PARTICIPANT_COL = "Participant ID" 
PPDAY_COL = "PP_day_num"      # postpartum day
SECRETOR_COL = "secretor_status"

# ---- Exact column lists based on your file ----
UG_ALL = [
    '2FL (µg/mL)', '3FL (µg/mL)', 'DFLac (µg/mL)', '3SL (µg/mL)', '6SL (µg/mL)',
    'LNT (µg/mL)', 'LNnT (µg/mL)', 'LNFP I (µg/mL)', 'LNFP II (µg/mL)',
    'LNFP III (µg/mL)', 'LSTb (µg/mL)', 'LSTc (µg/mL)', 'DFLNT (µg/mL)',
    'LNH (µg/mL)', 'DSLNT (µg/mL)', 'FLNH (µg/mL)', 'DFLNH (µg/mL)',
    'FDSLNH (µg/mL)', 'DSLNH (µg/mL)'
]
UG_SUM = 'SUM (µg/mL)'

PCT_ALL = [
    '2FL (%)', '3FL (%)', 'DFLac (%)', '3SL (%)', '6SL (%)',
    'LNT (%)', 'LNnT (%)', 'LNFP I (%)', 'LNFP II (%)', 'LNFP III (%)',
    'LSTb (%)', 'LSTc (%)', 'DFLNT (%)', 'LNH (%)', 'DSLNT (%)',
    'FLNH (%)', 'DFLNH (%)', 'FDSLNH (%)', 'DSLNH (%)'
]
PCT_SUM = 'SUM (%)'


# =======================================
# 2) Type hygiene (once, on the raw frame)
# =======================================
# Assumes your DataFrame is named `hmo`
hmo = hmo.copy()

# Ensure identifier columns exist
for req in [ID_COL, SECRETOR_COL, PPDAY_COL]:
    if req not in hmo.columns:
        st.error(f"Required column missing: {req}")
        st.stop()

# Coerce filters to numeric
hmo[SECRETOR_COL] = pd.to_numeric(hmo[SECRETOR_COL], errors="coerce")
hmo[PPDAY_COL]    = pd.to_numeric(hmo[PPDAY_COL], errors="coerce")


# ==========================
# 3) Sidebar: user controls
# ==========================
with st.sidebar:
    st.subheader("Filters")

    sec_label = st.selectbox(
        "Secretor status",
        ["All", "Secretor", "Non-secretor"],
        index=0
    )

    all_days = (
        hmo[PPDAY_COL]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    days_selected = st.multiselect(
        "PP day (choose one or more)",
        options=all_days,
        default=all_days
    )

    # IMPORTANT: The label string here is what we compare below
    view_mode = st.radio(
        "View Mode:",
        ["Concentration (ug/mL)", "Relative Abundance (%)"],
        horizontal=False
    )



# =================
# 4) Apply filters
# =================
filt = hmo.copy()

if sec_label == "Secretor":
    filt = filt[filt[SECRETOR_COL] == 1]
elif sec_label == "Non-secretor":
    filt = filt[filt[SECRETOR_COL] == 0]

if days_selected and len(days_selected) < len(all_days):
    filt = filt[filt[PPDAY_COL].isin(days_selected)]

if filt.empty:
    st.warning("No rows after filtering. Try a different combination.")
    st.stop()

st.caption(
    f"Showing **{len(filt)}** samples "
    f"({len(np.unique(filt[PARTICIPANT_COL])) if PARTICIPANT_COL in filt.columns else 'n/a'} participants)"
)




# ===================================================
# 5) Choose columns to plot (based on the view toggle)
# ===================================================
# Build valid lists of column NAMES that are present after filtering
hmo_cols_ug  = [c for c in UG_ALL  if c in filt.columns]   # exclude SUM by design
hmo_cols_pct = [c for c in PCT_ALL if c in filt.columns]   # exclude SUM by design

# Match the *exact* radio label used above
view_is_ug = (view_mode == "Concentration (ug/mL)")

if view_is_ug:
    value_cols = hmo_cols_ug
    VALUE_COL  = "Concentration (µg/mL)"
else:
    value_cols = hmo_cols_pct
    VALUE_COL  = "Relative composition (%)"

if not value_cols:
    st.error("No HMO columns available for the selected view.")
    st.stop()

# Make selected columns numeric and non-negative
for c in value_cols:
    filt[c] = pd.to_numeric(filt[c], errors="coerce")

# If % data look like proportions (0–1), convert to %
if not view_is_ug:
    mx = filt[value_cols].to_numpy(dtype=float)
    mx = np.nanmax(mx) if mx.size else np.nan
    if pd.notna(mx) and mx <= 1.5:
        filt[value_cols] = filt[value_cols] * 100.0






# ======================================
# 6) Long format + x-axis sort by PP day
# ======================================
wide = filt[[ID_COL] + value_cols].copy()

long = wide.melt(
    id_vars=ID_COL,
    value_vars=value_cols,
    var_name="HMO",
    value_name=VALUE_COL
)
long[VALUE_COL] = pd.to_numeric(long[VALUE_COL], errors="coerce").clip(lower=0)
long["HMO"] = pd.Categorical(long["HMO"], categories=value_cols, ordered=True)

# Sort x by PP day (then Sample Name) if available
if PPDAY_COL in filt.columns:
    order_df = (
        filt[[ID_COL, PPDAY_COL]]
        .dropna()
        .sort_values([PPDAY_COL, ID_COL])
    )
    x_order = order_df[ID_COL].tolist()
else:
    x_order = long[ID_COL].drop_duplicates().tolist()





# =================
# 7) Draw the chart
# =================
chart = (
    alt.Chart(long)
    .mark_bar()
    .encode(
        x=alt.X(f"{ID_COL}:N", sort=x_order, title="Sample"),
        y=alt.Y(f"{VALUE_COL}:Q", stack="zero", title=VALUE_COL),
        color=alt.Color("HMO:N", legend=alt.Legend(title="HMO")),
        tooltip=[ID_COL, "HMO", VALUE_COL]
    )
    .properties(height=450)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)







# --- Additional chart using the same filtered data ---

st.markdown(f"### Average HMO {'Concentration' if view_is_ug else 'Relative Abundance'} by PP day")

# Calculate averages per day
avg_by_day = (
    filt.groupby(PPDAY_COL)[value_cols]  #active list of columns ug/mL or %
    .mean()
    .reset_index()
    .melt(id_vars=PPDAY_COL, var_name="HMO", value_name=VALUE_COL)
)

avg_chart = (
    alt.Chart(avg_by_day)
    .mark_bar()
    .encode(
        x=alt.X(f"{PPDAY_COL}:O", title="PP day"),
        y=alt.Y(f"{VALUE_COL}:Q", title=VALUE_COL),
        color=alt.Color("HMO:N", legend=alt.Legend(title="HMO")),
        tooltip=[PPDAY_COL, "HMO", VALUE_COL]
    )
    .properties(height=400)
    .interactive()
)

st.altair_chart(avg_chart, use_container_width=True)



# --- Additional chart using the same filtered data ---

# Define fixed colors for PP days (consistent across filtering)
PPDAY_COLORS = {
    1: "#4E79A7",  # blue
    2: "#F28E2B",  # orange
    3: "#E15759",  # red
    4: "#76B7B2",  # teal
    5: "#59A14F",  # green
    6: "#EDC948",  # yellow
}

# --- Optional: Total HMO (SUM) chart ---

if UG_SUM and UG_SUM in filt.columns:
    st.markdown("### Total HMO Concentration per Sample")

    # Basic chart of total HMO per sample
    total_chart = (
        alt.Chart(filt)
        .mark_bar()
        .encode(
            x=alt.X(f"{ID_COL}:N", sort=x_order, title="Sample"),
            y=alt.Y(f"{UG_SUM}:Q", title="Total HMO (µg/mL)"),
            color=alt.Color(
                f"{PPDAY_COL}:N",
                title="PP day",
                scale=alt.Scale(
                    domain=list(PPDAY_COLORS.keys()),
                    range=list(PPDAY_COLORS.values())
                )
            ),
            tooltip=[ID_COL, PARTICIPANT_COL, PPDAY_COL, UG_SUM]
        )
        .properties(height=400)
        .interactive()
)

    st.altair_chart(total_chart, use_container_width=True)







st.markdown("### HMO distribution by secretor status")

# Prepare long-form data for boxplot
box_data = filt.melt(
    id_vars=SECRETOR_COL,
    value_vars=value_cols,
    var_name="HMO",
    value_name=VALUE_COL
)

# Optional: make secretor numeric values human-readable strings
box_data["Secretor Group"] = box_data[SECRETOR_COL].map({1: "Secretor", 0: "Non-secretor"})

# Build the chart
box_chart = (
    alt.Chart(box_data)
    .mark_boxplot(outliers=True)
    .encode(
        x=alt.X("HMO:N", title="HMO", sort=value_cols),
        y=alt.Y(f"{VALUE_COL}:Q", title=VALUE_COL),
        color=alt.Color(
            "Secretor Group:N",
            title="Secretor status",
            scale=alt.Scale(domain=["Non-secretor", "Secretor"],
                            range=["#E15759", "#4E79A7"])
        )
    )
    .properties(height=400)
    .interactive()
)

st.altair_chart(box_chart, use_container_width=True)
