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


df = pd.read_excel("Cleaned Data/Merged/merged_hmo_meta.xlsx")

st.title("Infant Growth Metrics")

# --- 1) Define the core + optional columns ---
core_cols = [
    "Participant ID", "baby_birthweight", "Day 5 weight",
    "d5weightchange", "percentd5weightchange"
]

optional_cols = [
    "Gest_Age_Birth", "infantsex", "Gravida", "Parity",
    "Primiparity", "prepreg BMI", "PP_day_num"
]


# --- 2) Build cleaned DataFrame subset ---
selected_cols = core_cols + optional_cols
df_growth = df[selected_cols].copy()
st.session_state.df_growth = df_growth

# st.success(f"Subset created with {len(df_growth)} rows × {len(df_growth.columns)} columns")
# st.dataframe(df_growth.head(), use_container_width=True)

# # Display the final column mapping for clarity
# st.write("**Column mapping used:**")
# st.json(col_map)


# Identify participant ID column
id_col = None
for c in df_growth.columns:
    if "id" in c.lower():
        id_col = c
        break

if id_col is None:
    st.error("Could not automatically find participant ID column. Please check your mapping in Step 2.")
    st.stop()

# st.write(f"Participant ID column detected as: **{id_col}**")


# Count duplicates
dup_counts = df_growth[id_col].value_counts()
n_dupes = (dup_counts > 1).sum()
# if n_dupes > 0:
#     st.warning(f"Found {n_dupes} participants with multiple entries — keeping the first occurrence for each.")

# Keep only the first entry per participant
df_unique = df_growth.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)

# Save cleaned version
st.session_state.df_growth_unique = df_unique

# Display results
# st.success(f"✅ Cleaned dataset now has {df_unique.shape[0]} unique participants (down from {df_growth.shape[0]} rows).")
# st.dataframe(df_unique.head(), use_container_width=True)



# Keep only complete cases for plotting
df_plot = df_unique.dropna(subset=["baby_birthweight", "Day 5 weight", "percentd5weightchange", "Participant ID", "infantsex"]).copy()

# Normalize sex labels a bit
df_plot["infantsex"] = df_plot["infantsex"].astype(str).str.strip().str.title().replace({"F":"Female","M":"Male"})

# --- Remove unrealistic / outlier values ---
# Define your acceptable percent change range (e.g., between -20% and +20%)
lower_limit = -20
upper_limit = 20

# # Optionally, display a summary before filtering
# outlier_rows = df_plot[
#     (df_plot["percentd5weightchange"] < lower_limit) | 
#     (df_plot["percentd5weightchange"] > upper_limit)
# ]
# if not outlier_rows.empty:
#     st.warning(f"Excluding {len(outlier_rows)} outlier(s) beyond {lower_limit}% to {upper_limit}% change.")
#     st.dataframe(outlier_rows)

# Filter them out
df_plot = df_plot[
    (df_plot["percentd5weightchange"] >= lower_limit) &
    (df_plot["percentd5weightchange"] <= upper_limit)
].copy()

# 3) Build long-form for slope graph
long = df_plot.melt(
    id_vars=["Participant ID", "infantsex"],
    value_vars=["baby_birthweight", "Day 5 weight"],
    var_name="Timepoint", value_name="Weight_g"
).replace({"baby_birthweight": "Birth", "Day 5 weight": "Day 5"})


# Quick dashboard stats
colA, colB, colC = st.columns(3)
with colA:
    st.metric("# Infants with Birth & D5 Weight", f"{df_plot.shape[0]}")
with colB:
    st.metric("Median % change (Birth→D5)", f"{np.nanmedian(df_plot['percentd5weightchange']):.1f}%")
with colC:
    pct_loss_over_5 = np.mean(df_plot["percentd5weightchange"] <= -10) * 100      #infants who lost more than 5%
    st.metric("% of Infants with Lost >10% weight loss", f"{pct_loss_over_5:.1f}%")


# 4) Tabs for three complementary views
tab1, tab2, tab3 = st.tabs([
    "Per-infant slope graph",
    "Distribution of % change (negative = loss)",
    "Magnitude of % change (absolute)"
])

with tab1:
    st.caption("Each line represents one infant (Birth → Day 5), colored by sex.")
    fig = px.line(
        long.sort_values(["Participant ID","Timepoint"]),
        x="Timepoint", y="Weight_g",
        color="infantsex",
        line_group="Participant ID",
        markers=True,
        hover_data={
            "Participant ID": True,
            "infantsex": True,
            "Weight_g": ":,.0f",
            "Timepoint": True
        },
        title="Infant Weight Trajectories: Birth vs Day 5"
    )
    fig.update_layout(yaxis_title="Weight (g)", xaxis_title="")
    fig.update_traces(opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption("Negative values reflect expected early neonatal weight loss. Dotted lines at 0% and -10%.")
    fig2 = px.violin(
        df_plot,
        x="infantsex",
        y="percentd5weightchange",
        color="infantsex",
        box=True, points="outliers",
        title="% Weight Change (Birth → Day 5) — Negative = Loss"
    )
    fig2.update_layout(yaxis_title="% change", xaxis_title="Infant sex", showlegend=False)
    fig2.add_hline(y=0, line_dash="dot")
    fig2.add_hline(y=-10, line_dash="dot")
    st.plotly_chart(fig2, use_container_width=True)

    # Optional boxplot version (toggle)
    if st.checkbox("Show boxplot instead of violin", value=False):
        fig2b = px.box(
            df_plot,
            x="infantsex",
            y="percentd5weightchange",
            color="infantsex",
            points="suspectedoutliers",
            title="% Weight Change (Birth → Day 5) — Boxplot"
        )
        fig2b.update_layout(yaxis_title="% change", xaxis_title="Infant sex", showlegend=False)
        fig2b.add_hline(y=0, line_dash="dot")
        fig2b.add_hline(y=-10, line_dash="dot")
        st.plotly_chart(fig2b, use_container_width=True)

with tab3:
    st.caption("Emphasizes *magnitude* of change regardless of gain/loss direction.")
    df_plot["abs_percent_change"] = df_plot["percentd5weightchange"].abs()
    fig3 = px.histogram(
        df_plot,
        x="abs_percent_change",
        nbins=20,
        title="Magnitude of % Weight Change (Absolute Value)"
    )
    fig3.update_layout(xaxis_title="|% change|", yaxis_title="Count", bargap=0.2)
    st.plotly_chart(fig3, use_container_width=True)