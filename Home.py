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

st.title("Oxford Metadata and Sample Overview")
st.set_page_config(page_title="Oxford Dashboard", layout="wide", initial_sidebar_state="expanded")


# load metadata overview
df = pd.read_excel('/Users/kspann/Desktop/Oxford/Cleaned Data/Metadata/metadata_overview_oxford.xlsx')
hmo = pd.read_excel('Cleaned Data/HMO/hmo_data_oxford.xlsx')

# Make sure PP_day_num is numeric
hmo["PP_day_num"] = pd.to_numeric(hmo["PP_day_num"], errors="coerce")

# Define categories
n_colostrum = hmo[hmo["PP_day_num"].between(1, 5, inclusive="both")].shape[0]
n_mature = (hmo["PP_day_num"] == 6).sum()


# Create 3 equal-width columns for layout (you can change the number)
col1, col2, col3, col4 = st.columns(4)

# Show the unique participant count as a card
with col1:
    st.metric(
        label="Unique Participants",
        value=int(df['Participant ID'].nunique())
   )  
with col2:
    st.metric(
        label="Total Samples",
        value=int(hmo.shape[0])
   )  
with col3:
    st.metric("Colostrum Samples (Day 1-5)", int(n_colostrum))
with col4:
    st.metric("Mature Milk Samples (Day 6)", int(n_mature))

st.divider()

st.markdown("## Metadata Overview")
left, right = st.columns([1, 1])

with left:
    # Age
    fig_age, ax = plt.subplots(figsize=(8,5), dpi=120)
    ax.hist(
        df["mat_age"].dropna(), 
        bins=20,
        color="#AFB1B3",
        edgecolor="white",
        alpha=0.9)
    ax.set_title("Maternal Age Distribution", fontweight='bold')
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Participants")
    st.pyplot(fig_age, use_container_width=True)

with right:
    # Count + (optional) collapse tiny groups into "Other"
    eth = df["mat_ethnicity"].dropna().astype(str)
    counts = eth.value_counts().reset_index()
    counts.columns = ["Ethnicity", "Count"]
    counts["Percent"] = counts["Count"] / counts["Count"].sum()

    # OPTIONAL: collapse categories under 3% into "Other"
    small = counts["Percent"] < 0.03
    if small.any():
        other_sum = counts.loc[small, "Count"].sum()
        counts = pd.concat([
            counts.loc[~small],
            pd.DataFrame([{"Ethnicity": "Other", "Count": other_sum}])
        ], ignore_index=True)
        counts["Percent"] = counts["Count"] / counts["Count"].sum()

    # Custom, calmer color palette (edit to taste)
    palette = {
        "White": "#4C78A8",
        "Black": "#E45756",
        "Asian/Asian British": "#72B7B2",
        "Mixed/multiple ethnic groups": "#F58518",
        "Hispanic/Latino": "#54A24B",
        "Other": "#302D2D"
    }

    fig = px.pie(
        counts.sort_values("Count", ascending=False),
        names="Ethnicity",
        values="Count",
        hole=0.45,                      # donut thickness
        color="Ethnicity",
        color_discrete_map=palette
    )

    # Neat labels, small slices pulled slightly
    fig.update_traces(
        texttemplate="%{label}<br>%{percent:.1%}",
        textposition="inside",
        insidetextorientation="radial",
        pull=[0.06 if p < 0.06 else 0 for p in counts["Percent"]]
    )

    # Smaller, cleaner layout
    fig.update_layout(
        title="Maternal Ethnicity Distribution (%)",
        height=380,                     # ↓ shorter
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False                # legend off (labels are inside)
    )

    st.plotly_chart(fig, use_container_width=True)




bmi_col = "prepreg BMI"
bmi_cat_col = "BMI_categories"


# --- Layout: two main columns ---
left_col, right_col = st.columns([0.6, 1], gap="large")

with left_col:
    # st.markdown("### Maternal BMI Overview")

    # KPIs stacked vertically
    st.metric("Median Pre-preg BMI", f"{df[bmi_col].median(skipna=True):.1f}")

    q1, q3 = df[bmi_col].quantile([0.25, 0.75])
    st.metric("IQR (Q1–Q3)", f"{q1:.1f}–{q3:.1f}")

    if bmi_cat_col in df.columns:
        ow_obese = df[bmi_cat_col].astype(str).str.lower().isin({"overweight", "obese"})
        pct = ow_obese.mean() * 100
    else:
        pct = (df[bmi_col] >= 25).mean() * 100
    st.metric("Overweight/Obese (%)", f"{pct:.1f}%")

    st.metric("Missing BMI", int(df[bmi_cat_col].isna().sum()))

with right_col:
    if bmi_cat_col in df.columns:
        cat_counts = (
            df[bmi_cat_col].astype(str)
            .value_counts(dropna=True)
            .rename_axis("BMI Category")
            .reset_index(name="Count")
        )

        bmi_palette = {
            "Underweight": "#E2ED91",
            "Normal weight": "#9ED090",
            "Overweight": "#F58518",
            "Obese": "#E45756",
            "nan": "#AFB1B3"
        }

        fig_pie = px.pie(
            cat_counts,
            names="BMI Category",
            values="Count",
            hole=0.45,
            color="BMI Category",
            color_discrete_map=bmi_palette,
            title="BMI Category Share (%)"
        )
        fig_pie.update_traces(
            textposition="inside", 
            textinfo="percent+label"
        )
        fig_pie.update_layout(
            height=360,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)




st.divider()




# Ensure numeric
df["Gravida"] = pd.to_numeric(df["Gravida"], errors="coerce")
df["Parity"]  = pd.to_numeric(df["Parity"], errors="coerce")

# Make long-form DataFrame (stacked structure)
grav_counts = (
    df["Gravida"].value_counts().sort_index().rename_axis("Value").reset_index(name="Count")
)
grav_counts["Variable"] = "Gravida"

par_counts = (
    df["Parity"].value_counts().sort_index().rename_axis("Value").reset_index(name="Count")
)
par_counts["Variable"] = "Parity"

# Combine into one table
combo = pd.concat([grav_counts, par_counts], ignore_index=True)

# Optional: convert to percent within variable
show_percent = st.toggle("Show percent instead of counts", value=False)
if show_percent:
    combo["Percent"] = combo.groupby("Variable")["Count"].transform(lambda x: 100 * x / x.sum())
    y = "Percent"
    y_label = "% within variable"
else:
    y = "Count"
    y_label = "Count"

# Plot stacked bar
fig = px.bar(
    combo,
    x="Value",
    y=y,
    color="Variable",
    barmode="group",  # you can change to "stack" for stacked bars
    text=y,
    color_discrete_map={"Gravida": "#B7A5A5", "Parity": "#917D7D"},
    labels={"Value": "Count (0–6)", y: y_label},
    title="Gravida vs. Parity Distribution",
)

# Format aesthetics
fig.update_traces(
    texttemplate="%{text:.1f}%" if show_percent else "%{text}",
    textposition="outside",
)
fig.update_layout(
    xaxis_title="Value (0–6)",
    yaxis_title=y_label,
    height=450,
    margin=dict(l=10, r=10, t=50, b=10),
    bargap=0.15,
)

st.plotly_chart(fig, use_container_width=True)



























# delivery method bar chart side by side of bar chart of primiparity OR KPI Cards

# --- Delivery Method x Primiparity: tidy, consistent visuals + stats ---


# 0) Clean columns (safe reassign)
df = df.copy()
df["Delivery Method"] = df["Delivery Method"].astype(str).str.strip()
df["Primiparity"] = (
    df["Primiparity"]
      .astype(str)
      .str.strip()
      .str.lower()
      .replace({"yes": "Yes", "no": "No", "nan": "Missing", "none": "Missing"})
)
# Ensure age numeric
df["mat_age"] = pd.to_numeric(df["mat_age"], errors="coerce")

# 1) Single category order used everywhere (spellings matter!)
delivery_order_master = [
    "Forcepts",  # <- if your data actually uses "Forcepts"; otherwise change to "Forceps"
    "Elective Caesarean Section",
    "Emergency Caesarean Section",
    "Spontaneous Vaginal Delivery (SVD)",
    "Ventouse",
]
# Keep only categories that exist in the current data
present_order = [d for d in delivery_order_master if d in df["Delivery Method"].unique().tolist()]


# 3) Maternal age by delivery method (box + points)

fig_age = px.box(
    df,
    x="Delivery Method",
    y="mat_age",
    color="Delivery Method",
    points="all",  # overlay raw points
    category_orders={"Delivery Method": present_order},
    title="Maternal Age by Delivery Method",
    color_discrete_sequence=px.colors.qualitative.Safe,
)
fig_age.update_layout(
    xaxis_title="Delivery Method",
    yaxis_title="Maternal Age (years)",
    showlegend=False,
    xaxis_tickangle=-25,
    height=450,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_age, use_container_width=True)

# --- Statistical test: Maternal age vs. Delivery Method ---
data = df[["mat_age", "Delivery Method"]].dropna()

if data["Delivery Method"].nunique() < 2:
    st.info("Not enough groups in 'Delivery Method' to run the test.")
else:
    # Normality check per group
    normal = True
    for _, g in data.groupby("Delivery Method"):
        vals = g["mat_age"].values
        if len(vals) >= 3 and len(vals) <= 5000:
            if shapiro(vals).pvalue <= 0.05:
                normal = False
                break

    if normal:
        # One-way ANOVA
        model = smf.ols("mat_age ~ C(`Delivery Method`)", data=data).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        F = float(anova["F"][0])
        p = float(anova["PR(>F)"][0])
        st.markdown(
            f"**ANOVA:** F = {F:.2f}, p = {p:.4g}  "
            f"*(maternal age ~ delivery method; groups approximately normal)*"
        )
    else:
        # Kruskal–Wallis (non-parametric)
        groups = [g["mat_age"].values for _, g in data.groupby("Delivery Method")]
        H, p = kruskal(*groups)
        st.markdown(
            f"**Kruskal–Wallis test:** H = {H:.2f}, p = {p:.4g}  "
            f"*(maternal age ~ delivery method; non-normal groups)*"
        )

# 6) Interpretation (your narrative)
with st.expander("Delivery Method – Interpretation", expanded=False):
    st.markdown("""
    - Mothers who had **elective C-sections** were significantly older than those with **spontaneous vaginal deliveries** 

    """)


# 4) Delivery method by primiparity (grouped counts)
count_table = (
    df.groupby(["Delivery Method", "Primiparity"])
        .size()
        .reset_index(name="Count")
)
fig_prim = px.bar(
    count_table,
    x="Delivery Method",
    y="Count",
    color="Primiparity",
    barmode="group",
    text="Count",
    category_orders={"Delivery Method": present_order},
    color_discrete_map={"Yes": "#917D7D", "No": "#A1A8AF", "Missing": "#9E9E9E"},
    title="Delivery Method by Primiparity",
)
fig_prim.update_traces(textposition="outside")
fig_prim.update_layout(
    xaxis_title="Delivery Method",
    yaxis_title="Count",
    xaxis_tickangle=-25,
    height=450,
    margin=dict(l=10, r=10, t=50, b=10),
    bargap=0.25,
)
st.plotly_chart(fig_prim, use_container_width=True)


# 5) Chi-square test + Cramer's V (uses same cleaned columns)
ct = pd.crosstab(
    df["Delivery Method"],
    df["Primiparity"]
)

if ct.shape[0] >= 2 and ct.shape[1] >= 2 and ct.values.sum() > 0:
    chi2, p, dof, expected = chi2_contingency(ct.values)
    n = ct.values.sum()
    r, c = ct.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1)))
    st.markdown(
        f"**Chi-square test:** χ²({dof}) = {chi2:.2f}, p = {p:.4g}; "
        f"**Cramer's V** = {cramers_v:.2f}"
    )
else:
    st.info("Not enough variation to run chi-square (need ≥2 categories in both variables).")


# 6) Interpretation (your narrative)
with st.expander("Delivery Method – Interpretation", expanded=False):
    st.markdown("""
    - **Emergency C-section** and **Forceps** deliveries are much more common among *primiparous* mothers.  
    - **Elective C-sections** and **Spontaneous vaginal deliveries** are more common among *multiparous* mothers.
    """)

st.divider()





############################################################
## Breastfeeding and Skin-to-Skin KPIs
bf_col = "breastfeeding_status_at_discharge"
ss_col = "skinskin_firsthour"

# Work on a copy
d = df.copy()

# --- Normalize columns to strings ---
d[bf_col] = d[bf_col].astype(str).str.strip()
d[ss_col] = d[ss_col].astype(str).str.strip().str.lower()

# --- Exclusive breastfeeding mask (robust to label variants) ---
# Matches phrases like "Exclusive breastfeeding", "Exclusive breast feeding", "EBF"
# and avoids "exclusive formula".
bf_norm = d[bf_col].str.lower()
exclusive_bf_mask = (
    bf_norm.str.contains("exclusive") &
    bf_norm.str.contains("breast") &
    ~bf_norm.str.contains("formula")
)

# Denominator: non-missing breastfeeding status
bf_valid = bf_norm.replace({"nan": pd.NA, "": pd.NA}).notna()
ebf_num = int((exclusive_bf_mask & bf_valid).sum())
ebf_den = int(bf_valid.sum())
ebf_pct = (ebf_num / ebf_den * 100) if ebf_den else 0.0

# --- Skin-to-skin (Yes) mask ---
ss_yes = ss_col  # alias
yes_mask = d[ss_yes].isin({"yes", "y", "true", "1"})
# Denominator: non-missing skin-to-skin
ss_valid = d[ss_col].replace({"nan": pd.NA, "": pd.NA, "none": pd.NA}).notna()
ss_num = int((yes_mask & ss_valid).sum())
ss_den = int(ss_valid.sum())
ss_pct = (ss_num / ss_den * 100) if ss_den else 0.0

# --- KPI cards side-by-side ---
k1, k2 = st.columns(2, gap="large")

with k1:
    st.metric("Exclusive breastfeeding at discharge", f"{ebf_pct:.1f}%")
    st.caption(f"{ebf_num} of {ebf_den} with recorded status")

with k2:
    st.metric("Skin-to-skin in first hour", f"{ss_pct:.1f}%")
    st.caption(f"{ss_num} of {ss_den} with recorded status")


