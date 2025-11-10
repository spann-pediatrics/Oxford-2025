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



st.title("Oxford Metadata and Sample Overview")
st.set_page_config(page_title="Oxford Dashboard", layout="wide", initial_sidebar_state="expanded")


# load metadata overview
df = pd.read_excel('/Users/kspann/Desktop/Oxford/Cleaned Data/Metadata/metadata_overview_oxford.xlsx')
hmo = pd.read_excel('Cleaned Data/HMO/hmo_data_oxford.xlsx')

# Make sure PP_day_num is numeric
hmo["PP_day_num"] = pd.to_numeric(hmo["PP_day_num"], errors="coerce")


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





st.markdown("## Maternal Demographic Overview")
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




# Normalize category labels (case/spacing tolerant)
cat = (
    df["BMI_categories"]
      .astype(str).str.strip().str.lower()
      .replace({"nan": np.nan})
)

# Define buckets
NORMAL_SET = {"normal", "normal weight", "normalweight"}
OW_OBESE_SET = {
    "overweight", "obese", "obesity",
    "class i obesity", "class ii obesity", "class iii obesity",
    "obesity class i", "obesity class ii", "obesity class iii"
}

# Denominator = rows with any recognized category
valid_mask = cat.notna()
denom = int(valid_mask.sum())

normal_pct   = (cat.isin(NORMAL_SET).sum()   / denom * 100) if denom else np.nan
ow_obese_pct = (cat.isin(OW_OBESE_SET).sum() / denom * 100) if denom else np.nan

# KPI cards
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Median Pre-preg BMI", f"{df['prepreg BMI'].median(skipna=True):.1f}")

with c2:
    st.metric("Normal weight (%)", f"{normal_pct:.1f}%" if pd.notna(normal_pct) else "N/A")

with c3:
    st.metric("Overweight / Obese (%)", f"{ow_obese_pct:.1f}%" if pd.notna(ow_obese_pct) else "N/A")


st.divider()



st.markdown("## Baseline Maternal Comorbidities")


# --- Define your columns ---
comorb_cols = [
    "baseline_comorbidities_685",
    "Iron_baseline",
    "Sertraline_baseline",
    "Levothyroxine_baseline"
]

# --- Clean and calculate percentages ---
df_comorb = df[comorb_cols].copy()  
df_comorb = df_comorb.apply(lambda s: s.astype(str).str.strip().str.title())

summary = []
for col in comorb_cols:
    yes_pct = (df_comorb[col] == "Yes").mean(skipna=True) * 100
    yes_n = (df_comorb[col] == "Yes").sum()
    total_n = df_comorb[col].notna().sum()
    summary.append({
    "Comorbidity": col.replace("_baseline", "").replace("baseline_comorbidities_685", "Baseline Comorbidity"),
    "Percent": yes_pct,
    "N (Yes)": yes_n,
    "N (Total)": total_n
})


summary_df = pd.DataFrame(summary).sort_values("Percent", ascending=True)

# --- Horizontal bar chart ---
fig = px.bar(
    summary_df,
    x="Percent",
    y="Comorbidity",
    orientation="h",
    text="Percent",
    color="Comorbidity",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    labels={"Percent": "% of participants", "Comorbidity": "Baseline Condition"},
    title="Baseline Maternal Comorbidities / Supplement Use",
)

fig.update_traces(
    texttemplate="%{x:.1f}%",
    textposition="outside"
)
fig.update_layout(
    height=450,
    showlegend=False,
    xaxis_range=[0, 100],
    margin=dict(l=10, r=10, t=60, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# --- Optional KPI cards for quick overview ---
c1, c2, c3, c4 = st.columns(4)
for i, col in enumerate(summary_df.itertuples(), 1):
    with [c1, c2, c3, c4][i - 1]:
        st.metric(
            label=col.Comorbidity,
            value=f"{col.Percent:.1f}%"
        )

st.divider()


















# delivery method bar chart side by side of bar chart of primiparity OR KPI Cards

# --- Delivery Method x Primiparity: tidy, consistent visuals + stats ---
st.markdown("## Birth Demographics")


# ---------- Render KPI cards ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="C-Section Births",
        value=(
            f"{(df['Csection'] == 'Yes').sum()/ len(df) *100:.0f}%"
            if 'Csection' in df.columns else "N/A"
        )
    )

with c2:
    st.metric(
        label="Epidural Births",
        value=(
            f"{(df['Epidural'] == 'Yes').sum()/ len(df) *100:.0f}%"
            if 'Epidural' in df.columns else "N/A"
        )
    )

with c3:
    st.metric(
        label="Oxytoxin Used At Birth",
        value=(
            f"{(df['Oxytocin_labour'] == 'Yes').sum()/ len(df) *100:.0f}%"
            if 'Oxytocin_labour' in df.columns else "N/A"
        )
    )





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




###########STACKED BAR CHART FOR GRAVIDA/PARITY###############

# # Ensure numeric
# df["Gravida"] = pd.to_numeric(df["Gravida"], errors="coerce")
# df["Parity"]  = pd.to_numeric(df["Parity"], errors="coerce")

# # Make long-form DataFrame (stacked structure)
# grav_counts = (
#     df["Gravida"].value_counts().sort_index().rename_axis("Value").reset_index(name="Count")
# )
# grav_counts["Variable"] = "Gravida"

# par_counts = (
#     df["Parity"].value_counts().sort_index().rename_axis("Value").reset_index(name="Count")
# )
# par_counts["Variable"] = "Parity"

# # Combine into one table
# combo = pd.concat([grav_counts, par_counts], ignore_index=True)

# # Optional: percent toggle
# show_percent = st.toggle("Show percent instead of counts", value=False)
# if show_percent:
#     combo["Percent"] = combo.groupby("Variable")["Count"].transform(lambda x: 100 * x / x.sum())
#     y = "Percent"
#     y_label = "% within variable"
# else:
#     y = "Count"
#     y_label = "Count"

# # Plot grouped bar
# fig = px.bar(
#     combo,
#     x="Value",
#     y=y,  # <--- you need this for the y-axis to exist
#     color="Variable",
#     barmode="group",
#     color_discrete_map={"Gravida": "#B7A5A5", "Parity": "#917D7D"},
#     labels={"Value": "Value (0–6)", y: y_label},
#     title="Gravida vs. Parity Distribution",
# )

# # Format aesthetics
# fig.update_traces(
#     texttemplate="%{y:.1f}" if show_percent else "%{y}",
#     textposition="outside",
# )
# fig.update_layout(
#     xaxis_title="Value (0–6)",
#     yaxis_title=y_label,  # ✅ put this back
#     height=450,
#     margin=dict(l=10, r=10, t=50, b=10),
#     bargap=0.15,
# )

# st.plotly_chart(fig, use_container_width=True)



# ---------- Clean & prep ----------

# ---------- KPI 2: Avg Gravida:Parity ratio (only P>0) ----------
mask_p_gt0 = df["Parity"].fillna(0) > 0
ratio = (df.loc[mask_p_gt0, "Gravida"].astype(float) / df.loc[mask_p_gt0, "Parity"].astype(float))
ratio_mean   = float(ratio.mean()) if len(ratio) else np.nan
ratio_median = float(ratio.median()) if len(ratio) else np.nan
ratio_n      = int(mask_p_gt0.sum())

# ---------- KPI 3 (clinically useful): % Nulliparous (Parity == 0) ----------
nulliparous = (df["Parity"] == 0)
nulli_pct = float(nulliparous.mean(skipna=True) * 100) if nulliparous.notna().any() else np.nan
nulli_n   = int(nulliparous.sum(skipna=True))

# ---------- Render KPI cards ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="Primiparous (%) (Parity = 0)",
        value=(
            f"{(df['Primiparity'] == 'Yes').sum() / len(df) * 100:.1f}%"
            if 'Primiparity' in df.columns else "N/A"
        )
    )

with c2:
    st.metric(
        label="Multiparous (%) (Parity > 0)",
        value=(
            f"{(df['Primiparity'] == 'No').sum() / len(df) * 100:.1f}%"
            if 'Primiparity' in df.columns else "N/A"
        )
    )


# Compute total averages (including zeros)
grav_mean = df["Gravida"].mean(skipna=True)
par_mean  = df["Parity"].mean(skipna=True)

if pd.notna(grav_mean) and pd.notna(par_mean):
    # Format as "Gravida : Parity" → pregnancies per delivery
    ratio_text = f"{grav_mean:.0f} : {par_mean:.0f}"
else:
    ratio_text = "N/A"


with c3:
    st.metric(
        label="Avg Gravida : Parity",
        value=ratio_text
    )

# Helpful footnotes so the KPIs are unambiguous
st.caption(
    "Gravida:Parity ratio computed only where Parity>0 to avoid division by zero."
)



















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





# --- Convert to numeric ---
df["Gest_Age_Baseline"] = pd.to_numeric(df["Gest_Age_Baseline"], errors="coerce")
df["Gest_Age_Birth"]    = pd.to_numeric(df["Gest_Age_Birth"], errors="coerce")

# --- Compute summary stats (in weeks) ---
# Note: gestational age often stored in days → divide by 7
baseline_weeks = df["Gest_Age_Baseline"] / 7
birth_weeks    = df["Gest_Age_Birth"] / 7

baseline_min = baseline_weeks.min(skipna=True)
baseline_max = baseline_weeks.max(skipna=True)
baseline_mean = baseline_weeks.mean(skipna=True)

birth_min = birth_weeks.min(skipna=True)
birth_max = birth_weeks.max(skipna=True)
birth_mean = birth_weeks.mean(skipna=True)

# --- Display KPI cards ---
c1, c2 = st.columns(2)

with c1:
    st.metric(
        label="Gestational Age at Enrollment (Range)",
        value=f"{baseline_min:.1f} – {baseline_max:.1f} wks"
    )

with c2:
    st.metric(
        label="Average Gestational Age at Enrollment",
        value=f"{baseline_mean:.1f} wks"
    )



st.divider()
st.markdown("## Infant Demographics")

# --- KPI Cards ---
c1, c2, c3, c4 = st.columns(4)

sga_yes = (df["SGA"] == "Yes").sum()
sga_total = len(df)
sga_pct = sga_yes / sga_total * 100 if sga_total > 0 else 0



neo_yes = (df["neonatalobs"] == "Yes").sum()
neo_total = df["neonatalobs"].notna().sum()
neo_pct = neo_yes / neo_total * 100 if neo_total > 0 else 0


PROM_yes = (df["PROM"] == "Yes").sum()
PROM_total = len(df)
PROM_pct = PROM_yes / PROM_total * 100

sep_yes = (df["Infant_sepsis"] == "Yes").sum()
PROM_total = len(df)
sep_pct = sep_yes / PROM_total * 100

with c1:
    st.metric(
        label="SGA (Small for Gestational Age)",
        value=f"{sga_pct:.1f}%"
    )


with c2:
    st.metric(
        label="Neonatal Observation / Complication (neonatalobs)",
        value=f"{neo_pct:.1f}%"
    )


with c3:
    st.metric(
        label="PROM (Pre-labour Rupture of Membranes)",
        value=f"{PROM_pct:.1f}%"
    )


with c4:
    st.metric(
        label="Infant Sepsis %",
        value = f'{sep_pct:.1f}%'
    )





# --- Histogram: Gestational Age at Birth ---
st.markdown("### Gestational Age at Birth")

fig = px.histogram(
    x=birth_weeks,
    nbins=20,
    labels={"x": "Gestational Age at Birth (weeks)", "count": "Count"},
    title="Distribution of Gestational Age at Birth",
)
fig.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=60, b=10),
    bargap=0.1,
    showlegend=False
)
fig.update_traces(marker_color="#7D91B1", hovertemplate="%{x:.1f} weeks<br>Count=%{y}")
st.plotly_chart(fig, use_container_width=True)















# --- Compute counts for each category ---
season_counts = df["Delivery_Season"].value_counts(dropna=False).reset_index()
season_counts.columns = ["Season", "Count"]
season_counts["Percent"] = 100 * season_counts["Count"] / season_counts["Count"].sum()

sex_counts = df["infantsex"].value_counts(dropna=False).reset_index()
sex_counts.columns = ["Sex", "Count"]
sex_counts["Percent"] = 100 * sex_counts["Count"] / sex_counts["Count"].sum()


# --- Layout: side-by-side columns ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Delivery Season")
    fig1 = px.pie(
        season_counts,
        names="Season",
        values="Count",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.3,
    )
    fig1.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count=%{value} (%{percent})<extra></extra>",
    )
    fig1.update_layout(
        showlegend=True,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("Infant Sex")
    fig2 = px.pie(
        sex_counts,
        names="Sex",
        values="Count",
        color_discrete_sequence=px.colors.qualitative.Pastel1,
        hole=0.3,
    )
    fig2.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count=%{value} (%{percent})<extra></extra>",
    )
    fig2.update_layout(
        showlegend=True,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)


































st.divider()
st.markdown("## Breastfeeding Practices & Milk Sample Demographics")

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





# --- Count days ---
counts = (
    hmo["PP_day_num"]
    .value_counts(dropna=False)
    .sort_index()
    .reset_index()
)
counts.columns = ["PP_day_num", "Count"]

# --- Categorize colostrum vs mature ---
counts["MilkType"] = counts["PP_day_num"].apply(
    lambda x: "Colostrum" if x <= 5 else "Mature"
)

# --- Bar chart ---
fig = px.bar(
    counts,
    x="PP_day_num",
    y="Count",
    color="MilkType",
    text="Count",
    color_discrete_map={
        "Colostrum": "#7C93D8",  # soft blue
        "Mature": "#F2A65A"      # warm orange
    },
    labels={"PP_day_num": "Postpartum Day", "Count": "Count", "MilkType": "Milk Type"},
    title="Milk Sample Collection Distribution",
)

fig.update_traces(
    textposition="outside",
    hovertemplate="Day %{x}<br>%{customdata[0]} Milk<br>Count=%{y}<extra></extra>",
    customdata=counts[["MilkType"]]
)

fig.update_layout(
    height=450,
    bargap=0.15,
    xaxis=dict(
        tickmode="array",
        tickvals=counts["PP_day_num"],
        ticktext=[f"{int(x)}" for x in counts["PP_day_num"]],
        title="PP Day Number (1–5 = Colostrum, 6 = Mature)"
    ),
    legend_title_text="Milk Type",
    margin=dict(l=10, r=10, t=50, b=10)
)

st.plotly_chart(fig, use_container_width=True)




#### KPI cards showing how many participants provided day 1-6 samples & # participants who provided >3 time points

ID_COL   = "Participant ID"   # <-- change if your id col differs
DAY_COL  = "PP_day_num"
HMO_COLS = [c for c in hmo.columns if "µg/mL" in c]  # HMO concentration columns

# --- Stage definitions ---
COLOSTRUM_DAYS = [1,2,3,4,5]
MATURE_DAY     = 6

# --- Per-stage participant means ---
colostrum = (
    hmo[hmo[DAY_COL].isin(COLOSTRUM_DAYS)]
      .groupby(ID_COL)[HMO_COLS].mean()
      .add_suffix("_col")
)

mature = (
    hmo[hmo[DAY_COL] == MATURE_DAY]
      .groupby(ID_COL)[HMO_COLS].mean()
      .add_suffix("_mat")
)

# Only participants who have at least one stage value will appear; we want those with BOTH
paired = colostrum.join(mature, how="inner")

# --- Overall paired participants KPI (across any HMO) ---
# A participant is "paired" if they have BOTH a colostrum AND a mature value for at least one HMO
has_any_pair = pd.Series(False, index=paired.index)
for h in HMO_COLS:
    a = paired.get(f"{h}_col")
    b = paired.get(f"{h}_mat")
    if a is None or b is None:
        continue
    has_any_pair = has_any_pair | (a.notna() & b.notna())

n_paired_participants = int(has_any_pair.sum())
n_total_participants  = int(hmo[ID_COL].nunique())
pct = (100 * n_paired_participants / n_total_participants) if n_total_participants else 0.0

# --- KPI cards side-by-side ---
k1, k2 = st.columns(2, gap="large")

with k1: 
    st.metric(
        label="Participants with paired HMO samples (colostrum 1–5 & mature 6)",
        value=str(n_paired_participants)
    )


# --- KPI: participants with >=3 distinct time points among days 1..6 ---
# clean types and restrict to valid days
hmo[DAY_COL] = pd.to_numeric(hmo[DAY_COL], errors="coerce")
valid = (
    hmo[[ID_COL, DAY_COL]]
    .dropna(subset=[ID_COL, DAY_COL])
)
valid = valid[valid[DAY_COL].between(1, 6)]

# one row per (ID, day), then count unique days per participant
per_id_day_counts = (
    valid.drop_duplicates([ID_COL, DAY_COL])
         .groupby(ID_COL)[DAY_COL]
         .nunique()
)

n_3plus = int((per_id_day_counts >= 3).sum())
n_total_participants = int(hmo[ID_COL].nunique())
pct_3plus = (100 * n_3plus / n_total_participants) if n_total_participants else 0.0

with k2:
    st.metric(
        label="Participants with ≥3 milk collection days (1–6)",
        value=str(n_3plus)
    )



# --- Cupsize bar chart (counts from hmo) ---
cup_counts = (
    df["Cupsize"]
    .fillna("Missing")
    .astype(str)
    .str.strip()
    .value_counts()
    .reset_index()
)
cup_counts.columns = ["Cupsize", "Count"]

fig = px.bar(
    cup_counts.sort_values("Count", ascending=False),
    x="Cupsize",
    y="Count",
    text="Count",
    color="Cupsize",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Milk Sample Cup Sizes",
    labels={"Cupsize": "Cup Size", "Count": "Count"},
)

fig.update_traces(textposition="outside")
fig.update_layout(
    height=400,
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis_tickangle=-25,
)

st.plotly_chart(fig, use_container_width=True)