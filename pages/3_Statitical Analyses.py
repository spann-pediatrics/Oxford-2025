import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.header("HMO Composition Stats")

# ========== SECRETOR x DAY: MixedLM across HMOs (µg/mL) ==========
st.subheader("Secretor × Day effect on HMO concentrations (µg/mL)")

df = pd.read_excel("Cleaned Data/Merged/merged_hmo_meta.xlsx")

# Small helper so we don't crash on odd characters in column names
# --- Helper: get HMO µg/mL columns, excluding any that contain "SUM" ---
def hmo_ugmL_cols(df):
    return [
        c for c in df.columns
        if "(µg/mL)" in c and "SUM" not in c.upper()
    ]


@st.cache_data(show_spinner=False)
def fit_secretor_day_interactions(df_in: pd.DataFrame):
    cols = hmo_ugmL_cols(df_in)
    rows = []

    # make sure types are sane
    d = df_in.copy()
    d["PP_day_num"] = pd.to_numeric(d["PP_day_num"], errors="coerce")
    # treat secretor_status as categorical (expects 0/1 or strings)
    d["secretor_status"] = d["secretor_status"].astype("category")

    for hmo in cols:
        sub = d[[hmo, "PP_day_num", "secretor_status", "Participant ID"]].dropna()
        if sub.empty or sub["Participant ID"].nunique() < 2:
            continue
        try:
            # Mixed model with random intercept per mom
            formula = f'Q("{hmo}") ~ PP_day_num * C(secretor_status)'
            res = smf.mixedlm(formula, data=sub, groups=sub["Participant ID"]).fit(method="lbfgs", reml=True)

            # grab key terms
            def grab(term_substr):
                # find first matching param/p-value name
                name = next((n for n in res.pvalues.index if term_substr in n), None)
                if name is None:
                    return np.nan, np.nan
                return res.params[name], res.pvalues[name]

            b_sec, p_sec = grab("C(secretor_status)")
            b_day, p_day = grab("PP_day_num")
            b_int, p_int = grab("PP_day_num:C(secretor_status)")

            rows.append({
                "HMO": hmo,
                "beta_secretor": b_sec, "p_secretor": p_sec,
                "beta_day": b_day,       "p_day": p_day,
                "beta_interaction": b_int, "p_interaction": p_int
            })
        except Exception as e:
            rows.append({"HMO": hmo, "error": str(e)})

    out = pd.DataFrame(rows)

    # FDR for each family of tests
    for pcol in ["p_secretor", "p_day", "p_interaction"]:
        mask = out[pcol].notna()
        if mask.sum() > 0:
            reject, qvals, _, _ = multipletests(out.loc[mask, pcol], method="fdr_bh")
            out.loc[mask, f"q_{pcol[2:]}"] = qvals
            out.loc[mask, f"sig_{pcol[2:]}"] = reject

    # nice ordering for the plot: strongest (by |beta_interaction|) first, or by q
    out = out.sort_values(["sig_interaction", "beta_interaction"], ascending=[False, True]).reset_index(drop=True)
    return out

results = fit_secretor_day_interactions(df)

# --- Plot (horizontal bar; color by FDR significance on interaction) ---
plot_df = results.dropna(subset=["beta_interaction"]).copy()
plot_df["sig_interaction"] = plot_df["sig_interaction"].fillna(False)

fig = px.bar(
    plot_df,
    x="beta_interaction",
    y="HMO",
    color="sig_interaction",
    orientation="h",
    title="Secretor × Milk Collection Day interaction effects across HMOs (µg/mL)",
    labels={"beta_interaction":"Interaction coefficient (β)", "HMO":"HMO"},
)
# zero line
fig.add_vline(x=0, line_dash="dash", opacity=0.5)

st.plotly_chart(fig, use_container_width=True)

# --- Table + filters ---
st.markdown("**Model results (per HMO):**")
only_sig = st.toggle("Show only FDR-significant interactions (q < 0.05)", value=True)
table_df = results.copy()
if only_sig:
    table_df = table_df[table_df["sig_interaction"] == True]

# choose neat columns for display
show_cols = [
    "HMO",
    "beta_interaction", "p_interaction", "q_interaction", "sig_interaction",
    "beta_secretor", "p_secretor", "q_secretor",
    "beta_day", "p_day", "q_day"
]
table_df = table_df[[c for c in show_cols if c in table_df.columns]].reset_index(drop=True)

st.dataframe(table_df, use_container_width=True)

# --- Download ---
csv = table_df.to_csv(index=False).encode("utf-8")
st.download_button("Download results (CSV)", data=csv, file_name="secretor_x_day_HMO_conc_results.csv", mime="text/csv")

# --- Brief interpretation helper (appears under the table) ---
with st.expander("How to read this"):
    st.write(
        "- **beta_interaction** (PP_day_num × secretor_status) is the difference in daily slope for secretors vs non-secretors. "
        "Negative → secretors decline faster; positive → increase faster.\n"
        "- **q_interaction** is the FDR-adjusted p-value across HMOs for the interaction term.\n"
        "- **beta_secretor** is the baseline (day 1) difference between groups; **beta_day** is the day slope for non-secretors."
    )







st.subheader("HMO trajectories across postpartum days (µg/mL)")

# helpers
def hmo_ugmL_cols_including_sum(df):
    return [c for c in df.columns if "(µg/mL)" in c]  # includes SUM

# data hygiene
df_plot = df.copy()
df_plot["PP_day_num"] = pd.to_numeric(df_plot["PP_day_num"], errors="coerce")
df_plot = df_plot[df_plot["PP_day_num"].between(1, 6, inclusive="both")]
df_plot["secretor_status"] = df_plot["secretor_status"].astype("category")

# UI: pick HMO
hmo_options = sorted(hmo_ugmL_cols_including_sum(df_plot))
default_hmo = next((c for c in hmo_options if c.startswith("2FL")), hmo_options[0])
selected_hmo = st.selectbox("Choose HMO (µg/mL)", hmo_options, index=hmo_options.index(default_hmo))

# UI: optional raw “spaghetti” overlay
show_individuals = st.toggle("Show individual moms (thin lines)", value=False)

# plot
fig, ax = plt.subplots(figsize=(6, 2.5))

# mean ± 95% CI by secretor group
sns.lineplot(
    data=df_plot,
    x="PP_day_num", y=selected_hmo,
    hue="secretor_status",
    estimator="mean", errorbar=("ci", 95),
    linewidth=2, ax=ax
)

# optional per-mom lines (light)
if show_individuals:
    # draw faint lines per participant, per secretor group
    for (sec, pid), sub in df_plot.groupby(["secretor_status", "Participant ID"]):
        sub = sub.sort_values("PP_day_num")
        ax.plot(sub["PP_day_num"], sub[selected_hmo], alpha=0.15, linewidth=0.8)

ax.set_title(f"{selected_hmo} by Postpartum Day, split by Secretor")
ax.set_xlabel("Postpartum Day")
ax.set_ylabel(selected_hmo)
ax.set_xlim(1, 6)
ax.legend(title="secretor_status", frameon=False)
st.pyplot(fig, clear_figure=True, use_container_width=False)


# also show the group means table + download
summary = (
    df_plot.groupby(["secretor_status", "PP_day_num"])[selected_hmo]
           .agg(["count", "mean", "std"])
           .reset_index()
           .sort_values(["secretor_status", "PP_day_num"])
)
st.markdown("**Group means (by secretor × day):**")
st.dataframe(summary, use_container_width=True)

st.download_button(
    "Download summary (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_hmo.replace(' ','_').replace('/','-')}_by_day_secretor_summary.csv",
    mime="text/csv"
)





