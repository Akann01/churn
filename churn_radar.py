import os

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide")

st.title("Customer Churn Risk Dashboard")
st.caption("Select a row in the customer table to inspect that customer's profile.")

top_at_risk = pd.read_csv("top_20_at_risk_customers.csv", index_col=0)
top_at_risk.index.name = "Customer ID"

# Load model + scaler so we can re-score "what-if" scenarios live
model = joblib.load("churn_model.pkl")          # Random Forest, trained on unscaled features

# Retention emails written by retention_agent.py (optional: the dashboard still works without it)
campaign = None
if os.path.exists("retention_campaign.csv"):
    campaign = pd.read_csv("retention_campaign.csv")
    campaign["customer_id"] = campaign["customer_id"].astype(str)

st.sidebar.header("Filters")
risk_threshold = st.sidebar.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.05)
sort_order = st.sidebar.radio("Sort by", ["Highest risk first", "Lowest risk first"])

filtered = top_at_risk[top_at_risk["Churn_Probability"] >= risk_threshold].sort_values(
    "Churn_Probability", ascending=(sort_order == "Lowest risk first")
)
st.sidebar.metric("Customers matching filter", len(filtered))

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Customers Analyzed", len(top_at_risk))
metric_col2.metric("Average Churn Probability", f"{top_at_risk['Churn_Probability'].mean():.1%}")
metric_col3.metric("High Risk (>70%)", int((top_at_risk["Churn_Probability"] > 0.7).sum()))

if filtered.empty:
    st.warning("No customers match the current filter. Lower the minimum churn probability to continue.")
    st.stop()

table_data = filtered[["Actual_Churn", "Churn_Probability"]].reset_index()
table_data["Customer ID"] = table_data["Customer ID"].astype(str)
table_data["Risk"] = table_data["Churn_Probability"].map(lambda value: f"{value:.1%}")
table_data["Actual Churn"] = table_data["Actual_Churn"].map({0: "No", 1: "Yes"})

st.subheader("At-Risk Customers")
st.caption("Click any row to update the customer details below.")
selection = st.dataframe(
    table_data[["Customer ID", "Risk", "Actual Churn"]],
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    key="customer_table",
)

customer_ids = table_data["Customer ID"].tolist()
selected_id = st.selectbox("Or choose a customer", customer_ids, key="customer_selector")
selected_rows = selection.selection.rows
if selected_rows:
    selected_id = customer_ids[selected_rows[0]]

selected_customer = filtered.loc[filtered.index.astype(str) == selected_id].iloc[0]
st.divider()
st.subheader(f"Customer {selected_id}")

detail_col1, detail_col2, detail_col3 = st.columns(3)
detail_col1.metric("Churn Probability", f"{selected_customer['Churn_Probability']:.1%}")
detail_col2.metric("Actual Churn", "Yes" if selected_customer["Actual_Churn"] else "No")
detail_col3.metric("Risk Tier", "High" if selected_customer["Churn_Probability"] >= 0.7 else "Watch")

# ---- SHAP "why" chart ----
st.divider()
st.subheader("Why is this customer at risk?")

shap_reasons = pd.read_csv("shap_reasons.csv")
customer_reasons = shap_reasons[
    shap_reasons["customer_id"].astype(str) == selected_id
].sort_values("shap_value")

fig_shap = px.bar(
    customer_reasons,
    x="shap_value",
    y="feature",
    orientation="h",
    color="shap_value",
    color_continuous_scale=["blue", "red"],
    title=f"Top factors driving churn risk for customer {selected_id}",
)
fig_shap.update_layout(yaxis_title=None, xaxis_title="Impact on churn probability", coloraxis_showscale=False)
st.plotly_chart(fig_shap, use_container_width=True)
st.caption("Red bars push churn risk up. Blue bars push it down.")

# ---- Suggested outreach ----
st.divider()
st.subheader("Suggested outreach")

if campaign is None:
    st.info("No outreach drafts yet. Run `python retention_agent.py` to generate them.")
else:
    match = campaign[campaign["customer_id"] == selected_id]
    if match.empty:
        st.info("No outreach draft was generated for this customer.")
    else:
        row = match.iloc[0]
        oc1, oc2 = st.columns([3, 1])
        oc1.markdown(f"**Action:** {row['action']}")
        if row["source"] == "llm":
            oc2.success("Written by the model, passed validation")
        else:
            oc2.warning("Generic fallback email")
        st.caption(f"Why this action: {str(row['reason']).replace('{first_name}', 'the customer')}")
        st.markdown(f"**Subject:** {row['subject']}")
        st.text(row["message"])
        st.caption("{first_name} is filled in when the campaign is sent. Drafts are for review, not sent automatically.")

    n_llm = int((campaign["source"] == "llm").sum())
    st.caption(f"Campaign overview: {n_llm} of {len(campaign)} emails written by the model, {len(campaign) - n_llm} fallback.")

# ---- What-if sliders ----
st.divider()
st.subheader("What-if: adjust key drivers")

top_features = customer_reasons.reindex(
    customer_reasons["shap_value"].abs().sort_values(ascending=False).index
)["feature"].head(3).tolist()

feature_cols = [c for c in filtered.columns if c not in ["Actual_Churn", "Churn_Probability"]]
whatif_row = filtered.loc[filtered.index.astype(str) == selected_id, feature_cols].copy()

st.caption("Drag a slider to see how changing that factor would move this customer's churn probability.")
slider_cols = st.columns(len(top_features))
for col, feat in zip(slider_cols, top_features):
    current_val = float(whatif_row[feat].iloc[0])
    col_min = float(filtered[feat].min())
    col_max = float(filtered[feat].max())
    new_val = col.slider(feat, col_min, col_max, current_val)
    whatif_row[feat] = new_val

# Re-score with the model
X_whatif = whatif_row[feature_cols]
# The Random Forest was trained on unscaled data, so it must not be scaled here
new_prob = model.predict_proba(X_whatif)[0, 1]

wc1, wc2, wc3 = st.columns(3)
wc1.metric("Original Probability", f"{selected_customer['Churn_Probability']:.1%}")
wc2.metric("Adjusted Probability", f"{new_prob:.1%}",
           delta=f"{(new_prob - selected_customer['Churn_Probability']) * 100:.1f} pts")
wc3.metric("Risk Tier (adjusted)", "High" if new_prob >= 0.7 else "Watch")

# ---- Overview chart ----
st.divider()
chart_data = table_data.sort_values("Churn_Probability", ascending=True)
fig = px.bar(
    chart_data,
    x="Churn_Probability",
    y="Customer ID",
    orientation="h",
    color="Churn_Probability",
    color_continuous_scale="Reds",
    title="Churn Probability by Customer",
)
fig.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_title="Probability")
fig.update_yaxes(type="category")  # customer IDs are labels, not numbers
st.plotly_chart(fig, use_container_width=True)