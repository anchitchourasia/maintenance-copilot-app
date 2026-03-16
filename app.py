import numpy as np
import plotly.graph_objects as go

# ---------- THEME ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #6d1b7b 0%, #b13f87 45%, #f06aa7 100%);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    .dash-card {
        background: rgba(133, 61, 143, 0.58);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        backdrop-filter: blur(6px);
        color: white;
        min-height: 92px;
    }

    .dash-title {
        font-size: 28px;
        font-weight: 700;
        color: white;
        margin-bottom: 0;
    }

    .mini-label {
        color: #f7d7ef;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .mini-value {
        color: white;
        font-size: 32px;
        font-weight: 800;
        line-height: 1.1;
    }

    .chart-card {
        background: rgba(133, 61, 143, 0.58);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 14px;
        padding: 10px 10px 0 10px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        margin-top: 10px;
    }

    .section-head {
        color: white;
        font-size: 15px;
        font-weight: 700;
        padding: 6px 8px 0 8px;
    }

    div[data-testid="stDataFrame"] {
        background: rgba(133, 61, 143, 0.50);
        border-radius: 12px;
        padding: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def card_html(label, value):
    return f"""
    <div class="dash-card">
        <div class="mini-label">{label}</div>
        <div class="mini-value">{value}</div>
    </div>
    """

def themed_fig(fig, title):
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color="white", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="white")
        )
    )
    fig.update_xaxes(showgrid=False, color="white")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", color="white")
    return fig

# ---------- FILTER BAR ----------
top1, top2 = st.columns([2.2, 3.8])

with top1:
    st.markdown('<p class="dash-title">Maintenance Dashboard</p>', unsafe_allow_html=True)

with top2:
    selected_view = st.segmented_control(
        "View",
        options=["Overview", "High Risk", "Medium Risk", "Low Risk"],
        default="Overview",
        label_visibility="collapsed"
    )

filtered_df = predictions_df.copy()

if selected_view != "Overview":
    filtered_df = filtered_df[filtered_df["risk_level"] == selected_view.upper()]

if product_search:
    filtered_df = filtered_df[
        filtered_df["product_id"].astype(str).str.contains(product_search, case=False, na=False)
    ]

if risk_filter:
    filtered_df = filtered_df[filtered_df["risk_level"].isin(risk_filter)]

# ---------- KPI VALUES ----------
total_machines = len(filtered_df)
high_risk_count = int((filtered_df["risk_level"] == "HIGH RISK").sum()) if not filtered_df.empty else 0
medium_risk_count = int((filtered_df["risk_level"] == "MEDIUM RISK").sum()) if not filtered_df.empty else 0
priority_actions = len(priority_df)

auc_value = "0.954"
if not kpis_df.empty:
    possible_cols = ["failure_rate", "avg_failure_rate", "actual_failure_rate", "prediction"]
    found_col = next((c for c in possible_cols if c in kpis_df.columns), None)
    observed_text = "Not mapped"
    if found_col:
        vals = pd.to_numeric(kpis_df[found_col], errors="coerce").dropna()
        if len(vals) > 0:
            observed_text = f"{vals.mean():.1%}"
else:
    observed_text = "Not available"

# ---------- KPI ROW ----------
k1, k2, k3, k4 = st.columns([2.1, 1.2, 1.2, 1.2])

with k1:
    st.markdown(card_html("Dashboard", "Predictive Maintenance"), unsafe_allow_html=True)
with k2:
    st.markdown(card_html("Total Machines", f"{total_machines:,}"), unsafe_allow_html=True)
with k3:
    st.markdown(card_html("High Risk", f"{high_risk_count:,}"), unsafe_allow_html=True)
with k4:
    st.markdown(card_html("Priority Actions", f"{priority_actions:,}"), unsafe_allow_html=True)

k5, k6, k7 = st.columns(3)
with k5:
    st.markdown(card_html("Model AUC", auc_value), unsafe_allow_html=True)
with k6:
    st.markdown(card_html("Medium Risk", f"{medium_risk_count:,}"), unsafe_allow_html=True)
with k7:
    st.markdown(card_html("Observed KPI", observed_text), unsafe_allow_html=True)

# ---------- CHART DATA ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    if not filtered_df.empty and "machine_type" in filtered_df.columns:
        type_counts = (
            filtered_df["machine_type"]
            .value_counts()
            .sort_values(ascending=True)
            .reset_index()
        )
        type_counts.columns = ["machine_type", "count"]

        fig_type = px.bar(
            type_counts,
            x="count",
            y="machine_type",
            orientation="h",
            text="count",
            color_discrete_sequence=["#3fa7ff"]
        )
        fig_type.update_traces(textposition="outside")
        themed_fig(fig_type, "Machines by Type")
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("Machine type data not available.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    if not filtered_df.empty and "risk_level" in filtered_df.columns:
        risk_counts = filtered_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        fig_donut = px.pie(
            risk_counts,
            values="count",
            names="risk_level",
            hole=0.58,
            color="risk_level",
            color_discrete_map={
                "HIGH RISK": "#1e90ff",
                "MEDIUM RISK": "#8a56ff",
                "LOW RISK": "#ff8c42"
            }
        )
        themed_fig(fig_donut, "Risk Distribution")
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Risk distribution not available.")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    if not priority_df.empty and {"machine_type", "priority"}.issubset(priority_df.columns):
        pr = (
            priority_df.groupby("machine_type", as_index=False)["priority"]
            .count()
            .rename(columns={"priority": "count"})
            .sort_values("count", ascending=True)
        )

        fig_pr = px.bar(
            pr,
            x="count",
            y="machine_type",
            orientation="h",
            text="count",
            color_discrete_sequence=["#3fa7ff"]
        )
        fig_pr.update_traces(textposition="outside")
        themed_fig(fig_pr, "Priority Machines by Type")
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info("Priority data not available.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- SECOND ROW ----------
b1, b2, b3 = st.columns([1.25, 1.25, 1.5])

with b1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    if not filtered_df.empty and "risk_level" in filtered_df.columns:
        risk_counts2 = filtered_df["risk_level"].value_counts().reset_index()
        risk_counts2.columns = ["risk_level", "count"]

        fig_small = px.pie(
            risk_counts2,
            values="count",
            names="risk_level",
            hole=0.40,
            color="risk_level",
            color_discrete_map={
                "HIGH RISK": "#5ec5ff",
                "MEDIUM RISK": "#9b6dff",
                "LOW RISK": "#ffb347"
            }
        )
        themed_fig(fig_small, "Risk Share")
        st.plotly_chart(fig_small, use_container_width=True)
    else:
        st.info("No data available.")
    st.markdown('</div>', unsafe_allow_html=True)

with b2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    if not filtered_df.empty and {"machine_type", "risk_level"}.issubset(filtered_df.columns):
        mt = (
            filtered_df.groupby("machine_type", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )

        fig_amt = px.bar(
            mt,
            x="machine_type",
            y="count",
            color_discrete_sequence=["#3fa7ff"]
        )
        themed_fig(fig_amt, "Machine Count by Category")
        st.plotly_chart(fig_amt, use_container_width=True)
    else:
        st.info("No grouped data available.")
    st.markdown('</div>', unsafe_allow_html=True)

with b3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    date_col = next(
        (c for c in filtered_df.columns if c.lower() in ["date", "event_date", "prediction_date", "timestamp", "created_at"]),
        None
    )

    if date_col:
        trend_df = filtered_df.copy()
        trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
        trend_df = trend_df.dropna(subset=[date_col])
        trend_df["month"] = trend_df[date_col].dt.to_period("M").astype(str)

        month_counts = trend_df.groupby("month", as_index=False).size()
        month_counts.columns = ["month", "count"]

        fig_line = px.line(
            month_counts,
            x="month",
            y="count",
            markers=True
        )
        fig_line.update_traces(line=dict(color="#ffe600", width=3), marker=dict(size=8))
        themed_fig(fig_line, "Monthly Risk Trend")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        fallback = (
            filtered_df.groupby("risk_level", as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        fig_line = px.line(fallback, x="risk_level", y="count", markers=True)
        fig_line.update_traces(line=dict(color="#ffe600", width=3), marker=dict(size=10))
        themed_fig(fig_line, "Risk Trend Placeholder")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- TABLE ----------
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-head">Top Maintenance Priorities</div>', unsafe_allow_html=True)

if not priority_df.empty:
    show_cols = [c for c in ["udi", "product_id", "machine_type", "risk_level", "priority"] if c in priority_df.columns]
    st.dataframe(priority_df[show_cols], use_container_width=True, height=260)
else:
    st.info("No priority data available.")

st.markdown('</div>', unsafe_allow_html=True)
