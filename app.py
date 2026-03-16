"""
Predictive Maintenance Copilot
Databricks Lakehouse + Random Forest (AUC 0.954) + Gemini 3 Flash
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import databricks.sql
import google.generativeai as genai

load_dotenv()

st.set_page_config(
    layout="wide",
    page_title="Maintenance Copilot",
    page_icon="🔧"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }
    .kpi-card {
        background: white;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 5px solid #1f77b4;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    .floating-chat-note {
        position: fixed;
        right: 22px;
        bottom: 18px;
        background: #0f62fe;
        color: white;
        padding: 10px 14px;
        border-radius: 999px;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(15,98,254,0.35);
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔧 Predictive Maintenance Copilot")
st.markdown("**Databricks Lakehouse | Random Forest AUC 0.954 | Gemini 3 Flash**")

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.title("⚙️ Controls")
product_search = st.sidebar.text_input("🔍 Search Product ID")
risk_filter = st.sidebar.multiselect(
    "Filter Risk Level",
    ["HIGH RISK", "MEDIUM RISK", "LOW RISK"],
    default=["HIGH RISK", "MEDIUM RISK", "LOW RISK"]
)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

@st.cache_resource
def get_connection():
    return databricks.sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )

@st.cache_resource
def get_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-3-flash-preview")

@st.cache_data(ttl=60)
def load_predictions():
    conn = get_connection()
    query = """
    SELECT * FROM default.gold_predictions
    WHERE risk_level IN ('HIGH RISK', 'MEDIUM RISK', 'LOW RISK')
    """
    df = pd.read_sql(query, conn)

    if product_search and "product_id" in df.columns:
        df = df[df["product_id"].astype(str).str.contains(product_search, case=False, na=False)]

    if risk_filter and "risk_level" in df.columns:
        df = df[df["risk_level"].isin(risk_filter)]

    return df

@st.cache_data(ttl=60)
def load_kpis():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM default.gold_machine_kpis", conn)

@st.cache_data(ttl=60)
def load_priority():
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM default.maintenance_priority WHERE priority <= 20 ORDER BY priority",
        conn
    )

predictions_df = load_predictions()
kpis_df = load_kpis()
priority_df = load_priority()

def kpi_card(label, value, subtext=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# KPI calculations
if not predictions_df.empty and "risk_level" in predictions_df.columns:
    total_machines = len(predictions_df)
    high_risk_count = int((predictions_df["risk_level"] == "HIGH RISK").sum())
    medium_risk_count = int((predictions_df["risk_level"] == "MEDIUM RISK").sum())
    low_risk_count = int((predictions_df["risk_level"] == "LOW RISK").sum())
    high_risk_pct = high_risk_count / total_machines if total_machines else 0
else:
    total_machines = high_risk_count = medium_risk_count = low_risk_count = 0
    high_risk_pct = 0

observed_text = "Not mapped"
if not kpis_df.empty:
    possible_cols = ["failure_rate", "avg_failure_rate", "actual_failure_rate", "prediction"]
    found_col = next((c for c in possible_cols if c in kpis_df.columns), None)
    if found_col:
        vals = pd.to_numeric(kpis_df[found_col], errors="coerce").dropna()
        if len(vals) > 0:
            observed_text = f"{vals.mean():.1%}"

r1, r2, r3, r4, r5 = st.columns(5)
with r1:
    kpi_card("Total Machines", f"{total_machines:,}", "Filtered live view")
with r2:
    kpi_card("High Risk", f"{high_risk_count:,}", f"{high_risk_pct:.1%} of filtered machines")
with r3:
    kpi_card("Medium Risk", f"{medium_risk_count:,}", "Current machine count")
with r4:
    kpi_card("Priority Actions", f"{len(priority_df):,}", "Priority <= 20")
with r5:
    kpi_card("Model AUC", "0.954", f"Observed KPI: {observed_text}")

def style_fig(fig, title):
    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    return fig

# Row 1
c1, c2 = st.columns(2)

with c1:
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        risk_counts = predictions_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        fig_pie = px.pie(
            risk_counts,
            values="count",
            names="risk_level",
            hole=0.45,
            color="risk_level",
            color_discrete_map={
                "HIGH RISK": "#d62728",
                "MEDIUM RISK": "#ff7f0e",
                "LOW RISK": "#2ca02c"
            }
        )
        style_fig(fig_pie, "Risk Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No prediction data available.")

with c2:
    if not predictions_df.empty and {"machine_type", "risk_level"}.issubset(predictions_df.columns):
        chart_df = (
            predictions_df.groupby(["machine_type", "risk_level"])
            .size()
            .reset_index(name="count")
        )
        fig_bar = px.bar(
            chart_df,
            x="machine_type",
            y="count",
            color="risk_level",
            barmode="group",
            color_discrete_map={
                "HIGH RISK": "#d62728",
                "MEDIUM RISK": "#ff7f0e",
                "LOW RISK": "#2ca02c"
            }
        )
        style_fig(fig_bar, "Risk by Machine Type")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Machine type chart data not available.")

# Row 2
c3, c4, c5 = st.columns(3)

with c3:
    if not predictions_df.empty and "machine_type" in predictions_df.columns:
        type_counts = predictions_df["machine_type"].value_counts().reset_index()
        type_counts.columns = ["machine_type", "count"]

        fig_type = px.bar(
            type_counts.sort_values("count", ascending=False),
            x="machine_type",
            y="count",
            color_discrete_sequence=["#1f77b4"]
        )
        style_fig(fig_type, "Machine Volume by Type")
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("Machine type data not available.")

with c4:
    if not predictions_df.empty and "risk_level" in predictions_df.columns:
        fig_hist = px.histogram(
            predictions_df,
            x="risk_level",
            color="risk_level",
            color_discrete_map={
                "HIGH RISK": "#d62728",
                "MEDIUM RISK": "#ff7f0e",
                "LOW RISK": "#2ca02c"
            }
        )
        style_fig(fig_hist, "Risk Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Risk histogram not available.")

with c5:
    gauge_value = round(high_risk_pct * 100, 1)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={"text": "High Risk %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#d62728"},
            "steps": [
                {"range": [0, 30], "color": "#dff5e1"},
                {"range": [30, 60], "color": "#ffe9c7"},
                {"range": [60, 100], "color": "#ffd1d1"}
            ]
        }
    ))
    style_fig(fig_gauge, "High Risk Gauge")
    st.plotly_chart(fig_gauge, use_container_width=True)

# Row 3
c6, c7 = st.columns(2)

with c6:
    if not priority_df.empty and {"machine_type", "priority"}.issubset(priority_df.columns):
        priority_type = (
            priority_df.groupby("machine_type", as_index=False)["priority"]
            .count()
            .rename(columns={"priority": "count"})
        )
        fig_priority = px.bar(
            priority_type.sort_values("count", ascending=False),
            x="machine_type",
            y="count",
            color_discrete_sequence=["#6f42c1"]
        )
        style_fig(fig_priority, "Priority Actions by Machine Type")
        st.plotly_chart(fig_priority, use_container_width=True)
    else:
        st.info("Priority data not available.")

with c7:
    date_col = next(
        (c for c in predictions_df.columns if c.lower() in ["date", "event_date", "prediction_date", "timestamp", "created_at"]),
        None
    )

    if date_col:
        trend_df = predictions_df.copy()
        trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
        trend_df = trend_df.dropna(subset=[date_col])
        trend_df["day"] = trend_df[date_col].dt.date.astype(str)

        if not trend_df.empty:
            daily = trend_df.groupby("day", as_index=False).size()
            daily.columns = ["day", "count"]
            fig_line = px.line(daily, x="day", y="count", markers=True)
            style_fig(fig_line, "Daily Prediction Trend")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Date column exists but no valid timestamps found.")
    else:
        fallback = predictions_df["risk_level"].value_counts().reset_index() if not predictions_df.empty and "risk_level" in predictions_df.columns else pd.DataFrame()
        if not fallback.empty:
            fallback.columns = ["risk_level", "count"]
            fig_fallback = px.line(fallback, x="risk_level", y="count", markers=True)
            style_fig(fig_fallback, "Risk Trend Fallback")
            st.plotly_chart(fig_fallback, use_container_width=True)
        else:
            st.info("Trend data not available.")

st.subheader("🎯 Top Maintenance Priorities")
if not priority_df.empty:
    display_cols = ["udi", "product_id", "machine_type", "risk_level", "priority"]
    available_cols = [col for col in display_cols if col in priority_df.columns]
    st.dataframe(priority_df[available_cols], use_container_width=True, height=260)
else:
    st.info("No priority data available.")

st.markdown("---")

# Floating launcher note
st.markdown(
    '<div class="floating-chat-note">🤖 Open AI Advisor from below</div>',
    unsafe_allow_html=True
)

chat_col1, chat_col2, chat_col3 = st.columns([6, 1.2, 1.2])
with chat_col3:
    if st.button("💬 Chat", use_container_width=True):
        st.session_state.chat_open = not st.session_state.chat_open

if st.session_state.chat_open:
    st.subheader("🤖 AI Maintenance Advisor")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about machine health, risk, or maintenance priority")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            llm = get_llm()
            safe_cols = [col for col in ["udi", "product_id", "machine_type", "risk_level", "priority"] if col in priority_df.columns]
            context = (
                priority_df[safe_cols].head(10).to_string(index=False)
                if not priority_df.empty and safe_cols
                else "No priority data available"
            )

            ai_prompt = f"""
You are a predictive maintenance assistant.

Use ONLY the data provided below.
Do NOT use outside knowledge.
Do NOT invent dates, thresholds, causes, downtime, cost savings, tool wear, machine_failure values, or any field not explicitly present.
If something is not explicitly present in the data, say: Not available in provided data.

Provided data:
{context}

User question:
{prompt}

Return the answer in exactly this plain-text format:

Summary:
- ...

Priority machines:
- udi: ..., product_id: ..., machine_type: ..., risk_level: ..., priority: ...

Recommended actions:
- ...
- ...

Missing data:
- ...
- ...
"""

            response = llm.generate_content(ai_prompt)
            answer = response.text.strip()

            formatted = (
                answer.replace("Summary:", "#### Summary")
                      .replace("Priority machines:", "#### Priority machines")
                      .replace("Recommended actions:", "#### Recommended actions")
                      .replace("Missing data:", "#### Missing data")
            )

            st.session_state.chat_history.append({"role": "assistant", "content": formatted})

            with st.chat_message("assistant"):
                st.markdown(formatted)

        except Exception as e:
            err = f"AI service error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.error(err)

st.markdown("---")
f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("**🗄️ Databricks Lakehouse**")
    st.markdown("- Medallion Architecture (Bronze/Silver/Gold)")
    st.markdown("- Delta Lake (ACID + Time Travel)")

with f2:
    st.markdown("**🤖 ML Pipeline**")
    st.markdown("- Random Forest Classification")
    st.markdown("- AUC: 0.954")

with f3:
    st.markdown("**🚀 Production**")
    st.markdown("- Batch Inference Pipeline")
    st.markdown("- Real-time Risk Dashboard")

st.markdown("---")
st.caption("**Built by Anchit Chourasia** | Aspiring AI Engineer | [GitHub](https://github.com/anchitchourasia)")
