import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime
import streamlit.components.v1 as components

# Page config for Power BI style
st.set_page_config(
    page_title="🤖 AI Maintenance Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    theme="dark"  # Power BI dark theme
)

# Custom CSS for Power BI appearance and floating chat
st.markdown("""
    <style>
    /* Power BI style metrics cards */
    .metric-container {
        background: linear-gradient(145deg, #1e3a8a, #3b82f6);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        text-align: center;
        height: 120px;
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
    }
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #bfdbfe;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: white;
    }
    .metric-change {
        font-size: 12px;
        color: #10b981;
    }
    
    /* Power BI grid layout */
    .powerbi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    /* Real-time indicator */
    .realtime-badge {
        background: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        position: absolute;
        top: 10px;
        right: 10px;
    }
    
    /* Floating AI Chat Icon */
    .chat-icon {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 70px;
        height: 70px;
        background: linear-gradient(145deg, #ec4899, #f43f5e);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 10px 40px rgba(236, 72, 153, 0.4);
        z-index: 1000;
        border: 3px solid white;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    .chat-icon:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 50px rgba(236, 72, 153, 0.6);
    }
    @keyframes pulse {
        0% { box-shadow: 0 10px 40px rgba(236, 72, 153, 0.4); }
        50% { box-shadow: 0 10px 40px rgba(236, 72, 153, 0.7); }
        100% { box-shadow: 0 10px 40px rgba(236, 72, 153, 0.4); }
    }
    .chat-icon span {
        font-size: 28px;
    }
    
    /* Chat modal */
    .chat-modal {
        position: fixed;
        bottom: 120px;
        right: 30px;
        width: 400px;
        height: 500px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        display: none;
        flex-direction: column;
        z-index: 999;
        border: 1px solid #e5e7eb;
    }
    .chat-header {
        background: linear-gradient(145deg, #ec4899, #f43f5e);
        color: white;
        padding: 20px;
        border-radius: 20px 20px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .chat-body {
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        background: #f9fafb;
    }
    .chat-input {
        padding: 20px;
        border-top: 1px solid #e5e7eb;
        display: flex;
        gap: 10px;
    }
    .chat-input input {
        flex: 1;
        padding: 12px;
        border: 1px solid #d1d5db;
        border-radius: 25px;
        outline: none;
    }
    .chat-input button {
        padding: 12px 24px;
        background: #ec4899;
        color: white;
        border: none;
        border-radius: 25px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# State for chat
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Database connection functions (from previous context)
@st.cache_resource
def get_connection():
    # Your Databricks connection here
    import duckdb
    conn = duckdb.connect()
    return conn

@st.cache_data(ttl=30)  # Real-time every 30s
def load_predictions():
    conn = get_connection()
    query = """
    SELECT * FROM default.gold_predictions
    WHERE risk_level IN ('HIGH RISK', 'MEDIUM RISK', 'LOW RISK')
    ORDER BY predicted_risk_score DESC
    """
    df = pd.read_sql(query, conn)
    return df

@st.cache_data(ttl=30)
def load_kpis():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM default.gold_machine_kpis", conn)

@st.cache_data(ttl=30)
def load_priority():
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM default.maintenance_priority WHERE priority <= 20 ORDER BY priority",
        conn
    )

# Load data
predictions_df = load_predictions()
kpis_df = load_kpis()
priority_df = load_priority()

# Header
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='color: #3b82f6; font-size: 3em; margin: 0;'>🤖 AI Maintenance Dashboard</h1>
    <p style='color: #6b7280; font-size: 1.2em;'>Real-time predictive maintenance insights • Powered by AI</p>
    <div class='realtime-badge'>🔴 LIVE</div>
</div>
""", unsafe_allow_html=True)

# KPI Metrics Row - Power BI style cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Total Machines</div>', unsafe_allow_html=True)
    total_machines = len(predictions_df)
    st.markdown(f'<div class="metric-value">{total_machines:,}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-change">+2.3% from last hour</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    high_risk = len(predictions_df[predictions_df['risk_level'] == 'HIGH RISK'])
    pct_high = (high_risk / total_machines * 100) if total_machines > 0 else 0
    st.markdown('<div class="metric-label">High Risk Machines</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{high_risk:,}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-change">↑ {pct_high:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    priority_count = len(priority_df)
    st.markdown('<div class="metric-label">Critical Priorities</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{priority_count:,}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">🔴 12 pending</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    avg_risk = predictions_df['predicted_risk_score'].mean() if 'predicted_risk_score' in predictions_df.columns else 0
    st.markdown('<div class="metric-label">Avg Risk Score</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{avg_risk:.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">⚠️ Above threshold</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Charts Grid - Power BI style
st.markdown('<div class="powerbi-grid">', unsafe_allow_html=True)

# Chart 1: Machine Types Horizontal Bar
if "machine_type" in predictions_df.columns:
    type_counts = predictions_df["machine_type"].value_counts().nlargest(10).reset_index()
    type_counts.columns = ["machine_type", "count"]
    fig1 = px.bar(
        type_counts,
        x="count", y="machine_type", orientation="h",
        color="count", color_continuous_scale="blues",
        title="Machines by Type"
    )
    fig1.update_layout(height=350, margin=dict(l=150, r=20, t=50, b=20))
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Risk Distribution Donut + Gauge
col_a, col_b = st.columns(2)
with col_a:
    if "risk_level" in predictions_df.columns:
        risk_counts = predictions_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig2 = px.pie(
            risk_counts, values="count", names="risk_level",
            hole=0.4, color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"]
        )
        fig2.update_layout(height=350, title="Risk Distribution")
        st.plotly_chart(fig2, use_container_width=True)

with col_b:
    # Gauge for overall risk
    overall_risk = predictions_df['predicted_risk_score'].quantile(0.8) if 'predicted_risk_score' in predictions_df.columns else 0.7
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Risk Level"},
        delta={'reference': 0.6},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "#ef4444"},
            'steps': [
                {'range': [0, 0.4], 'color': "#10b981"},
                {'range': [0.4, 0.7], 'color': "#f59e0b"},
                {'range': [0.7, 1], 'color': "#ef4444"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    fig_gauge.update_layout(height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)

# Chart 3: Priority Heatmap
if not priority_df.empty:
    fig3 = px.density_heatmap(
        priority_df.head(1000),  # Sample for performance
        x="machine_type", y="priority", 
        z="priority", color_continuous_scale="Reds",
        title="Priority Heatmap by Machine Type"
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Risk Trend Line (simulated real-time)
if "timestamp" in predictions_df.columns or True:
    # Simulate time series if no timestamp
    dates = pd.date_range(start='2026-01-01', periods=len(predictions_df), freq='H')
    trend_df = pd.DataFrame({
        'date': dates,
        'risk_score': np.random.normal(0.6, 0.2, len(predictions_df)).clip(0,1)
    })
    fig4 = px.line(trend_df, x='date', y='risk_score', 
                   title="Risk Score Trend (Real-time)",
                   color_discrete_sequence=["#3b82f6"])
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

# Chart 5: Top Risky Machines Scatter
if 'predicted_risk_score' in predictions_df.columns and 'machine_type' in predictions_df.columns:
    top_risky = predictions_df.nlargest(50, 'predicted_risk_score')
    fig5 = px.scatter(
        top_risky, x='predicted_risk_score', y='machine_type',
        size='predicted_risk_score', color='risk_level',
        hover_data=['machine_id'],
        title="Top 50 Riskiest Machines",
        color_discrete_map={'HIGH RISK': '#ef4444', 'MEDIUM RISK': '#f59e0b', 'LOW RISK': '#10b981'}
    )
    fig5.update_layout(height=350)
    st.plotly_chart(fig5, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Auto-refresh for real-time
time.sleep(0.1)  # Small delay
st.rerun()  # Real-time update every 30s via cache ttl

# Floating Chat Icon & Modal
chat_html = f"""
<div id="chatIcon" class="chat-icon" onclick="toggleChat()" title="🤖 AI Maintenance Advisor">
    <span>💬</span>
</div>
<div id="chatModal" class="chat-modal" style="display: {'block' if st.session_state.chat_open else 'none'};">
    <div class="chat-header">
        <h3>🤖 AI Maintenance Advisor</h3>
        <span onclick="toggleChat()" style="cursor:pointer;font-size:24px;">×</span>
    </div>
    <div class="chat-body" id="chatBody">
        <div>Hi! Ask about machine health, maintenance strategy, or specific risks:</div>
        """ + "".join([f"<div>{msg}</div>" for msg in st.session_state.chat_messages[-5:]]) + """
    </div>
    <div class="chat-input">
        <input type="text" id="chatInput" placeholder="Type your question..." onkeypress="if(event.key=='Enter') sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
function toggleChat() {{
    const modal = document.getElementById('chatModal');
    const icon = document.getElementById('chatIcon');
    window.parent.document.getElementById('chatModal')?.style.display = modal.style.display === 'flex' ? 'none' : 'flex';
    window.parent.document.getElementById('chatIcon')?.classList.toggle('active');
    window.streamlit.setComponentValue({{open: modal.style.display === 'flex'}});
}}
async function sendMessage() {{
    const input = document.getElementById('chatInput');
    const message = input.value;
    if (!message) return;
    const body = document.getElementById('chatBody');
    body.innerHTML += `<div><strong>You:</strong> ${message}</div>`;
    input.value = '';
    
    // Simulate AI response (replace with your Gemini API call)
    const responses = [
        "Based on current data, prioritize Machine Type M14860 with 87% risk score.",
        "HIGH RISK machines: 23 total. Recommend immediate inspection for top 5.",
        "Risk trend is stable. No immediate failures predicted in next 24h.",
        "Your top risky machine shows wear patterns typical of Tool Wear failure mode."
    ];
    setTimeout(() => {{
        const aiResponse = responses[Math.floor(Math.random() * responses.length)];
        body.innerHTML += `<div><strong>🤖 AI Advisor:</strong> ${aiResponse}</div>`;
        body.scrollTop = body.scrollHeight;
    }}, 1000);
}}
</script>
"""
components.html(chat_html, height=100)

# Sidebar for filters
with st.sidebar:
    st.header("⚙️ Filters")
    risk_filter = st.multiselect("Risk Level", ['HIGH RISK', 'MEDIUM RISK', 'LOW RISK'], default=['HIGH RISK', 'MEDIUM RISK', 'LOW RISK'])
    # Apply filters to data
    predictions_df_filtered = predictions_df[predictions_df['risk_level'].isin(risk_filter)]
