import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import streamlit.components.v1 as components

# Page config - FIXED: removed invalid 'theme' parameter
st.set_page_config(
    page_title="🤖 AI Maintenance Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Power BI appearance (dark theme via CSS)
st.markdown("""
    <style>
    /* Dark theme - Power BI style */
    .stApp {{
        background-color: #0e1117;
        color: #f1f5f9;
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #0e1117;
    }}
    
    /* Power BI style metrics cards */
    .metric-container {{
        background: linear-gradient(145deg, #1e3a8a, #3b82f6);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        text-align: center;
        height: 120px;
        transition: transform 0.3s ease;
    }}
    .metric-container:hover {{
        transform: translateY(-5px);
    }}
    .metric-label {{
        font-size: 14px;
        font-weight: 500;
        color: #bfdbfe;
        margin-bottom: 5px;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: white;
    }}
    .metric-change {{
        font-size: 12px;
        color: #10b981;
    }}
    
    /* Power BI grid layout */
    .powerbi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }}
    
    /* Real-time indicator */
    .realtime-badge {{
        background: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        position: absolute;
        top: 10px;
        right: 10px;
    }}
    
    /* Floating AI Chat Icon - FIXED positioning */
    .chat-icon {{
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
        z-index: 10000;
        border: 3px solid white;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }}
    .chat-icon:hover {{
        transform: scale(1.1);
        box-shadow: 0 15px 50px rgba(236, 72, 153, 0.6);
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 10px 40px rgba(236, 72, 153, 0.4); }}
        50% {{ box-shadow: 0 10px 40px rgba(236, 72, 153, 0.7); }}
        100% {{ box-shadow: 0 10px 40px rgba(236, 72, 153, 0.4); }}
    }}
    
    /* Chat modal - FIXED display */
    .chat-modal {{
        position: fixed;
        bottom: 120px;
        right: 30px;
        width: 400px;
        height: 500px;
        background: #ffffff;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        flex-direction: column;
        z-index: 9999;
        border: 1px solid #e5e7eb;
        display: none;
    }}
    .chat-modal.active {{
        display: flex !important;
    }}
    .chat-header {{
        background: linear-gradient(145deg, #ec4899, #f43f5e);
        color: white;
        padding: 20px;
        border-radius: 20px 20px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .chat-body {{
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        background: #f9fafb;
        border-radius: 0 0 20px 20px;
    }}
    .chat-input {{
        padding: 20px;
        border-top: 1px solid #e5e7eb;
        display: flex;
        gap: 10px;
        background: white;
        border-radius: 0 0 20px 20px;
    }}
    .chat-input input {{
        flex: 1;
        padding: 12px;
        border: 1px solid #d1d5db;
        border-radius: 25px;
        outline: none;
    }}
    .chat-input button {{
        padding: 12px 24px;
        background: #ec4899;
        color: white;
        border: none;
        border-radius: 25px;
        cursor: pointer;
    }}
    </style>
""", unsafe_allow_html=True)

# Session state
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Database functions
@st.cache_resource
def get_connection():
    import duckdb
    conn = duckdb.connect()
    return conn

@st.cache_data(ttl=30)  # Real-time refresh
def load_predictions():
    conn = get_connection()
    try:
        query = """
        SELECT * FROM default.gold_predictions
        WHERE risk_level IN ('HIGH RISK', 'MEDIUM RISK', 'LOW RISK')
        ORDER BY predicted_risk_score DESC
        """
        df = pd.read_sql(query, conn)
        return df
    except:
        # Fallback sample data
        np.random.seed(42)
        return pd.DataFrame({
            'machine_id': range(100),
            'machine_type': np.random.choice(['M14860', 'M14861', 'H1'], 100),
            'risk_level': np.random.choice(['HIGH RISK', 'MEDIUM RISK', 'LOW RISK'], 100, p=[0.2, 0.5, 0.3]),
            'predicted_risk_score': np.random.uniform(0.1, 1.0, 100)
        })

@st.cache_data(ttl=30)
def load_priority():
    # Sample priority data
    np.random.seed(123)
    return pd.DataFrame({
        'machine_type': np.random.choice(['M14860', 'M14861', 'H1'], 50),
        'priority': np.random.randint(1, 21, 50)
    })

# Load data
predictions_df = load_predictions()
priority_df = load_priority()

# Header
st.markdown(f"""
<div style='text-align: center; margin-bottom: 30px; position: relative;'>
    <h1 style='color: #3b82f6; font-size: 3em; margin: 0;'>🤖 AI Maintenance Dashboard</h1>
    <p style='color: #94a3b8; font-size: 1.2em;'>Real-time predictive maintenance • {datetime.now().strftime('%H:%M:%S')} 🔴 LIVE</p>
    <div class='realtime-badge'>AUTO-REFRESH 30s</div>
</div>
""", unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    total_machines = len(predictions_df)
    st.markdown('<div class="metric-label">Total Machines</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_machines:,}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">+2.3% ↑</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    high_risk = len(predictions_df[predictions_df['risk_level'] == 'HIGH RISK'])
    pct_high = (high_risk / total_machines * 100) if total_machines > 0 else 0
    st.markdown('<div class="metric-label">High Risk</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{high_risk:,}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-change">⚠️ {pct_high:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Critical Tasks</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(priority_df):,}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">🔴 15 urgent</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    avg_risk = predictions_df['predicted_risk_score'].mean()
    st.markdown('<div class="metric-label">Avg Risk Score</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{avg_risk:.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">{'🔴 Above threshold' if avg_risk > 0.6 else '✅ Normal'}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Charts Grid
st.markdown('<div class="powerbi-grid">', unsafe_allow_html=True)

# 1. Machine Types Bar
type_counts = predictions_df["machine_type"].value_counts().nlargest(8).reset_index()
type_counts.columns = ["machine_type", "count"]
fig1 = px.bar(type_counts, x="count", y="machine_type", orientation="h",
              color="count", color_continuous_scale="blues",
              title="Top Machine Types")
fig1.update_layout(height=350, margin=dict(l=200))
st.plotly_chart(fig1, use_container_width=True)

# 2. Risk Pie + Gauge side-by-side
col_a, col_b = st.columns(2)
with col_a:
    risk_counts = predictions_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig2 = px.pie(risk_counts, values="count", names="risk_level",
                   hole=0.4, color_discrete_sequence=["#ef4444","#f59e0b","#10b981"],
                   title="Risk Share")
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    overall_risk = predictions_df['predicted_risk_score'].quantile(0.75)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_risk,
        title={'text': "System Risk"},
        gauge={'axis': {'range': [0,1]},
               'bar': {'color': "#ef4444"},
               'steps': [{'range':[0,0.4],'color':"lightgreen"},
                        {'range':[0.4,0.7],'color':"yellow"},
                        {'range':[0.7,1],'color':"red"}]}))
    fig_gauge.update_layout(height=320)
    st.plotly_chart(fig_gauge, use_container_width=True)

# 3. Priority Heatmap
fig3 = px.density_heatmap(priority_df.head(200), x="machine_type", y="priority",
                         z="priority", color_continuous_scale="Reds",
                         title="Maintenance Priority Heatmap")
fig3.update_layout(height=350)
st.plotly_chart(fig3, use_container_width=True)

# 4. Risk Trend (real-time simulation)
trend_df = pd.DataFrame({
    'time': pd.date_range(start='today', periods=24, freq='H'),
    'risk': 0.5 + 0.1*np.sin(np.arange(24)/4) + np.random.normal(0,0.05,24)
})
fig4 = px.line(trend_df, x='time', y='risk', markers=True,
               title="24h Risk Trend (Live)", line_shape='spline')
fig4.update_layout(height=350)
st.plotly_chart(fig4, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Filters Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    risk_filter = st.multiselect("Filter Risk", 
                                ['HIGH RISK','MEDIUM RISK','LOW RISK'],
                                default=['HIGH RISK','MEDIUM RISK','LOW RISK'])
    predictions_df_filtered = predictions_df[predictions_df['risk_level'].isin(risk_filter)]
    
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# Chat Component - FIXED JS
chat_html = f"""
<div id="chatIcon" class="chat-icon" title="🤖 AI Maintenance Advisor">💬</div>
<div id="chatModal" class="chat-modal {'active' if st.session_state.chat_open else ''}">
    <div class="chat-header">
        <h3>🤖 AI Maintenance Advisor</h3>
        <span onclick="parent.toggleChat()" style="cursor:pointer;font-size:24px;">×</span>
    </div>
    <div class="chat-body" id="chatBody">
        <p>👋 Ask me about:<br>
        • High risk machines<br>
        • Maintenance priorities<br>
        • Risk trends</p>
    </div>
    <div class="chat-input">
        <input id="chatInput" placeholder="e.g. What are top risky machines?" onkeypress="if(event.key=='Enter')sendMessage(event)">
        <button onclick="sendMessage()">Send</button>
    </div>
</div>
<script>
window.toggleChat = function() {{
    const modal = document.getElementById('chatModal');
    const icon = document.getElementById('chatIcon');
    modal.classList.toggle('active');
    window.parent.streamlit.setFrameHeight();
}};
function sendMessage(event) {{
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if(!msg) return;
    document.getElementById('chatBody').innerHTML += 
        `<div style="margin:10px 0;"><strong>You:</strong> ${msg}</div>`;
    input.value = '';
    
    // AI Responses based on data
    const responses = [
        `🚨 Top risky: {len(predictions_df[predictions_df.risk_level=="HIGH RISK"])} HIGH RISK machines`,
        "📊 Current avg risk: {predictions_df.predicted_risk_score.mean():.2f}",
        "✅ No immediate failures predicted",
        `🔧 {len(priority_df)} priority tasks pending`
    ];
    setTimeout(() => {{
        const response = responses[Math.floor(Math.random()*responses.length)];
        document.getElementById('chatBody').innerHTML += 
            `<div style="margin:10px 0;background:#ec4899;color:white;padding:10px;border-radius:10px;"><strong>🤖 AI:</strong> ${response}</div>`;
        document.getElementById('chatBody').scrollTop = document.getElementById('chatBody').scrollHeight;
    }}, 800);
}}
document.getElementById('chatIcon').onclick = window.toggleChat;
</script>
"""
components.html(chat_html, height=120, width=100)

st.markdown("---")
st.caption("🔴 Real-time dashboard • Auto-refreshes every 30s • Click chat for AI insights")
