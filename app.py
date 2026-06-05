import streamlit as st
import pandas as pd
from max_config import MaxConfig
from max_data_engine import MaxDataEngine
from max_ml_model import MaxPredictor
from max_nlp_analyzer import MaxSentimentAnalyzer

# 1. Page Configuration for a Premium Wide Layout
st.set_page_config(
    page_title="Max Engine Premier Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Cache Core Infrastructure Objects for Peak Web Performance
@st.cache_resource
def init_system_cores():
    return MaxDataEngine(), MaxPredictor(), MaxSentimentAnalyzer()

data_engine, ml_predictor, nlp_analyzer = init_system_cores()

# 3. Inject Customized CSS for a Premium Dark/Gold Financial Theme
st.markdown("""
    <style>
    /* Global Background Adjustments */
    .stApp { background-color: #121212; }
    
    /* Metrics Component Cards */
    .metric-container {
        background-color: #1A1A1A;
        border: 1px solid #242424;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, border-color 0.2s;
        margin-bottom: 15px;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: #D4AF37;
    }
    .metric-label {
        color: #A0A0A0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-val {
        color: #D4AF37;
        font-size: 2.2rem;
        font-weight: 800;
    }
    
    /* Live Activity Log Styling */
    .log-box {
        background-color: #1A1A1A;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        color: #FFFFFF;
        border-left: 4px solid #D4AF37;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Sidebar Control Panel
with st.sidebar:
    st.markdown("<h1 style='color: #D4AF37; font-size: 2rem; font-weight: bold; margin-bottom: 0;'>Max Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #A0A0A0; font-size: 0.9rem; margin-top: 0;'>Predictive Algorithmic Suite</p>", unsafe_allow_html=True)
    
    # Replaced non-existent st.hr() with native divider
    st.divider()
    
    ticker = st.text_input("💎 Stock Ticker Symbol", value="AAPL").strip().upper()
    
    st.markdown("### Engine Parameters")
    st.caption("Configured dynamically via structural configuration layer.")
    st.info("⚡ Live Connection: Stable\n\n🔮 Model: Random Forest\n\n📰 NLP: Hybrid Pipeline")
    
    analyze_btn = st.button("RUN QUANT ANALYSIS", type="primary", use_container_width=True)

# 5. Main Dashboard View Layout
st.markdown(f"<h2 style='color: white; font-weight: 700;'>📊 Core Analytical Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #A0A0A0;'>Real-time technical indicators, machine learning forecasting models, and deep NLP market sentiment tracking.</p>", unsafe_allow_html=True)

if analyze_btn and ticker:
    # Set up a real-time console block mimicking your previous background logging engine
    log_placeholder = st.empty()
    
    with log_placeholder.container():
        st.markdown(f"<div class='log-box'>[SYSTEM] Initiating cloud analysis pipeline for {ticker}...</div>", unsafe_allow_html=True)
        
    try:
        # Step A: Data Engine Extraction and Feature Engineering
        df = data_engine.fetch_data(ticker)
        if df is None or df.empty:
            st.error(f"❌ Error: Unable to query market records for asset '{ticker}'. Verify symbol or configuration parameters.")
        else:
            with log_placeholder.container():
                st.markdown("<div class='log-box'>[DATA] Engineering structural indicators (SMA, EMA, RSI)...</div>", unsafe_allow_html=True)
            df = data_engine.add_indicators(df)
            
            # Step B: Machine Learning Intelligence Processing
            with log_placeholder.container():
                st.markdown("<div class='log-box'>[ML] Optimizing Random Forest Regressor hyper-parameters...</div>", unsafe_allow_html=True)
            success, mse = ml_predictor.train_model(df)
            pred_price = ml_predictor.predict_next_close(df)
            
            # Step C: Natural Language Sentiment Calculations
            with log_placeholder.container():
                st.markdown("<div class='log-box'>[NLP] Evaluating global market headline clusters...</div>", unsafe_allow_html=True)
            sentiment_data = nlp_analyzer.analyze_news(ticker)
            
            # Remove execution logs upon successful structural rendering
            log_placeholder.empty()
            
            # Step D: Construct Premium Metrics Grid Layout
            m_col1, m_col2, m_col3 = st.columns(3)
            
            current_price = df['Close'].iloc[-1]
            pred_str = f"${pred_price:.2f}" if pred_price else "N/A"
            sent_str = f"{sentiment_data['sentiment']} ({sentiment_data['score']:.2f})"
            
            with m_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Current Spot Price</div>
                    <div class="metric-val">${current_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Predicted Day Close</div>
                    <div class="metric-val">{pred_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">NLP Sentiment Index</div>
                    <div class="metric-val">{sent_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Step E: High-Fidelity Charting Visualizations via Native View Tabs
            chart_tab1, chart_tab2 = st.tabs(["📈 Interactive Price History", "🛠️ Technical Analytics Layer"])
            
            with chart_tab1:
                st.markdown("#### Dynamic Historical Asset Valuations")
                # Responsive line chart supporting panning, zoom windows, and precise hover cards
                st.line_chart(df[['Close']], color="#D4AF37", height=400)
                
            with chart_tab2:
                st.markdown("#### Algorithmic Overlay Analysis (SMA vs EMA)")
                st.line_chart(df[['Close', 'SMA', 'EMA']], color=["#D4AF37", "#4A90E2", "#E24A4A"], height=350)
                
                st.markdown("#### Momentum Vector (Relative Strength Index)")
                st.area_chart(df['RSI'], color="#A0A0A0", height=150)
                
            # System Metrics Evaluation Summary Footer
            if success:
                st.caption(f"Engine Core: Optimization Matrix Complete. Structural Backtest MSE: {mse:.6f} | NLP Dataset: {sentiment_data['headline_count']} headlines ingested via {sentiment_data['source']}.")
                
    except Exception as e:
        log_placeholder.empty()
        st.error(f"Engine Exception Triggered: {str(e)}")
else:
    # Dashboard state before ticker submission
    st.info("💡 Input a valid stock symbol in the left control panel and execute the analytical matrix to generate premium visualization telemetry.")
