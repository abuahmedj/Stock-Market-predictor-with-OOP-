import streamlit as st
import pandas as pd
from max_config import MaxConfig
from max_data_engine import MaxDataEngine
from max_ml_model import MaxPredictor
from max_nlp_analyzer import MaxSentimentAnalyzer

# Configure the Streamlit browser tab
st.set_page_config(page_title="Max Engine Predictor", layout="wide")

# Initialize and cache core engines for efficient web performance
@st.cache_resource
def init_engines():
    return MaxDataEngine(), MaxPredictor(), MaxSentimentAnalyzer()

data_engine, ml_predictor, nlp_analyzer = init_engines()

# Inject custom HTML/CSS to replicate your premium dark/gold aesthetic
st.markdown("""
    <style>
    .metric-card {
        background-color: #242424;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #D4AF37;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title { color: #A0A0A0; font-size: 14px; }
    .metric-value { color: #D4AF37; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Web Sidebar Setup
st.sidebar.title("Max Engine")
ticker = st.sidebar.text_input("Enter Ticker", value="AAPL").strip().upper()
analyze_button = st.sidebar.button("Analyze", type="primary")

# Main Content Dashboard
st.title("📈 Stock Market Predictor Dashboard")

if analyze_button and ticker:
    with st.spinner(f"Initiating analysis for {ticker}..."):
        try:
            # 1. Fetch & Engineer Data
            st.info(f"[DATA] Fetching historical data and technical indicators...")
            df = data_engine.fetch_data(ticker)
            
            if df is None or df.empty:
                st.error(f"Failed to fetch data for {ticker}.")
            else:
                df = data_engine.add_indicators(df)
                
                # 2. ML Training & Prediction
                st.info("[ML] Running predictive models...")
                success, mse = ml_predictor.train_model(df)
                pred_price = ml_predictor.predict_next_close(df)
                
                # 3. NLP Sentiment Analysis
                st.info("[NLP] Extracting and analyzing news sentiment...")
                sentiment_data = nlp_analyzer.analyze_news(ticker)
                
                # 4. Display Premium Styled Metric Cards
                st.success("Analysis complete!")
                col1, col2, col3 = st.columns(3)
                
                current_price = df['Close'].iloc[-1]
                pred_str = f"${pred_price:.2f}" if pred_price else "N/A"
                sent_str = f"{sentiment_data['sentiment']} ({sentiment_data['score']:.2f})"
                
                with col1:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Current Price</div><div class="metric-value">${current_price:.2f}</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Close</div><div class="metric-value">{pred_str}</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">NLP Sentiment</div><div class="metric-value">{sent_str}</div></div>', unsafe_allow_html=True)
                
                # 5. Display Interactive Web Chart
                st.subheader(f"{ticker} Historical Close Price")
                st.line_chart(df['Close'], color="#D4AF37")
                
                # Model evaluation context
                if success:
                    st.caption(f"Model trained successfully. Mean Squared Error (MSE): {mse:.4f}")
                    
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
