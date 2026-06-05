import threading
from max_config import MaxConfig
from max_data_engine import MaxDataEngine
from max_ml_model import MaxPredictor
from max_nlp_analyzer import MaxSentimentAnalyzer
from max_gui_core import MaxInterface

class MaxOrchestrator:
    """
    Main Orchestrator demonstrating Dependency Injection and Threading.
    Coordinates between Data, ML, NLP, and GUI components.
    """
    def __init__(self):
        # Dependency Injection
        self.config = MaxConfig()
        self.data_engine = MaxDataEngine()
        self.ml_predictor = MaxPredictor()
        self.nlp_analyzer = MaxSentimentAnalyzer()
        
        # Pass callback to GUI for decoupling
        self.gui = MaxInterface(on_analyze_callback=self.run_analysis_async)

    def run_analysis_async(self, ticker: str):
        """Submits the heavy processing to a background thread."""
        self.log(f"[SYSTEM] Initiating analysis for {ticker}...")
        thread = threading.Thread(target=self._analysis_pipeline, args=(ticker,))
        thread.daemon = True
        thread.start()

    def _analysis_pipeline(self, ticker: str):
        """The core business logic running in the background thread."""
        try:
            # 1. Fetch Data
            self.log(f"[DATA] Fetching historical data for {ticker}...")
            df = self.data_engine.fetch_data(ticker)
            if df is None:
                self.log(f"[ERROR] Failed to fetch data for {ticker}.")
                self._safe_gui_update(state_normal=True)
                return
            
            # 2. Add Indicators
            self.log(f"[DATA] Engineering technical indicators...")
            df = self.data_engine.add_indicators(df)
            
            # 3. ML Training
            self.log(f"[ML] Training RandomForest on historical data...")
            success, mse = self.ml_predictor.train_model(df)
            if success:
                self.log(f"[ML] Training complete. MSE: {mse:.4f}")
            else:
                self.log(f"[ML] Insufficient data for training.")
            
            # 4. ML Prediction
            self.log(f"[ML] Predicting next close price...")
            pred_price = self.ml_predictor.predict_next_close(df)
            
            # 5. NLP Sentiment
            self.log(f"[NLP] Analyzing latest news sentiment...")
            sentiment_data = self.nlp_analyzer.analyze_news(ticker)
            self.log(f"[NLP] Evaluated {sentiment_data['headline_count']} headlines. Sentiment: {sentiment_data['sentiment']}")
            
            # 6. GUI Updates (Thread-Safe)
            current_price = df['Close'].iloc[-1]
            price_str = f"${current_price:.2f}"
            pred_str = f"${pred_price:.2f}" if pred_price else "N/A"
            sent_str = f"{sentiment_data['sentiment']} ({sentiment_data['score']:.2f})"
            
            self.log(f"[SYSTEM] Analysis complete for {ticker}.")
            
            # Use 'after' to safely update GUI from background thread
            self.gui.after(0, self.gui.update_metrics, price_str, pred_str, sent_str)
            self.gui.after(0, self.gui.update_chart, df, ticker)

        except Exception as e:
            self.log(f"[ERROR] {str(e)}")
            self._safe_gui_update(state_normal=True)

    def _safe_gui_update(self, state_normal: bool):
        if state_normal:
            self.gui.after(0, lambda: self.gui.analyze_btn.configure(state="normal"))

    def log(self, message: str):
        """Thread-safe logging by pushing to the GUI event loop."""
        self.gui.after(0, self.gui.log_message, message)

    def start(self):
        """Launch the system."""
        self.log("[SYSTEM] Max Engine Initialized. Calibrating predictive algorithms...")
        self.gui.mainloop()

if __name__ == "__main__":
    orchestrator = MaxOrchestrator()
    orchestrator.start()
