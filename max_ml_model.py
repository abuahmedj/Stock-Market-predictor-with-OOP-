import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class MaxPredictor:
    """Encapsulates the Machine Learning logic using a Random Forest Regressor."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.features = ['SMA', 'EMA', 'RSI']

    def train_model(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """Trains the ML model on the provided historical data and engineered features."""
        if data.empty or len(data) < 50:
            return False, 0.0
            
        try:
            X = data[self.features]
            y = data['Target']
            
            # Clean separation of training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            
            self.is_trained = True
            return True, mse
        except Exception as e:
            raise RuntimeError(f"ML Training Error: {e}")

    def predict_next_close(self, current_data: pd.DataFrame) -> Optional[float]:
        """Predicts the next day's closing price based on the latest indicators."""
        if not self.is_trained or current_data.empty:
            return None
            
        try:
            latest_features = current_data[self.features].iloc[-1:]
            prediction = self.model.predict(latest_features)[0]
            return float(prediction)
        except Exception as e:
            raise RuntimeError(f"ML Prediction Error: {e}")
