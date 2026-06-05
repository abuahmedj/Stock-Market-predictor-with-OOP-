import os
from typing import Dict, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MaxConfig:
    """Centralized configuration for the Max Engine."""
    
    # GUI Settings
    WINDOW_TITLE: str = "Max Engine Predictor"
    WINDOW_SIZE: str = "1200x800"
    
    # Theme Colors (Premium Dark/Gold Aesthetic)
    COLOR_BG_MAIN: str = "#121212"
    COLOR_BG_SIDEBAR: str = "#1A1A1A"
    COLOR_ACCENT_GOLD: str = "#D4AF37"
    COLOR_TEXT_PRIMARY: str = "#FFFFFF"
    COLOR_TEXT_SECONDARY: str = "#A0A0A0"
    COLOR_CARD_BG: str = "#242424"
    
    # ML & Data Settings
    DEFAULT_TICKER: str = "AAPL"
    HISTORY_YEARS: int = 2
    SMA_WINDOW: int = 10
    EMA_WINDOW: int = 50
    RSI_WINDOW: int = 14
    
    # LLM Settings (OpenRouter)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = "mistralai/mistral-7b-instruct:free" # Default free model
