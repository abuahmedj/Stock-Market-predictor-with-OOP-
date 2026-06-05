import requests
import json
from typing import Dict, List, Any
from textblob import TextBlob
from max_config import MaxConfig

class MaxSentimentAnalyzer:
    """
    Integrates an NLP sentiment pipeline.
    Uses OpenRouter LLM for finer analysis if API key is present,
    otherwise gracefully falls back to TextBlob.
    """
    
    def __init__(self):
        self.config = MaxConfig()
        self.is_ready = True

    def _fetch_mock_headlines(self, ticker: str) -> List[str]:
        """Mock method simulating a News API call."""
        return [
            f"{ticker} announces groundbreaking new product.",
            f"{ticker} faces supply chain disruptions this quarter.",
            f"Analysts upgrade {ticker} following strong earnings report.",
            f"{ticker} expands globally despite market uncertainty.",
            f"Investors remain bullish on {ticker}'s long-term growth."
        ]

    def _analyze_with_openrouter(self, ticker: str, headlines: List[str]) -> Dict[str, Any]:
        """Calls OpenRouter LLM for deep reasoning and sentiment analysis."""
        prompt = (
            f"Analyze the following recent headlines for the stock ticker {ticker}. "
            "Determine the overall market sentiment based on the news. "
            "Respond ONLY with a valid JSON object in the exact format: "
            "{\"sentiment\": \"Positive\" | \"Negative\" | \"Neutral\", \"score\": 0.0_to_1.0_float}\n\n"
            f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines)
        )
        
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://maxengine.local", # Recommended by OpenRouter
            "X-Title": "Max Engine", # Recommended by OpenRouter
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1 # Keep temperature low for structured JSON output
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        result_content = response.json()["choices"][0]["message"]["content"]
        
        # Try to parse the JSON returned by the LLM
        # Handle cases where LLMs wrap JSON in markdown blocks (e.g., ```json ... ```)
        clean_json = result_content.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(clean_json)
        
        return {
            "sentiment": parsed.get("sentiment", "Neutral"),
            "score": float(parsed.get("score", 0.5)),
            "headline_count": len(headlines),
            "source": "OpenRouter LLM"
        }

    def _analyze_with_textblob(self, ticker: str, headlines: List[str]) -> Dict[str, Any]:
        """Fallback lightweight analysis using local TextBlob."""
        total_polarity = 0.0
        for h in headlines:
            blob = TextBlob(h)
            total_polarity += blob.sentiment.polarity
            
        avg_polarity = total_polarity / len(headlines)
        normalized_score = (avg_polarity + 1.0) / 2.0
        
        if normalized_score > 0.6:
            sentiment_label = "Positive"
        elif normalized_score < 0.4:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        return {
            "sentiment": sentiment_label,
            "score": normalized_score,
            "headline_count": len(headlines),
            "source": "TextBlob Fallback"
        }

    def analyze_news(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches the latest headlines for a ticker and calculates a weighted sentiment score.
        Uses OpenRouter if API key is configured, else uses TextBlob.
        """
        headlines = self._fetch_mock_headlines(ticker)
        if not headlines:
            return {"sentiment": "Neutral", "score": 0.5, "headline_count": 0, "source": "None"}

        if self.config.OPENROUTER_API_KEY:
            try:
                return self._analyze_with_openrouter(ticker, headlines)
            except Exception as e:
                # Fallback ensures the pipeline doesn't crash if the API limit is reached or networking fails
                print(f"OpenRouter API failed: {e}. Falling back to TextBlob.")
                return self._analyze_with_textblob(ticker, headlines)
        else:
            return self._analyze_with_textblob(ticker, headlines)
