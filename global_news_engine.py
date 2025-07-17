import re
import string
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pytz
import logging

class GlobalNewsAnalyzer:
    """Analyzes financial news headlines to determine market sentiment."""
    
    def _init_(
        self,
        sentiment_config: Optional[Dict] = None,
        timezone: str = "Asia/Kolkata",
        score_threshold: int = 2,
        negation_words: List[str] = None
    ):
        """
        Args:
            sentiment_config: Custom sentiment keywords and weights
                Format: {"sentiment_type": {"keyword": weight}}
            timezone: Timezone for output timestamps
            score_threshold: Minimum absolute score to change neutral sentiment
            negation_words: Words that invert sentiment (e.g., "not")
        """
        self.sentiment_config = sentiment_config or {
            "bullish": {
                "rally": 3, "surge": 3, "gain": 2, "breakout": 3,
                "all-time high": 4, "buy": 2, "support": 2, "rise": 2
            },
            "bearish": {
                "crash": 4, "dump": 3, "fall": 2, "sell": 2,
                "resistance": 2, "rejection": 2, "collapse": 3
            }
        }
        
        self.timezone = pytz.timezone(timezone)
        self.score_threshold = score_threshold
        self.negation_words = negation_words or ["not", "no", "never"]
        self.logger = logging.getLogger(_name_)
        
    def _clean_text(self, text: str) -> str:
        """Normalize text for analysis."""
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        return text.translate(str.maketrans("", "", string.punctuation))

    def _contains_negation(self, text: str, keyword: str) -> bool:
        """Check if keyword is negated in text."""
        for negation in self.negation_words:
            if re.search(f"{negation}\s+{keyword}", text):
                return True
        return False

    def analyze(self, news_data: Dict) -> Dict:
        """
        Analyze news headlines for market sentiment.
        
        Args:
            news_data: Dictionary containing 'headlines' key with list of strings
            
        Returns:
            Dictionary with analysis results including:
                - sentiment (bullish/bearish/neutral)
                - score (-100 to 100)
                - used_keywords (list of tuples with matched words and sentiment)
                - headline_count
                - timestamp
        """
        try:
            headlines = news_data.get("headlines", [])
            if not isinstance(headlines, list):
                raise ValueError("Headlines should be a list of strings")
                
            if not headlines:
                self.logger.info("No headlines provided, returning neutral sentiment")
                return self._neutral_response()
            
            total_score = 0
            used_keywords = []
            
            for headline in headlines:
                if not isinstance(headline, str):
                    continue
                    
                clean_headline = self._clean_text(headline)
                
                for sentiment, keywords in self.sentiment_config.items():
                    for word, weight in keywords.items():
                        if word in clean_headline:
                            if self._contains_negation(clean_headline, word):
                                # Invert sentiment for negated phrases
                                adjusted_sentiment = "bearish" if sentiment == "bullish" else "bullish"
                                adjusted_weight = -weight
                                used_keywords.append((word, f"negated_{sentiment}"))
                            else:
                                adjusted_sentiment = sentiment
                                adjusted_weight = weight
                                used_keywords.append((word, sentiment))
                            
                            total_score += adjusted_weight if sentiment == "bullish" else -adjusted_weight

            # Determine final sentiment
            abs_score = abs(total_score)
            if abs_score >= self.score_threshold:
                final_sentiment = "bullish" if total_score > 0 else "bearish"
            else:
                final_sentiment = "neutral"
            
            # Normalize score to -100 to 100 range
            normalized_score = max(-100, min(100, total_score * 10))
            
            return {
                "sentiment": final_sentiment,
                "score": normalized_score,
                "used_keywords": used_keywords,
                "headline_count": len(headlines),
                "timestamp": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return self._neutral_response()

    def _neutral_response(self) -> Dict:
        """Generate default neutral response."""
        return {
            "sentiment": "neutral",
            "score": 0,
            "used_keywords": [],
            "headline_count": 0,
            "timestamp": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
        }


# Example Usage
if _name_ == "_main_":
    logging.basicConfig(level=logging.INFO)
    
    analyzer = GlobalNewsAnalyzer()
    
    sample_news = {
        "headlines": [
            "Market surges to all-time high despite inflation fears",
            "Analysts warn of possible crash in tech sector",
            "The Fed does not support further rate hikes this year"
        ]
    }
    
    result = analyzer.analyze(sample_news)
    print("Analysis Result:", result)
