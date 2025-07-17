from smc_engine import SmartMoneyEngine
from chart_pattern_engine import AdvancedChartPatternEngine
from global_news_engine import GlobalNewsAnalyzer
from datetime import datetime
import pytz
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

class LAESignalCore:
    """Core signal generation engine that combines multiple analysis techniques.
    
    Combines Smart Money Concepts, Chart Patterns, and News Analysis to generate
    trading signals with weighted confidence scores.
    """
    
    def _init_(self, config_path: Optional[str] = None):
        """Initialize the signal core with analysis engines.
        
        Args:
            config_path: Optional path to configuration file. If not provided,
                        default weights will be used.
        """
        self.smc = SmartMoneyEngine()
        self.chart = AdvancedChartPatternEngine()
        self.news = GlobalNewsAnalyzer()
        self.timezone = pytz.timezone("Asia/Kolkata")
        self.logger = self._setup_logger()
        self.weights = self._load_config(config_path) or {
            'SMC': 0.5,
            'ChartPattern': 0.3,
            'News': 0.2
        }
        
    def _load_config(self, config_path: Optional[str]) -> Optional[Dict[str, float]]:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            Dictionary with configuration or None if not found/invalid.
        """
        if not config_path:
            return None
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'weights' in config:
                    return config['weights']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load config: {str(e)}")
        return None

    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger instance.
        
        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(_name_)
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "signal_core.log")
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _validate_market_data(self, market_data: Dict) -> bool:
        """Validate market data structure.
        
        Args:
            market_data: Dictionary containing market data.
            
        Returns:
            True if data is valid, False otherwise.
        """
        required_keys = {
            'open', 'high', 'low', 'close', 'volume', 
            'timeframe', 'symbol', 'timestamp'
        }
        if not all(key in market_data for key in required_keys):
            self.logger.error(f"Missing required keys in market data. Got: {market_data.keys()}")
            return False
            
        # Additional validation for numeric values
        numeric_keys = {'open', 'high', 'low', 'close', 'volume'}
        for key in numeric_keys:
            if not isinstance(market_data[key], (int, float)):
                self.logger.error(f"Invalid type for {key}. Expected numeric, got {type(market_data[key])}")
                return False
                
        return True

    def _calculate_weighted_confidence(
        self,
        smc_result: Optional[Dict],
        chart_result: Optional[List[Dict]],
        news_result: Optional[Dict]
    ) -> Tuple[float, str, List[str]]:
        """Calculate weighted confidence score from multiple analysis results.
        
        Args:
            smc_result: Smart Money Concepts analysis result.
            chart_result: Chart Patterns analysis result.
            news_result: News analysis result.
            
        Returns:
            Tuple containing (confidence, direction, logic_used)
        """
        confidence = 0.0
        direction = "NO TRADE"
        logic_used = []
        direction_sources = []

        # Process SMC results
        if smc_result and smc_result.get('confidence', 0) > 0:
            smc_conf = smc_result['confidence'] * self.weights['SMC']
            confidence += smc_conf
            direction_sources.append(smc_result['direction'])
            logic_used.append("SMC")

        # Process Chart Patterns
        if chart_result and len(chart_result) > 0:
            chart_conf = chart_result[0]['confidence'] * self.weights['ChartPattern']
            confidence += chart_conf
            direction_sources.append(chart_result[0]['direction'])
            logic_used.append("ChartPattern")

        # Process News Analysis
        if news_result and news_result.get('score', 0) > 0:
            news_conf = (news_result['score'] / 100.0) * self.weights['News']
            confidence += news_conf
            direction_sources.append(news_result['sentiment'])
            logic_used.append("News")

        # Determine final direction (majority vote)
        if direction_sources:
            direction_counts = {}
            for d in direction_sources:
                if d != "NO TRADE":
                    direction_counts[d] = direction_counts.get(d, 0) + 1
            
            if direction_counts:
                direction = max(direction_counts.items(), key=lambda x: x[1])[0]
                
                # Apply confidence adjustments for consensus
                consensus = direction_counts[direction] / len(direction_sources)
                if consensus >= 0.67:  # 2/3 agreement
                    confidence = min(confidence * 1.1, 1.0)
                elif consensus <= 0.33:  # Low agreement
                    confidence = max(confidence * 0.9, 0.0)

        return min(max(confidence, 0), 1), direction, logic_used

    def generate_signal(
        self, 
        market_data: Dict, 
        news_data: Optional[Dict] = None
    ) -> Dict[str, any]:
        """Generate trading signal based on market and news data.
        
        Args:
            market_data: Dictionary containing market data (OHLCV + metadata).
            news_data: Optional dictionary containing news data.
            
        Returns:
            Dictionary containing signal information.
        """
        try:
            if not self._validate_market_data(market_data):
                return self._create_no_trade_response("Invalid market data")

            # Run all analyses
            smc_result = self.smc.analyze(market_data)
            chart_result = self.chart.get_latest_patterns(market_data, n=1)
            news_result = self.news.analyze(news_data) if news_data else None

            confidence, direction, logic_used = self._calculate_weighted_confidence(
                smc_result, chart_result, news_result
            )

            confidence_pct = round(confidence * 100, 2)
            timestamp = datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
            
            # Base response
            response = {
                "decision": direction,
                "confidence": confidence_pct,
                "logic_used": logic_used,
                "timestamp": timestamp,
                "symbol": market_data.get("symbol"),
                "timeframe": market_data.get("timeframe"),
                "components": {
                    "smc": smc_result or {},
                    "chart": chart_result[0] if chart_result else {},
                    "news": news_result or {}
                }
            }

            # Only add trade details if confidence is high enough and we have a direction
            if confidence_pct >= 60 and direction != "NO TRADE":
                response.update(self._get_trade_details(smc_result))
            else:
                response.update({
                    "decision": "NO TRADE",
                    "reason": "Low confidence" if confidence_pct < 60 else "No clear direction"
                })

            self.logger.info(
                f"Generated signal for {market_data.get('symbol')} "
                f"{market_data.get('timeframe')}: {direction} "
                f"(Confidence: {confidence_pct}%)"
            )
            
            return response

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}", exc_info=True)
            return self._create_no_trade_response(f"System error: {str(e)}")

    def _get_trade_details(self, smc_result: Dict) -> Dict:
        """Extract trade details from SMC result.
        
        Args:
            smc_result: Smart Money Concepts analysis result.
            
        Returns:
            Dictionary with trade details.
        """
        entry = smc_result.get("entry")
        sl = smc_result.get("sl")
        tp = smc_result.get("tp")
        
        details = {
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward_ratio": self._calculate_rr_ratio(entry, sl, tp)
        }
        
        # Add TP details if available
        if isinstance(tp, list):
            details["take_profit"] = [
                {"level": level, "weight": weight}
                for level, weight in zip(tp, smc_result.get("tp_weights", [1]*len(tp)))
            ]
        elif tp is not None:
            details["take_profit"] = [{"level": tp, "weight": 1}]
            
        return details

    def _calculate_rr_ratio(
        self, 
        entry: Optional[float], 
        sl: Optional[float], 
        tp: Optional[float]
    ) -> Optional[float]:
        """Calculate risk-reward ratio.
        
        Args:
            entry: Entry price
            sl: Stop loss price
            tp: Take profit price (or list of prices)
            
        Returns:
            Calculated risk-reward ratio or None if invalid inputs.
        """
        if None in (entry, sl) or entry == 0:
            return None
            
        try:
            risk = abs(entry - sl)
            
            # Handle single TP or multiple TPs
            if isinstance(tp, list):
                if not tp:
                    return None
                avg_tp = sum(tp) / len(tp)
                reward = abs(avg_tp - entry)
            elif tp is not None:
                reward = abs(tp - entry)
            else:
                return None
                
            return round(reward / risk, 2) if risk > 0 else None
        except Exception as e:
            self.logger.warning(f"RR calculation error: {str(e)}")
            return None

    def _create_no_trade_response(self, reason: str) -> Dict[str, any]:
        """Create a standardized 'NO TRADE' response.
        
        Args:
            reason: Explanation for no trade decision.
            
        Returns:
            Dictionary with no trade response.
        """
        return {
            "decision": "NO TRADE",
            "confidence": 0.0,
            "reason": reason,
            "timestamp": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S"),
            "logic_used": [],
            "components": {}
        }
