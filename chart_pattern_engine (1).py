import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import pytz
from collections import defaultdict
import logging
from dataclasses import dataclass
from enum import Enum

class PatternDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"

@dataclass
class PatternResult:
    timestamp: str
    pattern_name: str
    direction: PatternDirection
    confidence: float
    price: float
    volume: float
    trend_alignment: bool
    confirmation_required: bool = False

class AdvancedChartPatternEngine:
    """
    Advanced candlestick pattern detection engine using TA-Lib with:
    - Volume-weighted confidence scoring
    - Trend context analysis
    - Statistical pattern tracking
    - Multi-timeframe support
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize pattern engine with configurable confidence threshold
        
        Args:
            confidence_threshold: Minimum confidence score (0-1) for valid patterns
        """
        self.logger = logging.getLogger(__name__)
        self._initialize_patterns()
        self.confidence_threshold = min(max(confidence_threshold, 0.1), 1.0)
        self.timezone = pytz.timezone("Asia/Kolkata")
        self.pattern_stats = defaultdict(int)
        self._setup_pattern_weights()

    def _initialize_patterns(self):
        """Configure all TA-Lib patterns with normalization"""
        self.patterns = {
            'Hammer': talib.CDLHAMMER,
            'InvertedHammer': talib.CDLINVERTEDHAMMER,
            'ShootingStar': talib.CDLSHOOTINGSTAR,
            'HangingMan': talib.CDLHANGINGMAN,
            'Doji': talib.CDLDOJI,
            'LongLeggedDoji': talib.CDLLONGLEGGEDDOJI,
            'GravestoneDoji': talib.CDLGRAVESTONEDOJI,
            'Engulfing': talib.CDLENGULFING,
            'Harami': talib.CDLHARAMI,
            'Piercing': talib.CDLPIERCING,
            'DarkCloudCover': talib.CDLDARKCLOUDCOVER,
            'MorningStar': talib.CDLMORNINGSTAR,
            'EveningStar': talib.CDLEVENINGSTAR,
            'ThreeWhiteSoldiers': talib.CDL3WHITESOLDIERS,
            'ThreeBlackCrows': talib.CDL3BLACKCROWS,
            'MorningDojiStar': talib.CDLMORNINGDOJISTAR,
            'EveningDojiStar': talib.CDLEVENINGDOJISTAR
        }

    def _setup_pattern_weights(self):
        """Configure base confidence weights for each pattern"""
        self.pattern_weights = {
            # Single candle patterns
            'Hammer': 0.65, 
            'InvertedHammer': 0.65,
            'ShootingStar': 0.7,
            'HangingMan': 0.7,
            'Doji': 0.5,
            'LongLeggedDoji': 0.55,
            'GravestoneDoji': 0.6,
            
            # Two candle patterns
            'Engulfing': 0.75,
            'Harami': 0.6,
            'Piercing': 0.7,
            'DarkCloudCover': 0.7,
            
            # Three candle patterns
            'MorningStar': 0.8,
            'EveningStar': 0.8,
            'ThreeWhiteSoldiers': 0.8,
            'ThreeBlackCrows': 0.8,
            'MorningDojiStar': 0.85,
            'EveningDojiStar': 0.85
        }

    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Detect all candlestick patterns in the given OHLCV data
        
        Args:
            df: Pandas DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            List of PatternResult objects sorted by timestamp (newest first)
        """
        self._validate_dataframe(df)
        results = []

        try:
            for name, pattern_func in self.patterns.items():
                signals = pattern_func(df['open'], df['high'], df['low'], df['close'])
                
                for idx in np.where(signals != 0)[0]:
                    pattern_result = self._analyze_pattern(
                        df=df,
                        pattern_name=name,
                        signal_value=signals[idx],
                        candle_index=idx
                    )
                    
                    if pattern_result.confidence >= self.confidence_threshold:
                        results.append(pattern_result)
                        self.pattern_stats[name] += 1
            
            return sorted(results, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}", exc_info=True)
            raise

    def get_latest_patterns(self, df: pd.DataFrame, n: int = 1) -> List[PatternResult]:
        """
        Get highest confidence patterns from the dataset
        
        Args:
            df: OHLCV DataFrame
            n: Number of patterns to return
            
        Returns:
            List of PatternResult objects sorted by confidence (highest first)
        """
        patterns = self.detect_patterns(df)
        return sorted(patterns, key=lambda x: -x.confidence)[:n]

    def _analyze_pattern(self, df: pd.DataFrame, pattern_name: str, 
                         signal_value: int, candle_index: int) -> PatternResult:
        """
        Analyze a single detected pattern with market context
        
        Args:
            df: OHLCV DataFrame
            pattern_name: Name of detected pattern
            signal_value: TA-Lib signal strength
            candle_index: Index location in DataFrame
            
        Returns:
            PatternResult with complete analysis
        """
        context = self._get_market_context(df, candle_index)
        confidence = self._calculate_confidence(
            pattern_name=pattern_name,
            signal_value=signal_value,
            context=context
        )
        
        direction = PatternDirection.BULLISH if signal_value > 0 else PatternDirection.BEARISH
        trend_alignment = (
            (direction == PatternDirection.BULLISH and context['trend'] == 'up') or
            (direction == PatternDirection.BEARISH and context['trend'] == 'down')
        )
        
        return PatternResult(
            timestamp=self._convert_to_ist(df['timestamp'].iloc[candle_index]),
            pattern_name=pattern_name,
            direction=direction,
            confidence=round(confidence, 4),
            price=float(df['close'].iloc[candle_index]),
            volume=float(df['volume'].iloc[candle_index]),
            trend_alignment=trend_alignment,
            confirmation_required=self._needs_confirmation(pattern_name)
        )

    def _get_market_context(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Calculate market context for pattern validation:
        - Volume analysis
        - Trend direction
        - Volatility
        """
        lookback = min(14, idx)
        avg_volume = df['volume'].iloc[max(0, idx - lookback):idx].mean()
        current_volume = df['volume'].iloc[idx]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        ema_9 = talib.EMA(df['close'], timeperiod=9).iloc[idx]
        ema_21 = talib.EMA(df['close'], timeperiod=21).iloc[idx]
        trend = 'up' if ema_9 > ema_21 else 'down'

        atr = talib.ATR(
            df['high'], 
            df['low'], 
            df['close'], 
            timeperiod=14
        ).iloc[idx]

        return {
            'volume_ratio': volume_ratio,
            'trend': trend,
            'volatility': atr,
            'candle_size': df['high'].iloc[idx] - df['low'].iloc[idx]
        }

    def _calculate_confidence(self, pattern_name: str, 
                              signal_value: int, context: Dict) -> float:
        """
        Calculate weighted confidence score (0-1) considering:
        - Base pattern reliability
        - Volume confirmation
        - Trend alignment
        - Signal strength
        """
        base_confidence = self.pattern_weights.get(pattern_name, 0.6)
        
        volume_factor = min(1.5, 0.5 + context['volume_ratio'])
        trend_factor = 1.2 if context['trend'] == 'up' else 0.8
        strength_factor = min(1.0, abs(signal_value) / 100.0)
        volatility_factor = min(1.2, max(0.8, context['volatility'] / (context['candle_size'] + 1e-6)))

        confidence = base_confidence * volume_factor * trend_factor * strength_factor * volatility_factor
        return min(max(confidence, 0), 1.0)

    def _needs_confirmation(self, pattern_name: str) -> bool:
        """Determine if pattern requires next-candle confirmation"""
        return pattern_name in ['Doji', 'Hammer', 'HangingMan', 'ShootingStar', 'InvertedHammer']

    def _convert_to_ist(self, timestamp) -> str:
        """Convert timestamp to IST timezone string"""
        if isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        if timestamp.tzinfo is None:
            timestamp = pytz.utc.localize(timestamp)
        return timestamp.astimezone(self.timezone).strftime("%Y-%m-%d %H:%M:%S")

    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate OHLCV dataframe structure"""
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < 21:
            raise ValueError("Dataframe must contain at least 21 periods for reliable pattern detection")
