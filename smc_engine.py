import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from enum import Enum
import pandas_ta as ta
from datetime import datetime

class MarketStructure(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    TRANSITION = "transition"

class LiquidityEvent(Enum):
    BUY_SIDE_SWEEP = "buy_side_liquidity_sweep"
    SELL_SIDE_SWEEP = "sell_side_liquidity_sweep"
    NONE = None

class OrderBlockType(Enum):
    BULLISH = "bullish_ob"
    BEARISH = "bearish_ob"
    NONE = None

class SmartMoneyEngine:
    def __init__(self, min_confidence: float = 0.6, lookback_period: int = 100):
        self.min_confidence = min_confidence
        self.lookback_period = lookback_period
        self.validated_ob_zones = []

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_100'] = ta.ema(df['close'], length=100)

        bb = ta.bbands(df['close'], length=20, std=2)
        df['upper_bb'] = bb['BBU_20_2.0']
        df['middle_bb'] = bb['BBM_20_2.0']
        df['lower_bb'] = bb['BBL_20_2.0']

        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        return df

    def _determine_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        price_above_ema20 = df['close'].iloc[-1] > df['ema_20'].iloc[-1]
        price_above_ema50 = df['close'].iloc[-1] > df['ema_50'].iloc[-1]
        ema20_above_ema50 = df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]
        ema50_above_ema100 = df['ema_50'].iloc[-1] > df['ema_100'].iloc[-1]

        if price_above_ema20 and price_above_ema50 and ema20_above_ema50 and ema50_above_ema100:
            return MarketStructure.BULLISH

        price_below_ema20 = df['close'].iloc[-1] < df['ema_20'].iloc[-1]
        price_below_ema50 = df['close'].iloc[-1] < df['ema_50'].iloc[-1]
        ema20_below_ema50 = df['ema_20'].iloc[-1] < df['ema_50'].iloc[-1]
        ema50_below_ema100 = df['ema_50'].iloc[-1] < df['ema_100'].iloc[-1]

        if price_below_ema20 and price_below_ema50 and ema20_below_ema50 and ema50_below_ema100:
            return MarketStructure.BEARISH

        ema_distance = abs(df['ema_20'].iloc[-1] - df['ema_50'].iloc[-1])
        avg_candle_size = (df['high'] - df['low']).rolling(20).mean().iloc[-1]

        if ema_distance < 0.5 * avg_candle_size:
            return MarketStructure.SIDEWAYS

        return MarketStructure.TRANSITION

    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> LiquidityEvent:
        high_sweep = df['high'].iloc[-1] > df['high'].rolling(window=5).max().shift(1).iloc[-1]
        low_sweep = df['low'].iloc[-1] < df['low'].rolling(window=5).min().shift(1).iloc[-1]

        volume_confirmation = True
        if 'volume' in df.columns:
            volume_confirmation = df['volume_ratio'].iloc[-1] > 1.5

        if high_sweep and volume_confirmation:
            return LiquidityEvent.BUY_SIDE_SWEEP
        elif low_sweep and volume_confirmation:
            return LiquidityEvent.SELL_SIDE_SWEEP

        return LiquidityEvent.NONE

    def _identify_order_blocks(self, df: pd.DataFrame, structure: MarketStructure) -> List[Tuple[float, OrderBlockType]]:
        ob_zones = []
        if structure in [MarketStructure.BULLISH, MarketStructure.TRANSITION]:
            for i in range(3, len(df)):
                if df['close'].iloc[i-1] < df['open'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i] > df['close'].iloc[i-1]:
                    ob_zones.append((df['open'].iloc[i], OrderBlockType.BULLISH))

        if structure in [MarketStructure.BEARISH, MarketStructure.TRANSITION]:
            for i in range(3, len(df)):
                if df['close'].iloc[i-1] > df['open'].iloc[i-1] and df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i] < df['close'].iloc[i-1]:
                    ob_zones.append((df['open'].iloc[i], OrderBlockType.BEARISH))

        return ob_zones

    def _validate_order_blocks(self, df: pd.DataFrame, ob_zones: List[Tuple[float, OrderBlockType]]) -> List[Tuple[float, OrderBlockType]]:
        validated_ob = []
        for price, ob_type in ob_zones:
            recent_data = df.iloc[-5:]
            if ob_type == OrderBlockType.BULLISH:
                if any(recent_data['low'] >= price * 0.995 and recent_data['close'].iloc[-1] > price):
                    validated_ob.append((price, ob_type))
            elif ob_type == OrderBlockType.BEARISH:
                if any(recent_data['high'] <= price * 1.005 and recent_data['close'].iloc[-1] < price):
                    validated_ob.append((price, ob_type))
        return validated_ob

    def _calculate_risk_parameters(self, df: pd.DataFrame, structure: MarketStructure) -> Dict:
        atr = df['atr'].iloc[-1]
        close = df['close'].iloc[-1]

        if structure == MarketStructure.BULLISH:
            sl = df['low'].rolling(5).min().iloc[-1] - 0.25 * atr
            entry = close
            tp = entry + 2 * (entry - sl)
            resistance = df['high'].rolling(20).max().iloc[-1]
            if resistance - entry < 3 * atr:
                tp = min(tp, resistance - 0.5 * atr)
        elif structure == MarketStructure.BEARISH:
            sl = df['high'].rolling(5).max().iloc[-1] + 0.25 * atr
            entry = close
            tp = entry - 2 * (sl - entry)
            support = df['low'].rolling(20).min().iloc[-1]
            if entry - support < 3 * atr:
                tp = max(tp, support + 0.5 * atr)
        else:
            return {'entry': None, 'sl': None, 'tp': None}

        return {'entry': round(entry, 4), 'sl': round(sl, 4), 'tp': round(tp, 4)}

    def _calculate_confidence(self, df: pd.DataFrame, structure: MarketStructure, liquidity_event: LiquidityEvent, ob_zones: List) -> float:
        confidence = 0.5
        if structure in [MarketStructure.BULLISH, MarketStructure.BEARISH]:
            confidence += 0.2
        elif structure == MarketStructure.TRANSITION:
            confidence += 0.1

        if liquidity_event != LiquidityEvent.NONE:
            confidence += 0.15
        if ob_zones:
            confidence += 0.1

        rsi = df['rsi'].iloc[-1]
        if (structure == MarketStructure.BULLISH and 30 < rsi < 70) or (structure == MarketStructure.BEARISH and 30 < rsi < 70):
            confidence += 0.05

        if 'volume' in df.columns and df['volume_ratio'].iloc[-1] > 1.5:
            confidence += 0.05

        ema_separation = abs(df['ema_20'].iloc[-1] - df['ema_50'].iloc[-1])
        avg_candle_size = (df['high'] - df['low']).rolling(20).mean().iloc[-1]
        if ema_separation > avg_candle_size:
            confidence += 0.05

        return min(max(confidence, 0), 1)

    def analyze(self, market_data: Dict) -> Optional[Dict]:
        try:
            df = pd.DataFrame([market_data])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            df = self._calculate_technical_indicators(df)
            structure = self._determine_market_structure(df)
            liquidity_event = self._detect_liquidity_sweeps(df)
            ob_zones = self._identify_order_blocks(df, structure)
            validated_ob = self._validate_order_blocks(df, ob_zones)
            risk_params = self._calculate_risk_parameters(df, structure)
            confidence = self._calculate_confidence(df, structure, liquidity_event, validated_ob)

            if structure == MarketStructure.BULLISH and confidence >= self.min_confidence:
                direction = "BUY"
            elif structure == MarketStructure.BEARISH and confidence >= self.min_confidence:
                direction = "SELL"
            else:
                direction = "NO_TRADE"
                risk_params = {'entry': None, 'sl': None, 'tp': None}

            result = {
                'timestamp': datetime.now().isoformat(),
                'market_structure': structure.value,
                'liquidity_event': liquidity_event.value,
                'order_blocks': [{'price': price, 'type': ob_type.value} for price, ob_type in validated_ob],
                'direction': direction,
                'entry': risk_params['entry'],
                'stop_loss': risk_params['sl'],
                'take_profit': risk_params['tp'],
                'confidence': round(confidence * 100, 2),
                'indicators': {
                    'ema_20': round(df['ema_20'].iloc[-1], 4),
                    'ema_50': round(df['ema_50'].iloc[-1], 4),
                    'rsi': round(df['rsi'].iloc[-1], 2),
                    'atr': round(df['atr'].iloc[-1], 4)
                }
            }

            return result

        except Exception as e:
            print(f"❌ SMC Analysis Error: {str(e)}")
            return None
