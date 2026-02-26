"""
strategy.py — Trading Strategy Engine v2.1
==========================================
FIXES v2.0 problems:
- BB: Relaxed buy condition (near lower band, not touching)
- Combined: Lowered to 2+ confirms with lookback window
- Sells: ATR-based trailing stop instead of indicator sells
- Bull market bias: Favor holding over selling

Author: Your Name
Version: 2.1.0
"""

import pandas as pd
import numpy as np
from loguru import logger

# Suppress pandas warnings
pd.set_option('future.no_silent_downcasting', True)


class TradingStrategy:
    """Generates trading signals with trend filtering and risk controls."""

    def __init__(self):
        logger.info("TradingStrategy v2.1 initialized")

    def apply_all_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all strategies to DataFrame."""
        logger.info("Applying all trading strategies (v2.1)...")

        df = self._ensure_columns(df)

        df = self.sma_crossover_strategy(df)
        df = self.rsi_strategy(df)
        df = self.macd_strategy(df)
        df = self.bollinger_strategy(df)
        df = self.combined_strategy(df)

        logger.info("✅ All strategies applied (v2.1)")
        return df

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make sure required indicator columns exist."""
        required = ["sma_20", "sma_50", "sma_200", "rsi_14",
                     "macd_line", "macd_signal", "bb_upper", "bb_lower",
                     "atr_14"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
        return df

    def _get_close(self, df):
        """Get the close price column."""
        for col in ["close", "Close"]:
            if col in df.columns:
                return col
        raise KeyError("No close column found")

    def _apply_cooldown(self, signals: pd.Series, cooldown: int = 10) -> pd.Series:
        """
        Enforce minimum bars between signals.
        Kills whipsaw and reduces commission drag.
        """
        filtered = signals.copy()
        last_signal_idx = -cooldown - 1

        for i in range(len(filtered)):
            if filtered.iloc[i] != 0:
                if (i - last_signal_idx) < cooldown:
                    filtered.iloc[i] = 0
                else:
                    last_signal_idx = i

        return filtered

    def _trend_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns True where we're in an UPTREND.
        Uptrend = close > SMA 200
        """
        close_col = self._get_close(df)
        if "sma_200" in df.columns:
            return df[close_col] > df["sma_200"]
        return pd.Series(True, index=df.index)

    def _trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns trend strength:
        Strong uptrend: close > SMA50 > SMA200
        Weak uptrend: close > SMA200 but < SMA50
        Downtrend: close < SMA200
        """
        close_col = self._get_close(df)
        close = df[close_col]

        strength = pd.Series(0, index=df.index)

        if "sma_200" in df.columns:
            strength[close > df["sma_200"]] = 1  # Uptrend
        if "sma_50" in df.columns and "sma_200" in df.columns:
            strength[(close > df["sma_50"]) & (df["sma_50"] > df["sma_200"])] = 2  # Strong uptrend

        return strength

    # ==============================================
    # STRATEGY 1: SMA CROSSOVER
    # ==============================================
    def sma_crossover_strategy(self, df: pd.DataFrame,
                                fast: int = 20, slow: int = 50,
                                cooldown: int = 10) -> pd.DataFrame:
        """
        SMA Crossover with trend filter.
        Buy: Golden cross in uptrend
        Sell: Death cross OR price drops below SMA200 (trend breakdown)
        """
        close_col = self._get_close(df)
        fast_col = f"sma_{fast}"
        slow_col = f"sma_{slow}"

        if fast_col not in df.columns or slow_col not in df.columns:
            df["sma_signal"] = 0
            return df

        uptrend = self._trend_filter(df)

        fast_above = df[fast_col] > df[slow_col]
        fast_above_prev = fast_above.shift(1, fill_value=False).astype(bool)

        raw_signal = pd.Series(0, index=df.index)

        # BUY: Golden cross in uptrend
        golden_cross = fast_above & ~fast_above_prev
        raw_signal[golden_cross & uptrend] = 1

        # SELL: Death cross only (not every dip)
        death_cross = ~fast_above & fast_above_prev
        raw_signal[death_cross] = -1

        df["sma_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["sma_signal"] == 1).sum()
        sells = (df["sma_signal"] == -1).sum()
        logger.info(f"📊 SMA Crossover v2.1 ({fast}/{slow}): {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 2: RSI (trend-aware dip buying)
    # ==============================================
    def rsi_strategy(self, df: pd.DataFrame,
                     cooldown: int = 15) -> pd.DataFrame:
        """
        RSI — Buy dips in uptrend, protect in downtrend.
        Uptrend: Buy when RSI 35-50 and rising (dip recovery)
        Downtrend: Sell when RSI > 55 (bear rally fades)
        Only sell in uptrend on extreme weakness (RSI < 25)
        """
        if "rsi_14" not in df.columns:
            df["rsi_signal"] = 0
            return df

        uptrend = self._trend_filter(df)
        rsi = df["rsi_14"]
        rsi_prev = rsi.shift(1, fill_value=50)
        rsi_rising = rsi > rsi_prev

        raw_signal = pd.Series(0, index=df.index)

        # UPTREND: Buy the dip when RSI recovers from 35-50 zone
        raw_signal[uptrend & (rsi > 35) & (rsi < 50) & rsi_rising] = 1

        # UPTREND: Sell only on severe momentum collapse
        raw_signal[uptrend & (rsi < 25)] = -1

        # DOWNTREND: Sell on bear rally exhaustion
        raw_signal[~uptrend & (rsi > 55) & ~rsi_rising] = -1

        df["rsi_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["rsi_signal"] == 1).sum()
        sells = (df["rsi_signal"] == -1).sum()
        logger.info(f"📊 RSI Strategy v2.1 (trend-aware): {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 3: MACD (trend-filtered)
    # ==============================================
    def macd_strategy(self, df: pd.DataFrame,
                      cooldown: int = 12) -> pd.DataFrame:
        """
        MACD — Crossovers with trend and momentum confirmation.
        Buy: Bullish cross + uptrend
        Sell: Bearish cross + histogram deeply negative
        """
        if "macd_line" not in df.columns or "macd_signal" not in df.columns:
            df["macd_trade_signal"] = 0
            return df

        uptrend = self._trend_filter(df)
        macd_line = df["macd_line"]
        signal_line = df["macd_signal"]

        if "macd_histogram" in df.columns:
            histogram = df["macd_histogram"]
        else:
            histogram = macd_line - signal_line

        macd_above = macd_line > signal_line
        macd_above_prev = macd_above.shift(1, fill_value=False).astype(bool)

        raw_signal = pd.Series(0, index=df.index)

        # BUY: Bullish crossover in uptrend
        bullish_cross = macd_above & ~macd_above_prev
        raw_signal[bullish_cross & uptrend] = 1

        # SELL: Bearish crossover only when histogram is significantly negative
        # Use rolling std to determine "significant"
        hist_std = histogram.rolling(50).std().fillna(histogram.std())
        bearish_cross = ~macd_above & macd_above_prev
        significant_bearish = histogram < (-0.5 * hist_std)
        raw_signal[bearish_cross & significant_bearish] = -1

        df["macd_trade_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["macd_trade_signal"] == 1).sum()
        sells = (df["macd_trade_signal"] == -1).sum()
        logger.info(f"📊 MACD Strategy v2.1: {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 4: BOLLINGER BANDS (FIXED - relaxed)
    # ==============================================
    def bollinger_strategy(self, df: pd.DataFrame,
                           cooldown: int = 12) -> pd.DataFrame:
        """
        Bollinger Bands — FIXED from v2.0
        OLD v2.0: Required price to TOUCH lower band (almost never happens in uptrend)
        NEW v2.1: Buy when price is in LOWER 20% of bands and rising
                  Sell when price breaks down through lower band in uptrend
                  Sell when price touches upper band in downtrend
        """
        close_col = self._get_close(df)

        if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
            df["bb_signal"] = 0
            return df

        uptrend = self._trend_filter(df)
        close = df[close_col]
        close_prev = close.shift(1)

        # Calculate where price is within the bands (0 = lower, 1 = upper)
        if "bb_percent_b" in df.columns:
            pct_b = df["bb_percent_b"]
        else:
            band_width = df["bb_upper"] - df["bb_lower"]
            band_width = band_width.replace(0, np.nan)
            pct_b = (close - df["bb_lower"]) / band_width

        price_rising = close > close_prev

        raw_signal = pd.Series(0, index=df.index)

        # UPTREND: Buy when price is in lower 25% of bands and bouncing up
        # This catches dips without requiring price to touch the band
        raw_signal[uptrend & (pct_b < 0.25) & (pct_b > 0) & price_rising] = 1

        # UPTREND: Also buy if price crosses back above lower band (recovery)
        crossed_above_lower = (close_prev < df["bb_lower"]) & (close > df["bb_lower"])
        raw_signal[uptrend & crossed_above_lower] = 1

        # UPTREND: Sell only if price crashes below lower band (breakdown)
        breakdown = (close < df["bb_lower"]) & (close_prev >= df["bb_lower"])
        raw_signal[uptrend & breakdown] = -1

        # DOWNTREND: Sell when price reaches upper band
        raw_signal[~uptrend & (pct_b > 0.9)] = -1

        df["bb_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["bb_signal"] == 1).sum()
        sells = (df["bb_signal"] == -1).sum()
        logger.info(f"📊 Bollinger Strategy v2.1: {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 5: COMBINED (FIXED - lookback window)
    # ==============================================
    def combined_strategy(self, df: pd.DataFrame,
                          min_confirmations: int = 2,
                          lookback: int = 5,
                          cooldown: int = 15) -> pd.DataFrame:
        """
        Combined Strategy — FIXED from v2.0
        OLD v2.0: Required 3+ strategies on SAME bar (almost never happens)
        NEW v2.1: Uses LOOKBACK WINDOW — if 2+ strategies fired BUY
                  within last 5 bars, that counts as confirmation.
                  This is more realistic since indicators fire at slightly
                  different times.
        """
        close_col = self._get_close(df)
        uptrend = self._trend_filter(df)
        trend_strength = self._trend_strength(df)

        signal_cols = ["sma_signal", "rsi_signal", "macd_trade_signal", "bb_signal"]
        existing = [col for col in signal_cols if col in df.columns]

        if not existing:
            df["combined_signal"] = 0
            return df

        # For each bar, count how many strategies fired BUY in the last N bars
        buy_recent = pd.Series(0, index=df.index, dtype=int)
        sell_recent = pd.Series(0, index=df.index, dtype=int)

        for col in existing:
            # Rolling window: was there a buy signal in the last `lookback` bars?
            buy_in_window = (df[col] == 1).rolling(lookback, min_periods=1).sum()
            sell_in_window = (df[col] == -1).rolling(lookback, min_periods=1).sum()
            buy_recent += (buy_in_window > 0).astype(int)
            sell_recent += (sell_in_window > 0).astype(int)

        # Momentum: price higher than 5 days ago
        momentum = df[close_col] > df[close_col].shift(5)

        raw_signal = pd.Series(0, index=df.index)

        # BUY: 2+ strategies agree within lookback window + uptrend + momentum
        raw_signal[(buy_recent >= min_confirmations) & uptrend & momentum] = 1

        # STRONGER BUY: In strong uptrend, even 2 confirms without momentum is OK
        raw_signal[(buy_recent >= min_confirmations) & (trend_strength >= 2)] = 1

        # SELL: 2+ strategies agree on sell AND downtrend
        raw_signal[(sell_recent >= min_confirmations) & ~uptrend] = -1

        # SELL: 3+ strategies agree on sell (regardless of trend — strong signal)
        raw_signal[(sell_recent >= 3)] = -1

        df["combined_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["combined_signal"] == 1).sum()
        sells = (df["combined_signal"] == -1).sum()
        logger.info(f"📊 Combined Strategy v2.1 ({min_confirmations}+ in {lookback}-bar window): {buys} BUYs, {sells} SELLs")

        return df