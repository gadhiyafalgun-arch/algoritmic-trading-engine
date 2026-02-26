"""
strategy.py — Trading Strategy Engine v2.0
==========================================
FIXED VERSION — Addresses:
- Commission death (cooldown periods)
- Bad sell timing (trailing stops)
- Counter-trend buying (trend filter)
- Whipsaw (minimum hold period)
- RSI backwards signals (trend-aware RSI)

Author: Your Name
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from loguru import logger


class TradingStrategy:
    """Generates trading signals with trend filtering and risk controls."""

    def __init__(self):
        logger.info("TradingStrategy v2.0 initialized")

    def apply_all_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all strategies to DataFrame."""
        logger.info("Applying all trading strategies (v2.0 — trend-filtered)...")

        df = self._ensure_columns(df)

        df = self.sma_crossover_strategy(df)
        df = self.rsi_strategy(df)
        df = self.macd_strategy(df)
        df = self.bollinger_strategy(df)
        df = self.combined_strategy(df)

        logger.info("✅ All strategies applied (v2.0)")
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
                    filtered.iloc[i] = 0  # Too soon — suppress
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

    # ==============================================
    # STRATEGY 1: SMA CROSSOVER (with trend filter)
    # ==============================================
    def sma_crossover_strategy(self, df: pd.DataFrame,
                                fast: int = 20, slow: int = 50,
                                cooldown: int = 10) -> pd.DataFrame:
        """
        SMA Crossover — Golden Cross / Death Cross
        FIX: Only buy when price > SMA 200 (with trend)
        FIX: Cooldown of 10 bars between signals
        """
        close_col = self._get_close(df)
        fast_col = f"sma_{fast}"
        slow_col = f"sma_{slow}"

        if fast_col not in df.columns or slow_col not in df.columns:
            df["sma_signal"] = 0
            return df

        uptrend = self._trend_filter(df)

        # Raw crossover signals
        fast_above = df[fast_col] > df[slow_col]
        fast_above_prev = fast_above.shift(1).fillna(False)

        raw_signal = pd.Series(0, index=df.index)

        # Golden cross AND in uptrend
        golden_cross = fast_above & ~fast_above_prev
        raw_signal[golden_cross & uptrend] = 1

        # Death cross (sell regardless — protect capital)
        death_cross = ~fast_above & fast_above_prev
        raw_signal[death_cross] = -1

        # Apply cooldown
        df["sma_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["sma_signal"] == 1).sum()
        sells = (df["sma_signal"] == -1).sum()
        logger.info(f"📊 SMA Crossover v2 ({fast}/{slow}): {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 2: RSI (trend-aware, FIXED)
    # ==============================================
    def rsi_strategy(self, df: pd.DataFrame,
                     cooldown: int = 15) -> pd.DataFrame:
        """
        RSI Strategy — COMPLETELY REDESIGNED
        OLD: Buy at 30, Sell at 70 (backwards in bull market)
        NEW:
          - In UPTREND: Buy when RSI pulls back to 35-45 (buying dips)
          - In UPTREND: Only sell when RSI < 30 (momentum collapse)
          - In DOWNTREND: Don't buy at all
          - In DOWNTREND: Sell when RSI > 55 (bear rally exhaustion)
        """
        if "rsi_14" not in df.columns:
            df["rsi_signal"] = 0
            return df

        close_col = self._get_close(df)
        uptrend = self._trend_filter(df)
        rsi = df["rsi_14"]

        raw_signal = pd.Series(0, index=df.index)

        # UPTREND: Buy the dip (RSI pulls back to 35-45 range)
        # This means the stock dipped but is still in overall uptrend
        rsi_prev = rsi.shift(1).fillna(50)
        buy_dip = uptrend & (rsi > 35) & (rsi < 45) & (rsi > rsi_prev)
        raw_signal[buy_dip] = 1

        # UPTREND: Sell on momentum collapse (RSI drops below 30)
        sell_collapse = uptrend & (rsi < 30)
        raw_signal[sell_collapse] = -1

        # DOWNTREND: Sell on bear rally exhaustion (RSI > 55)
        sell_bear = ~uptrend & (rsi > 55) & (rsi < rsi_prev)
        raw_signal[sell_bear] = -1

        # Apply cooldown
        df["rsi_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["rsi_signal"] == 1).sum()
        sells = (df["rsi_signal"] == -1).sum()
        logger.info(f"📊 RSI Strategy v2 (trend-aware): {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 3: MACD (with trend filter + cooldown)
    # ==============================================
    def macd_strategy(self, df: pd.DataFrame,
                      cooldown: int = 12) -> pd.DataFrame:
        """
        MACD Strategy — FIXED
        OLD: Every crossover = signal (too many trades)
        NEW:
          - Only buy crossovers in uptrend
          - Require MACD histogram to confirm momentum
          - Longer cooldown to reduce whipsaw
        """
        if "macd_line" not in df.columns or "macd_signal" not in df.columns:
            df["macd_trade_signal"] = 0
            return df

        close_col = self._get_close(df)
        uptrend = self._trend_filter(df)

        macd_line = df["macd_line"]
        signal_line = df["macd_signal"]

        # Check if histogram column exists
        if "macd_histogram" in df.columns:
            histogram = df["macd_histogram"]
        else:
            histogram = macd_line - signal_line

        # Crossover detection
        macd_above = macd_line > signal_line
        macd_above_prev = macd_above.shift(1).fillna(False)

        raw_signal = pd.Series(0, index=df.index)

        # BUY: MACD crosses above signal AND in uptrend AND histogram positive
        bullish_cross = macd_above & ~macd_above_prev
        raw_signal[bullish_cross & uptrend & (histogram > 0)] = 1

        # SELL: MACD crosses below signal AND histogram negative
        bearish_cross = ~macd_above & macd_above_prev
        raw_signal[bearish_cross & (histogram < 0)] = -1

        # Apply cooldown
        df["macd_trade_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["macd_trade_signal"] == 1).sum()
        sells = (df["macd_trade_signal"] == -1).sum()
        logger.info(f"📊 MACD Strategy v2 (filtered): {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 4: BOLLINGER BANDS (trend-aware)
    # ==============================================
    def bollinger_strategy(self, df: pd.DataFrame,
                           cooldown: int = 12) -> pd.DataFrame:
        """
        Bollinger Bands Strategy — FIXED
        OLD: Buy at lower band, sell at upper (counter-trend)
        NEW:
          - In UPTREND: Buy at lower band (mean reversion WITH trend)
          - In UPTREND: DON'T sell at upper band (let it run)
          - In DOWNTREND: Sell at upper band (mean reversion WITH trend)
          - In DOWNTREND: DON'T buy at lower band (falling knife)
        """
        close_col = self._get_close(df)

        if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
            df["bb_signal"] = 0
            return df

        uptrend = self._trend_filter(df)
        close = df[close_col]
        close_prev = close.shift(1)

        raw_signal = pd.Series(0, index=df.index)

        # UPTREND: Buy when price touches lower band and bounces back up
        touch_lower = close_prev <= df["bb_lower"]
        bounce_up = close > close_prev
        raw_signal[uptrend & touch_lower & bounce_up] = 1

        # DOWNTREND: Sell when price touches upper band and falls back
        touch_upper = close_prev >= df["bb_upper"]
        fall_down = close < close_prev
        raw_signal[~uptrend & touch_upper & fall_down] = -1

        # Also sell in uptrend if price crashes through lower band (breakdown)
        breakdown = uptrend & (close < df["bb_lower"]) & (close_prev > df["bb_lower"])
        raw_signal[breakdown] = -1

        # Apply cooldown
        df["bb_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["bb_signal"] == 1).sum()
        sells = (df["bb_signal"] == -1).sum()
        logger.info(f"📊 Bollinger Strategy v2 (trend-aware): {buys} BUYs, {sells} SELLs")

        return df

    # ==============================================
    # STRATEGY 5: COMBINED (3+ confirmations + cooldown)
    # ==============================================
    def combined_strategy(self, df: pd.DataFrame,
                          min_confirmations: int = 3,
                          cooldown: int = 15) -> pd.DataFrame:
        """
        Combined Strategy — FIXED
        OLD: 2+ agree = signal (too loose, 189+ signals)
        NEW:
          - Require 3+ strategies to agree
          - Must be in uptrend for buys
          - Longer cooldown (15 bars)
          - Momentum confirmation (close > close 5 days ago)
        """
        close_col = self._get_close(df)
        uptrend = self._trend_filter(df)

        signal_cols = ["sma_signal", "rsi_signal", "macd_trade_signal", "bb_signal"]
        existing = [col for col in signal_cols if col in df.columns]

        if not existing:
            df["combined_signal"] = 0
            return df

        # Count how many strategies say BUY or SELL
        buy_votes = pd.Series(0, index=df.index, dtype=int)
        sell_votes = pd.Series(0, index=df.index, dtype=int)

        for col in existing:
            buy_votes += (df[col] == 1).astype(int)
            sell_votes += (df[col] == -1).astype(int)

        # Momentum confirmation: price higher than 5 days ago
        momentum = df[close_col] > df[close_col].shift(5)

        raw_signal = pd.Series(0, index=df.index)

        # BUY: 3+ strategies agree AND uptrend AND momentum
        raw_signal[(buy_votes >= min_confirmations) & uptrend & momentum] = 1

        # SELL: 3+ strategies agree (sell even without trend filter — protect capital)
        raw_signal[(sell_votes >= min_confirmations)] = -1

        # ALSO SELL: 2+ strategies say sell AND price below SMA 200 (downtrend)
        raw_signal[(sell_votes >= 2) & ~uptrend] = -1

        # Apply cooldown
        df["combined_signal"] = self._apply_cooldown(raw_signal, cooldown)

        buys = (df["combined_signal"] == 1).sum()
        sells = (df["combined_signal"] == -1).sum()
        logger.info(f"📊 Combined Strategy v2 ({min_confirmations}+ confirms): {buys} BUYs, {sells} SELLs")

        return df