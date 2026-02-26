"""
diagnostics.py — Strategy Diagnostic Tool
Figures out WHY strategies are failing
"""

import pandas as pd
import numpy as np
from loguru import logger


class StrategyDiagnostics:
    """Analyzes strategy performance to find failure points."""

    def __init__(self):
        self.issues_found = []

    def _get_close_col(self, df):
        """Find the close price column name."""
        for col in ["close", "Close", "CLOSE", "adj_close", "Adj Close"]:
            if col in df.columns:
                return col
        raise KeyError(f"No close column found. Columns: {list(df.columns)}")

    def _get_high_col(self, df):
        for col in ["high", "High", "HIGH"]:
            if col in df.columns:
                return col
        return None

    def _get_low_col(self, df):
        for col in ["low", "Low", "LOW"]:
            if col in df.columns:
                return col
        return None

    def run_full_diagnosis(self, df, signals_col, symbol="UNKNOWN"):
        logger.info(f"Running diagnostics on {symbol} — {signals_col}")
        self.issues_found = []
        results = {}

        results["signal_freq"] = self._analyze_signal_frequency(df, signals_col)
        results["signal_timing"] = self._analyze_signal_timing(df, signals_col)
        results["market_regime"] = self._analyze_market_regime(df)
        results["trend_alignment"] = self._analyze_trend_alignment(df, signals_col)
        results["whipsaw"] = self._analyze_whipsaw(df, signals_col)
        results["volatility"] = self._analyze_volatility_at_signals(df, signals_col)
        results["commission_drag"] = self._estimate_commission_drag(df, signals_col)

        self._print_report(symbol, signals_col, results)
        return results

    def _analyze_signal_frequency(self, df, signals_col):
        total_bars = len(df)
        buys = (df[signals_col] == 1).sum()
        sells = (df[signals_col] == -1).sum()
        holds = (df[signals_col] == 0).sum()
        total_signals = buys + sells
        avg_bars_between = total_bars / max(total_signals, 1)

        result = {
            "total_bars": total_bars,
            "buy_signals": int(buys),
            "sell_signals": int(sells),
            "hold_signals": int(holds),
            "total_signals": int(total_signals),
            "signal_rate": round(total_signals / total_bars * 100, 2),
            "avg_bars_between_signals": round(avg_bars_between, 1),
        }

        if total_signals > total_bars * 0.3:
            self.issues_found.append("OVER-TRADING: Signals on >30% of bars — commission death")
        if total_signals < 10:
            self.issues_found.append("UNDER-TRADING: Less than 10 total signals")
        if buys > 0 and sells > 0 and abs(buys - sells) > max(buys, sells) * 0.5:
            self.issues_found.append("IMBALANCED: Buy/Sell signals very uneven")

        return result

    def _analyze_signal_timing(self, df, signals_col):
        close_col = self._get_close_col(df)
        buy_mask = df[signals_col] == 1
        sell_mask = df[signals_col] == -1
        result = {}

        for lookahead in [1, 5, 10, 20]:
            future_return = df[close_col].pct_change(lookahead).shift(-lookahead)

            if buy_mask.sum() > 0:
                result[f"avg_{lookahead}d_return_after_buy"] = round(future_return[buy_mask].mean() * 100, 3)
            else:
                result[f"avg_{lookahead}d_return_after_buy"] = None

            if sell_mask.sum() > 0:
                result[f"avg_{lookahead}d_return_after_sell"] = round(future_return[sell_mask].mean() * 100, 3)
            else:
                result[f"avg_{lookahead}d_return_after_sell"] = None

        if result.get("avg_5d_return_after_buy") is not None:
            if result["avg_5d_return_after_buy"] < 0:
                self.issues_found.append("BAD BUY TIMING: 5-day return after BUY is NEGATIVE")

        if result.get("avg_5d_return_after_sell") is not None:
            if result["avg_5d_return_after_sell"] > 0:
                self.issues_found.append("BAD SELL TIMING: 5-day return after SELL is POSITIVE")

        return result

    def _analyze_market_regime(self, df):
        close_col = self._get_close_col(df)
        total_return = (df[close_col].iloc[-1] / df[close_col].iloc[0] - 1) * 100

        # Find SMA 200 column
        sma200 = None
        for col in ["sma_200", "SMA_200"]:
            if col in df.columns:
                sma200 = df[col]
                break
        if sma200 is None:
            sma200 = df[close_col].rolling(200).mean()

        above_sma200 = (df[close_col] > sma200).sum()
        below_sma200 = (df[close_col] <= sma200).sum()
        valid_count = above_sma200 + below_sma200
        above_pct = above_sma200 / max(valid_count, 1) * 100

        if above_pct > 70:
            regime = "STRONG UPTREND"
        elif above_pct > 55:
            regime = "MILD UPTREND"
        elif above_pct > 45:
            regime = "SIDEWAYS/CHOPPY"
        elif above_pct > 30:
            regime = "MILD DOWNTREND"
        else:
            regime = "STRONG DOWNTREND"

        result = {
            "total_return_pct": round(total_return, 2),
            "pct_above_sma200": round(above_pct, 1),
            "regime": regime,
        }

        if regime == "SIDEWAYS/CHOPPY":
            self.issues_found.append("CHOPPY MARKET: Trend strategies will whipsaw")
        if total_return > 50:
            self.issues_found.append("BULL MARKET: Even buy-and-hold made money — strategy should beat it")

        return result

    def _analyze_trend_alignment(self, df, signals_col):
        close_col = self._get_close_col(df)

        sma200 = None
        for col in ["sma_200", "SMA_200"]:
            if col in df.columns:
                sma200 = df[col]
                break
        if sma200 is None:
            sma200 = df[close_col].rolling(200).mean()

        buy_mask = df[signals_col] == 1

        if buy_mask.sum() > 0:
            buys_with_trend = (buy_mask & (df[close_col] > sma200)).sum()
            buys_counter_trend = (buy_mask & (df[close_col] <= sma200)).sum()
            alignment = buys_with_trend / max(buy_mask.sum(), 1) * 100
        else:
            buys_with_trend = 0
            buys_counter_trend = 0
            alignment = 0

        result = {
            "buys_with_trend": int(buys_with_trend),
            "buys_counter_trend": int(buys_counter_trend),
            "buy_trend_alignment_pct": round(alignment, 1),
        }

        if alignment < 50:
            self.issues_found.append("COUNTER-TREND BUYING: Most buys happen AGAINST the trend")

        return result

    def _analyze_whipsaw(self, df, signals_col):
        signals = df[signals_col]
        non_zero = signals[signals != 0]

        if len(non_zero) < 2:
            return {"whipsaw_count": 0, "avg_bars_between_signals": 0, "whipsaw_rate": 0}

        signal_indices = non_zero.index.tolist()
        gaps = []
        flips = 0

        for i in range(1, len(signal_indices)):
            idx_curr = df.index.get_loc(signal_indices[i])
            idx_prev = df.index.get_loc(signal_indices[i - 1])
            gap = idx_curr - idx_prev
            gaps.append(gap)

            if gap <= 3 and non_zero.iloc[i] != non_zero.iloc[i - 1]:
                flips += 1

        avg_gap = np.mean(gaps) if gaps else 0

        result = {
            "whipsaw_count": flips,
            "avg_bars_between_signals": round(avg_gap, 1),
            "whipsaw_rate": round(flips / max(len(gaps), 1) * 100, 1),
        }

        if flips > len(gaps) * 0.2:
            self.issues_found.append("WHIPSAW HELL: >20% of signals are rapid flips")

        return result

    def _analyze_volatility_at_signals(self, df, signals_col):
        close_col = self._get_close_col(df)
        high_col = self._get_high_col(df)
        low_col = self._get_low_col(df)

        # Find ATR column
        atr = None
        for col in ["atr_14", "ATR_14"]:
            if col in df.columns:
                atr = df[col]
                break
        if atr is None and high_col and low_col:
            atr = (df[high_col] - df[low_col]).rolling(14).mean()
        elif atr is None:
            return {"overall_avg_atr_pct": 0, "buy_signal_avg_atr_pct": 0}

        atr_normalized = atr / df[close_col] * 100
        buy_mask = df[signals_col] == 1
        overall_avg = atr_normalized.mean()
        buy_avg = atr_normalized[buy_mask].mean() if buy_mask.sum() > 0 else 0

        result = {
            "overall_avg_atr_pct": round(float(overall_avg), 3),
            "buy_signal_avg_atr_pct": round(float(buy_avg), 3),
        }

        if buy_avg > overall_avg * 1.5:
            self.issues_found.append("HIGH VOL ENTRIES: Buying during high volatility")

        return result

    def _estimate_commission_drag(self, df, signals_col):
        total_signals = (df[signals_col] != 0).sum()
        round_trips = total_signals / 2
        cost_per_rt = 0.30
        total_drag = round_trips * cost_per_rt

        result = {
            "estimated_round_trips": int(round_trips),
            "cost_per_round_trip_pct": cost_per_rt,
            "total_commission_drag_pct": round(total_drag, 2),
        }

        if total_drag > 5:
            self.issues_found.append(f"COMMISSION DEATH: {total_drag:.1f}% lost to commissions")

        return result

    def _print_report(self, symbol, signals_col, results):
        print("\n" + "=" * 70)
        print(f"DIAGNOSTIC REPORT: {symbol} — {signals_col}")
        print("=" * 70)

        sf = results["signal_freq"]
        print(f"\n  SIGNAL FREQUENCY:")
        print(f"   Total Bars:          {sf['total_bars']}")
        print(f"   Buy Signals:         {sf['buy_signals']}")
        print(f"   Sell Signals:        {sf['sell_signals']}")
        print(f"   Signal Rate:         {sf['signal_rate']}%")
        print(f"   Avg Bars Between:    {sf['avg_bars_between_signals']}")

        st = results["signal_timing"]
        print(f"\n  SIGNAL TIMING (avg return AFTER signal):")
        for key, val in st.items():
            label = key.replace("avg_", "").replace("_return_after_", " after ")
            if val is not None:
                good_buy = val > 0 and "buy" in key
                good_sell = val < 0 and "sell" in key
                marker = "OK" if good_buy or good_sell else "BAD"
                print(f"   [{marker}] {label}: {val}%")

        mr = results["market_regime"]
        print(f"\n  MARKET REGIME:")
        print(f"   Total Return:        {mr['total_return_pct']}%")
        print(f"   Above SMA200:        {mr['pct_above_sma200']}%")
        print(f"   Regime:              {mr['regime']}")

        ta = results["trend_alignment"]
        print(f"\n  TREND ALIGNMENT:")
        print(f"   Buys WITH trend:     {ta['buys_with_trend']}")
        print(f"   Buys AGAINST trend:  {ta['buys_counter_trend']}")
        print(f"   Alignment:           {ta['buy_trend_alignment_pct']}%")

        ws = results["whipsaw"]
        print(f"\n  WHIPSAW DETECTION:")
        print(f"   Whipsaw Count:       {ws['whipsaw_count']}")
        print(f"   Whipsaw Rate:        {ws['whipsaw_rate']}%")

        vol = results["volatility"]
        print(f"\n  VOLATILITY AT ENTRIES:")
        print(f"   Overall Avg ATR%:    {vol['overall_avg_atr_pct']}%")
        print(f"   Buy Signal ATR%:     {vol['buy_signal_avg_atr_pct']}%")

        cd = results["commission_drag"]
        print(f"\n  COMMISSION DRAG:")
        print(f"   Round Trips:         {cd['estimated_round_trips']}")
        print(f"   Total Drag:          {cd['total_commission_drag_pct']}%")

        print(f"\n{'='*70}")
        print(f"  ISSUES FOUND: {len(self.issues_found)}")
        print(f"{'='*70}")
        if self.issues_found:
            for issue in self.issues_found:
                print(f"   >> {issue}")
        else:
            print("   No major issues detected")
        print("=" * 70)


def run_diagnostics_for_all(data_dict, strategy_columns):
    diag = StrategyDiagnostics()
    summary = []

    for symbol, df in data_dict.items():
        for col in strategy_columns:
            if col in df.columns:
                results = diag.run_full_diagnosis(df, col, symbol)
                summary.append({
                    "symbol": symbol,
                    "strategy": col,
                    "issues": len(diag.issues_found),
                    "signal_rate": results["signal_freq"]["signal_rate"],
                    "trend_align": results["trend_alignment"]["buy_trend_alignment_pct"],
                    "whipsaw": results["whipsaw"]["whipsaw_rate"],
                    "drag": results["commission_drag"]["total_commission_drag_pct"],
                    "5d_after_buy": results["signal_timing"].get("avg_5d_return_after_buy"),
                })

    if summary:
        print("\n" + "=" * 90)
        print("DIAGNOSTIC SUMMARY — ALL STOCKS x ALL STRATEGIES")
        print("=" * 90)
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        print("=" * 90)

    return summary