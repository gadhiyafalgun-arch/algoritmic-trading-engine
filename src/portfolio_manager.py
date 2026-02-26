"""
Portfolio Manager Module
=========================
Manages multi-stock portfolio with risk controls.

This runs the ENHANCED backtest that uses:
- Advanced position sizing
- Dynamic stop-losses
- Portfolio-level risk checks
- Correlation-aware allocation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger
import yaml

from src.risk_manager import RiskManager


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    highest_price: float  # For trailing stop
    sizing_method: str = "volatility"


@dataclass
class PortfolioTrade:
    """Completed trade record."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'SIGNAL', 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP'
    commission: float
    sizing_method: str


class PortfolioManager:
    """
    Advanced portfolio manager with risk controls.
    
    Differences from basic Backtester:
    - Manages MULTIPLE stocks simultaneously
    - Uses advanced position sizing
    - Dynamic trailing stop-losses
    - Portfolio-level risk checks before each trade
    - Correlation-aware decisions
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize portfolio manager."""

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.risk_manager = RiskManager(config_path)

        self.initial_capital = self.config["trading"]["initial_capital"]
        self.commission_rate = self.config["trading"]["commission"]
        self.slippage_rate = self.config["trading"]["slippage"]

        logger.info("PortfolioManager initialized")

    def run_portfolio_backtest(self, all_data: dict,
                                signal_column: str = "combined_signal") -> dict:
        """
        Run backtest across MULTIPLE stocks with portfolio management.
        
        Args:
            all_data: Dict of {symbol: DataFrame} with signals
            signal_column: Which signal to use
            
        Returns:
            Complete portfolio backtest results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🏃 Running Portfolio Backtest")
        logger.info(f"Strategy: {signal_column}")
        logger.info(f"Stocks: {list(all_data.keys())}")
        logger.info(f"{'='*60}")

        # --- Initialize ---
        cash = self.initial_capital
        peak_capital = self.initial_capital
        positions: Dict[str, Position] = {}
        completed_trades: List[PortfolioTrade] = []
        portfolio_history = []

        # Get all unique dates across all stocks
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        # Track win/loss for Kelly sizing
        running_wins = 0
        running_losses = 0
        running_win_amount = 0
        running_loss_amount = 0

        # --- Day by Day Simulation ---
        for date in all_dates:

            # ===== UPDATE EXISTING POSITIONS =====
            symbols_to_close = []

            for symbol, position in positions.items():
                if symbol not in all_data or date not in all_data[symbol].index:
                    continue

                row = all_data[symbol].loc[date]
                current_price = row["close"]
                high = row["high"]
                low = row["low"]
                atr = row.get("atr_14", current_price * 0.02)
                if pd.isna(atr) or atr <= 0:
                    atr = current_price * 0.02

                # Update highest price for trailing stop
                if current_price > position.highest_price:
                    position.highest_price = current_price

                # Update trailing stop
                new_trailing = self.risk_manager.calculate_trailing_stop(
                    current_price, position.highest_price, atr
                )
                position.stop_loss = max(position.stop_loss, new_trailing)

                # Check STOP LOSS
                if low <= position.stop_loss:
                    exit_price = position.stop_loss * (1 - self.slippage_rate)
                    commission = exit_price * position.shares * self.commission_rate
                    proceeds = (exit_price * position.shares) - commission
                    cash += proceeds

                    pnl = (exit_price - position.entry_price) * position.shares - commission
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price

                    completed_trades.append(PortfolioTrade(
                        symbol=symbol,
                        entry_date=position.entry_date,
                        entry_price=position.entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        shares=position.shares,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        exit_reason="TRAILING_STOP",
                        commission=commission,
                        sizing_method=position.sizing_method
                    ))

                    # Update running stats
                    if pnl > 0:
                        running_wins += 1
                        running_win_amount += pnl
                    else:
                        running_losses += 1
                        running_loss_amount += abs(pnl)

                    symbols_to_close.append(symbol)
                    logger.info(f"🛑 TRAILING STOP {symbol}: {date.strftime('%Y-%m-%d')} | "
                              f"PnL: ${pnl:.2f}")
                    continue

                # Check TAKE PROFIT
                if high >= position.take_profit:
                    exit_price = position.take_profit * (1 - self.slippage_rate)
                    commission = exit_price * position.shares * self.commission_rate
                    proceeds = (exit_price * position.shares) - commission
                    cash += proceeds

                    pnl = (exit_price - position.entry_price) * position.shares - commission
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price

                    completed_trades.append(PortfolioTrade(
                        symbol=symbol,
                        entry_date=position.entry_date,
                        entry_price=position.entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        shares=position.shares,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        exit_reason="TAKE_PROFIT",
                        commission=commission,
                        sizing_method=position.sizing_method
                    ))

                    if pnl > 0:
                        running_wins += 1
                        running_win_amount += pnl
                    else:
                        running_losses += 1
                        running_loss_amount += abs(pnl)

                    symbols_to_close.append(symbol)
                    logger.info(f"🎯 TAKE PROFIT {symbol}: {date.strftime('%Y-%m-%d')} | "
                              f"PnL: ${pnl:.2f}")
                    continue

                # Check SELL SIGNAL
                signal = row.get(signal_column, 0)
                if signal == -1:
                    exit_price = current_price * (1 - self.slippage_rate)
                    commission = exit_price * position.shares * self.commission_rate
                    proceeds = (exit_price * position.shares) - commission
                    cash += proceeds

                    pnl = (exit_price - position.entry_price) * position.shares - commission
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price

                    completed_trades.append(PortfolioTrade(
                        symbol=symbol,
                        entry_date=position.entry_date,
                        entry_price=position.entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        shares=position.shares,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        exit_reason="SIGNAL",
                        commission=commission,
                        sizing_method=position.sizing_method
                    ))

                    if pnl > 0:
                        running_wins += 1
                        running_win_amount += pnl
                    else:
                        running_losses += 1
                        running_loss_amount += abs(pnl)

                    symbols_to_close.append(symbol)
                    logger.info(f"🔴 SELL SIGNAL {symbol}: {date.strftime('%Y-%m-%d')} | "
                              f"PnL: ${pnl:.2f}")

            # Remove closed positions
            for s in symbols_to_close:
                del positions[s]

                        # ===== CHECK FOR NEW BUY SIGNALS =====
            
            # Do portfolio risk check ONCE per day (not per stock)
            holdings_value = sum(
                all_data[s].loc[date, "close"] * p.shares
                for s, p in positions.items()
                if date in all_data[s].index
            )
            total_value = cash + holdings_value
            peak_capital = max(peak_capital, total_value)

            risk_check = self.risk_manager.check_portfolio_risk(
                total_value, peak_capital, len(positions)
            )

            if not risk_check["can_trade"]:
                # Log ONCE per day, not per stock
                if not hasattr(self, '_last_halt_date') or self._last_halt_date != date:
                    for warning in risk_check["warnings"]:
                        logger.warning(f"⚠️ {warning}")
                    self._last_halt_date = date
            else:
                for symbol, df in all_data.items():
                    if date not in df.index:
                        continue
                    if symbol in positions:
                        continue

                    row = df.loc[date]
                    signal = row.get(signal_column, 0)

                    if signal != 1:
                        continue

                    # Calculate position size
                    entry_price = row["close"] * (1 + self.slippage_rate)
                    atr = row.get("atr_14", entry_price * 0.02)
                    if pd.isna(atr) or atr <= 0:
                        atr = entry_price * 0.02

                    shares = self.risk_manager.volatility_based_size(
                        cash, entry_price, atr
                    )

                    if shares <= 0:
                        continue

                    total_cost = shares * entry_price * (1 + self.commission_rate)
                    if total_cost > cash:
                        shares = int(cash / (entry_price * (1 + self.commission_rate)))
                        total_cost = shares * entry_price * (1 + self.commission_rate)

                    if shares <= 0:
                        continue

                    stop_loss = self.risk_manager.calculate_atr_stop_loss(
                        entry_price, atr, multiplier=2.0
                    )
                    take_profit = entry_price * (1 + self.risk_manager.take_profit_pct)

                    cash -= total_cost

                    positions[symbol] = Position(
                        symbol=symbol,
                        entry_date=date,
                        entry_price=entry_price,
                        shares=shares,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        highest_price=entry_price,
                        sizing_method="volatility"
                    )

                    logger.info(f"🟢 BUY {symbol}: {date.strftime('%Y-%m-%d')} | "
                              f"Price: ${entry_price:.2f} | Shares: {shares} | "
                              f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
                    

                # Calculate position size
                entry_price = row["close"] * (1 + self.slippage_rate)
                atr = row.get("atr_14", entry_price * 0.02)
                if pd.isna(atr) or atr <= 0:
                    atr = entry_price * 0.02

                # Use volatility-based sizing
                shares = self.risk_manager.volatility_based_size(
                    cash, entry_price, atr
                )

                if shares <= 0:
                    continue

                # Check affordability
                total_cost = shares * entry_price * (1 + self.commission_rate)
                if total_cost > cash:
                    shares = int(cash / (entry_price * (1 + self.commission_rate)))
                    total_cost = shares * entry_price * (1 + self.commission_rate)

                if shares <= 0:
                    continue

                # Calculate stops
                stop_loss = self.risk_manager.calculate_atr_stop_loss(
                    entry_price, atr, multiplier=2.0
                )
                take_profit = entry_price * (1 + self.risk_manager.take_profit_pct)

                # Execute buy
                cash -= total_cost

                positions[symbol] = Position(
                    symbol=symbol,
                    entry_date=date,
                    entry_price=entry_price,
                    shares=shares,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    highest_price=entry_price,
                    sizing_method="volatility"
                )

                logger.info(f"🟢 BUY {symbol}: {date.strftime('%Y-%m-%d')} | "
                          f"Price: ${entry_price:.2f} | Shares: {shares} | "
                          f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")

            # ===== RECORD DAILY PORTFOLIO STATE =====
            holdings_value = 0
            for s, p in positions.items():
                if date in all_data[s].index:
                    holdings_value += all_data[s].loc[date, "close"] * p.shares

            total_value = cash + holdings_value
            peak_capital = max(peak_capital, total_value)

            portfolio_history.append({
                "date": date,
                "cash": cash,
                "holdings_value": holdings_value,
                "total_value": total_value,
                "num_positions": len(positions),
                "drawdown": (peak_capital - total_value) / peak_capital if peak_capital > 0 else 0
            })

        # --- Close remaining positions ---
        for symbol, position in positions.items():
            last_date = all_data[symbol].index[-1]
            last_price = all_data[symbol]["close"].iloc[-1]
            exit_price = last_price * (1 - self.slippage_rate)
            commission = exit_price * position.shares * self.commission_rate
            cash += (exit_price * position.shares) - commission

            pnl = (exit_price - position.entry_price) * position.shares - commission
            pnl_pct = (exit_price - position.entry_price) / position.entry_price

            completed_trades.append(PortfolioTrade(
                symbol=symbol,
                entry_date=position.entry_date,
                entry_price=position.entry_price,
                exit_date=last_date,
                exit_price=exit_price,
                shares=position.shares,
                pnl=pnl,
                pnl_percent=pnl_pct,
                exit_reason="END_OF_BACKTEST",
                commission=commission,
                sizing_method=position.sizing_method
            ))

        # --- Build Results ---
        portfolio_df = pd.DataFrame(portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index("date", inplace=True)
            portfolio_df["daily_return"] = portfolio_df["total_value"].pct_change().fillna(0)

        trades_df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_percent": t.pnl_percent,
                "exit_reason": t.exit_reason,
                "commission": t.commission,
                "sizing_method": t.sizing_method
            }
            for t in completed_trades
        ])

        final_value = cash
        total_return = (final_value - self.initial_capital) / self.initial_capital

        results = {
            "portfolio_history": portfolio_df,
            "trades": trades_df,
            "final_value": final_value,
            "initial_capital": self.initial_capital,
            "total_return": total_return,
            "total_trades": len(completed_trades),
            "strategy": signal_column
        }

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PORTFOLIO BACKTEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Initial Capital:  ${self.initial_capital:>12,.2f}")
        logger.info(f"Final Value:      ${final_value:>12,.2f}")
        logger.info(f"Total Return:     {total_return*100:>11.2f}%")
        logger.info(f"Total Trades:     {len(completed_trades):>12}")

        if trades_df is not None and not trades_df.empty:
            # Exit reason breakdown
            logger.info(f"\n📋 Exit Reasons:")
            for reason in trades_df["exit_reason"].unique():
                count = (trades_df["exit_reason"] == reason).sum()
                logger.info(f"   {reason}: {count}")

            # Per-stock breakdown
            logger.info(f"\n📋 Per-Stock P&L:")
            for symbol in trades_df["symbol"].unique():
                stock_trades = trades_df[trades_df["symbol"] == symbol]
                stock_pnl = stock_trades["pnl"].sum()
                logger.info(f"   {symbol}: ${stock_pnl:,.2f} ({len(stock_trades)} trades)")

        return results