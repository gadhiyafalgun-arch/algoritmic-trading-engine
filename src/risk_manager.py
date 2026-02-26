"""
Advanced Risk Management Module
=================================
Controls risk at both TRADE level and PORTFOLIO level.

This is the SHIELD of your trading engine.
Without this, even the best strategy will blow up your account.

Features:
- Position Sizing (Fixed, Kelly, Volatility-based)
- Dynamic Stop-Loss (ATR-based trailing stop)
- Portfolio Risk Controls
- Correlation Analysis
- Drawdown Protection
- Risk Scoring
"""

import pandas as pd
import numpy as np
from loguru import logger
import yaml


class RiskManager:
    """
    Manages all risk-related decisions.
    
    Philosophy:
        "Risk management is not about avoiding risk.
         It's about understanding and controlling it."
    
    Key Rules:
        1. Never risk more than X% per trade
        2. Never have more than Y% in one stock
        3. Stop trading if drawdown exceeds Z%
        4. Diversify — watch correlations
        5. Size positions based on volatility
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize risk manager with configuration."""

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        trading = self.config["trading"]
        risk = self.config["risk"]

        # Capital
        self.initial_capital = trading["initial_capital"]

        # Trade-level risk
        self.risk_per_trade = trading["risk_per_trade"]      # 2%
        self.max_position_size = trading["max_position_size"]  # 20%
        self.commission_rate = trading["commission"]
        self.slippage_rate = trading["slippage"]

        # Portfolio-level risk
        self.stop_loss_pct = risk["stop_loss"]               # 5%
        self.take_profit_pct = risk["take_profit"]            # 10%
        self.max_drawdown_limit = risk["max_drawdown"]        # 15%
        self.max_open_positions = risk["max_open_positions"]   # 5

        logger.info("RiskManager initialized")

    # ==========================================
    # POSITION SIZING METHODS
    # ==========================================

    def fixed_fractional_size(self, capital: float, risk_per_trade: float,
                               entry_price: float, stop_loss_price: float) -> int:
        """
        Fixed Fractional Position Sizing
        
        Logic:
            Risk a fixed % of capital per trade.
            Size depends on how far your stop-loss is.
            
        Example:
            Capital: \$100,000
            Risk per trade: 2% = \$2,000
            Entry: \$150
            Stop Loss: \$142.50 (5% below)
            Risk per share: \$150 - \$142.50 = \$7.50
            Shares: \$2,000 / \$7.50 = 266 shares
            
        This is the MOST COMMON professional method.
        """
        if entry_price <= 0 or stop_loss_price <= 0 or pd.isna(entry_price) or pd.isna(stop_loss_price):
            return 0

        # Amount we're willing to lose
        risk_amount = capital * risk_per_trade

        # Risk per share (how much we lose if stop-loss hits)
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share <= 0:
            return 0

        # Calculate shares
        shares = int(risk_amount / risk_per_share)

        # Make sure we don't exceed max position size
        max_shares = int((capital * self.max_position_size) / entry_price)
        shares = min(shares, max_shares)

        # Make sure we can actually afford it
        total_cost = shares * entry_price * (1 + self.commission_rate + self.slippage_rate)
        if total_cost > capital:
            shares = int(capital / (entry_price * (1 + self.commission_rate + self.slippage_rate)))

        logger.debug(f"Fixed Fractional: {shares} shares "
                    f"(risk ${risk_amount:.0f}, ${risk_per_share:.2f}/share)")

        return max(0, shares)

    def kelly_criterion_size(self, capital: float, entry_price: float,
                              win_rate: float, avg_win: float,
                              avg_loss: float) -> int:
        """
        Kelly Criterion Position Sizing
        
        Logic:
            Mathematically optimal bet size to maximize long-term growth.
            
        Formula:
            Kelly % = W - [(1-W) / R]
            Where:
                W = Win rate (probability of winning)
                R = Win/Loss ratio (avg win / avg loss)
                
        WARNING:
            Full Kelly is VERY aggressive.
            Most professionals use Half-Kelly or Quarter-Kelly.
            
        We use QUARTER Kelly for safety.
        """
        if avg_loss == 0 or win_rate <= 0 or entry_price <= 0 or pd.isna(avg_loss) or pd.isna(win_rate):
            return 0

        # Win/Loss ratio
        win_loss_ratio = abs(avg_win / avg_loss)

        # Kelly percentage
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Use Quarter Kelly for safety
        kelly_pct = kelly_pct * 0.25

        # Don't go negative or too high
        kelly_pct = max(0, min(kelly_pct, self.max_position_size))

        # Calculate shares
        position_value = capital * kelly_pct
        shares = int(position_value / entry_price)

        logger.debug(f"Kelly Criterion: {kelly_pct*100:.1f}% = {shares} shares "
                    f"(W={win_rate:.1%}, R={win_loss_ratio:.2f})")

        return max(0, shares)

    def volatility_based_size(self, capital: float, entry_price: float,
                               atr: float, atr_multiplier: float = 2.0) -> int:
        """
        Volatility-Based Position Sizing (ATR Method)
        
        Logic:
            Size positions inversely to volatility.
            More volatile stock → smaller position.
            Less volatile stock → larger position.
            
        This is what PROFESSIONAL FUNDS use.
            
        Formula:
            Risk Amount = Capital × Risk Per Trade
            Stop Distance = ATR × Multiplier
            Shares = Risk Amount / Stop Distance
        """
        # Handle NaN or invalid values
        if pd.isna(atr) or atr <= 0 or entry_price <= 0 or capital <= 0:
            return 0

        # Risk amount
        risk_amount = capital * self.risk_per_trade

        # Stop distance based on ATR
        stop_distance = atr * atr_multiplier

        if stop_distance <= 0 or pd.isna(stop_distance):
            return 0

        # Shares
        shares = int(risk_amount / stop_distance)

        # Cap at max position size
        max_shares = int((capital * self.max_position_size) / entry_price)
        shares = min(shares, max_shares)

        # Affordability check
        total_cost = shares * entry_price * (1 + self.commission_rate + self.slippage_rate)
        if total_cost > capital:
            shares = int(capital / (entry_price * (1 + self.commission_rate + self.slippage_rate)))

        logger.debug(f"Volatility-Based: {shares} shares "
                    f"(ATR={atr:.2f}, stop_dist={stop_distance:.2f})")

        return max(0, shares)

    # ==========================================
    # DYNAMIC STOP-LOSS METHODS
    # ==========================================

    def calculate_atr_stop_loss(self, entry_price: float, atr: float,
                                 multiplier: float = 2.0,
                                 direction: str = "LONG") -> float:
        """
        ATR-Based Stop Loss
        
        Logic:
            Sets stop-loss based on market volatility.
            Volatile markets get wider stops.
            Calm markets get tighter stops.
            
        Much better than fixed percentage stops!
        """
        if direction == "LONG":
            stop_price = entry_price - (atr * multiplier)
        else:
            stop_price = entry_price + (atr * multiplier)

        return max(0, stop_price)

    def calculate_trailing_stop(self, current_price: float, highest_since_entry: float,
                                 atr: float, multiplier: float = 2.5) -> float:
        """
        ATR-Based Trailing Stop Loss
        
        Logic:
            Stop-loss MOVES UP as price goes up.
            But NEVER moves down.
            
            This lets you:
            - Lock in profits as price rises
            - Still give the trade room to breathe
            
        This is the GOLD STANDARD of stop-losses.
        """
        trailing_stop = highest_since_entry - (atr * multiplier)
        return max(0, trailing_stop)

    def calculate_chandelier_exit(self, df: pd.DataFrame,
                                   period: int = 22,
                                   multiplier: float = 3.0) -> pd.Series:
        """
        Chandelier Exit
        
        Logic:
            Trailing stop based on highest high minus ATR.
            Popular professional exit strategy.
            
        Formula:
            Chandelier Exit = Highest High (N periods) - ATR × Multiplier
        """
        highest_high = df["high"].rolling(window=period).max()
        atr = df["atr_14"] if "atr_14" in df.columns else self._calculate_atr(df, 14)

        chandelier = highest_high - (atr * multiplier)
        return chandelier

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Helper to calculate ATR if not already in DataFrame."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(com=period - 1, min_periods=period).mean()

        return atr

    # ==========================================
    # PORTFOLIO-LEVEL RISK CONTROLS
    # ==========================================

    def check_portfolio_risk(self, current_capital: float,
                              peak_capital: float,
                              open_positions: int) -> dict:
        """
        Check portfolio-level risk limits.
        
        Returns:
            Dictionary with risk status and warnings.
        """
        risk_status = {
            "can_trade": True,
            "warnings": [],
            "current_drawdown": 0,
            "positions_available": 0,
            "risk_level": "LOW"
        }

        # Check drawdown
        if peak_capital > 0:
            current_drawdown = (peak_capital - current_capital) / peak_capital
            risk_status["current_drawdown"] = current_drawdown

            if current_drawdown >= self.max_drawdown_limit:
                risk_status["can_trade"] = False
                risk_status["warnings"].append(
                    f"HALT: Max drawdown reached ({current_drawdown*100:.1f}% >= {self.max_drawdown_limit*100:.1f}%)"
                )
                risk_status["risk_level"] = "CRITICAL"

            elif current_drawdown >= self.max_drawdown_limit * 0.75:
                risk_status["warnings"].append(
                    f"WARNING: Approaching max drawdown ({current_drawdown*100:.1f}%)"
                )
                risk_status["risk_level"] = "HIGH"

            elif current_drawdown >= self.max_drawdown_limit * 0.5:
                risk_status["risk_level"] = "MEDIUM"

        # Check position limits
        positions_available = self.max_open_positions - open_positions
        risk_status["positions_available"] = positions_available

        if positions_available <= 0:
            risk_status["can_trade"] = False
            risk_status["warnings"].append(
                f"HALT: Max positions reached ({open_positions}/{self.max_open_positions})"
            )

        # Capital check
        min_trade_cost = current_capital * 0.01  # Need at least 1% for a trade
        if current_capital < min_trade_cost:
            risk_status["can_trade"] = False
            risk_status["warnings"].append("HALT: Insufficient capital")

        return risk_status

    def calculate_portfolio_var(self, returns_dict: dict,
                                 weights: dict = None,
                                 confidence: float = 0.95) -> dict:
        """
        Portfolio Value at Risk (VaR)
        
        What it tells you:
            "With 95% confidence, the portfolio won't lose more than X% in a day"
            
        Methods:
            1. Historical VaR — uses actual past returns
            2. Parametric VaR — assumes normal distribution
        """
        # Combine all stock returns into one DataFrame
        returns_df = pd.DataFrame(returns_dict)

        if returns_df.empty:
            return {"historical_var": 0, "parametric_var": 0}

        # Equal weights if not specified
        if weights is None:
            n_stocks = len(returns_df.columns)
            weights = {col: 1.0 / n_stocks for col in returns_df.columns}

        # Portfolio returns (weighted)
        weight_array = np.array([weights.get(col, 0) for col in returns_df.columns])
        portfolio_returns = returns_df.values @ weight_array

        # Method 1: Historical VaR
        historical_var = np.percentile(portfolio_returns, (1 - confidence) * 100)

        # Method 2: Parametric VaR (assumes normal distribution)
        from scipy import stats
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        z_score = stats.norm.ppf(1 - confidence)
        parametric_var = mean_return + (z_score * std_return)

        # Conditional VaR (Expected Shortfall) — average loss beyond VaR
        cvar = portfolio_returns[portfolio_returns <= historical_var].mean()

        result = {
            "historical_var": historical_var,
            "parametric_var": parametric_var,
            "conditional_var": cvar,
            "portfolio_mean_return": mean_return,
            "portfolio_std": std_return,
            "confidence_level": confidence
        }

        logger.info(f"Portfolio VaR ({confidence*100:.0f}%): "
                   f"Historical={historical_var*100:.3f}%, "
                   f"Parametric={parametric_var*100:.3f}%")

        return result

    # ==========================================
    # CORRELATION ANALYSIS
    # ==========================================

    def calculate_correlation_matrix(self, data: dict) -> pd.DataFrame:
        """
        Calculate correlation matrix between all stocks.
        
        Why it matters:
            - Highly correlated stocks = NOT diversified
            - If all stocks move together, your risk is concentrated
            - Ideal portfolio has LOW correlations
        """
        # Build returns DataFrame
        returns_dict = {}
        for symbol, df in data.items():
            if "daily_return" in df.columns:
                returns_dict[symbol] = df["daily_return"]
            elif "close" in df.columns:
                returns_dict[symbol] = df["close"].pct_change()

        returns_df = pd.DataFrame(returns_dict).dropna()

        # Correlation matrix
        corr_matrix = returns_df.corr()

        logger.info("Correlation Matrix:")
        logger.info(f"\n{corr_matrix.round(3).to_string()}")

        return corr_matrix

    def check_correlation_risk(self, corr_matrix: pd.DataFrame,
                                threshold: float = 0.7) -> dict:
        """
        Check for dangerously high correlations.
        
        If two stocks have correlation > threshold,
        holding both is like doubling down on the same bet.
        """
        warnings = []
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                stock1 = corr_matrix.columns[i]
                stock2 = corr_matrix.columns[j]

                if abs(corr) >= threshold:
                    high_corr_pairs.append({
                        "stock1": stock1,
                        "stock2": stock2,
                        "correlation": corr
                    })
                    warnings.append(
                        f"HIGH CORRELATION: {stock1} & {stock2} = {corr:.3f}"
                    )

        result = {
            "high_correlation_pairs": high_corr_pairs,
            "warnings": warnings,
            "is_diversified": len(high_corr_pairs) == 0,
            "avg_correlation": corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k=1)
            ].mean()
        }

        if result["is_diversified"]:
            logger.info("✅ Portfolio is well diversified")
        else:
            for w in warnings:
                logger.warning(f"⚠️ {w}")

        return result

    # ==========================================
    # RISK SCORING
    # ==========================================

    def calculate_risk_score(self, df: pd.DataFrame) -> dict:
        """
        Calculate a comprehensive risk score for a stock.
        
        Scores from 1 (very safe) to 10 (very risky).
        
        Factors considered:
        - Volatility
        - Maximum drawdown
        - Average daily range
        - Volume consistency
        - Trend strength
        """
        if df.empty:
            return {"risk_score": 5, "risk_level": "MEDIUM"}

        scores = {}

        # 1. Volatility Score (higher vol = higher risk)
        if "daily_return" in df.columns:
            vol = df["daily_return"].std() * np.sqrt(252)  # Annualized
            if vol > 0.5:
                scores["volatility"] = 10
            elif vol > 0.4:
                scores["volatility"] = 8
            elif vol > 0.3:
                scores["volatility"] = 6
            elif vol > 0.2:
                scores["volatility"] = 4
            else:
                scores["volatility"] = 2
        else:
            scores["volatility"] = 5

        # 2. Drawdown Score
        if "daily_return" in df.columns:
            cumulative = (1 + df["daily_return"]).cumprod()
            running_max = cumulative.cummax()
            max_dd = ((cumulative - running_max) / running_max).min()

            if abs(max_dd) > 0.4:
                scores["drawdown"] = 10
            elif abs(max_dd) > 0.3:
                scores["drawdown"] = 8
            elif abs(max_dd) > 0.2:
                scores["drawdown"] = 6
            elif abs(max_dd) > 0.1:
                scores["drawdown"] = 4
            else:
                scores["drawdown"] = 2
        else:
            scores["drawdown"] = 5

        # 3. Price Range Score
        if "hl_range" in df.columns and "close" in df.columns:
            avg_range_pct = (df["hl_range"] / df["close"]).mean()
            if avg_range_pct > 0.05:
                scores["price_range"] = 9
            elif avg_range_pct > 0.03:
                scores["price_range"] = 7
            elif avg_range_pct > 0.02:
                scores["price_range"] = 5
            else:
                scores["price_range"] = 3
        else:
            scores["price_range"] = 5

        # 4. Volume Consistency Score (inconsistent volume = risky)
        if "volume" in df.columns:
            vol_cv = df["volume"].std() / df["volume"].mean() if df["volume"].mean() > 0 else 1
            if vol_cv > 1.5:
                scores["volume"] = 8
            elif vol_cv > 1.0:
                scores["volume"] = 6
            elif vol_cv > 0.5:
                scores["volume"] = 4
            else:
                scores["volume"] = 2
        else:
            scores["volume"] = 5

        # 5. Trend Score (no clear trend = risky for trend strategies)
        if "sma_50" in df.columns and "sma_200" in df.columns:
            latest = df.iloc[-1]
            if latest["close"] > latest["sma_50"] > latest["sma_200"]:
                scores["trend"] = 2  # Strong uptrend = lower risk
            elif latest["close"] < latest["sma_50"] < latest["sma_200"]:
                scores["trend"] = 3  # Strong downtrend = moderate risk
            else:
                scores["trend"] = 7  # Choppy = higher risk
        else:
            scores["trend"] = 5

        # Calculate overall risk score
        overall = np.mean(list(scores.values()))

        # Risk level
        if overall >= 8:
            risk_level = "VERY HIGH"
        elif overall >= 6:
            risk_level = "HIGH"
        elif overall >= 4:
            risk_level = "MEDIUM"
        elif overall >= 2:
            risk_level = "LOW"
        else:
            risk_level = "VERY LOW"

        result = {
            "risk_score": round(overall, 1),
            "risk_level": risk_level,
            "component_scores": scores
        }

        logger.info(f"Risk Score: {overall:.1f}/10 ({risk_level})")
        return result

    # ==========================================
    # RISK REPORT
    # ==========================================

    def generate_risk_report(self, data: dict, portfolio_value: float) -> str:
        """
        Generate comprehensive risk report for all stocks.
        
        Args:
            data: Dictionary of {symbol: DataFrame}
            portfolio_value: Current portfolio value
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                🛡️ RISK MANAGEMENT REPORT                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Portfolio Value: ${portfolio_value:>12,.2f}                         ║
║  Initial Capital: ${self.initial_capital:>12,.2f}                    ║
║  P&L:            ${portfolio_value - self.initial_capital:>12,.2f}                    ║
║                                                              ║
║  📏 RISK PARAMETERS                                          ║
║  ─────────────────────────────────                           ║
║  Risk Per Trade:     {self.risk_per_trade*100:>8.1f}%                        ║
║  Max Position Size:  {self.max_position_size*100:>8.1f}%                        ║
║  Stop Loss:          {self.stop_loss_pct*100:>8.1f}%                        ║
║  Take Profit:        {self.take_profit_pct*100:>8.1f}%                        ║
║  Max Drawdown:       {self.max_drawdown_limit*100:>8.1f}%                        ║
║  Max Positions:      {self.max_open_positions:>8}                          ║
║                                                              ║
"""

        # Risk scores for each stock
        report += "║  📊 INDIVIDUAL STOCK RISK SCORES                            ║\n"
        report += "║  ─────────────────────────────────                           ║\n"

        for symbol, df in data.items():
            risk = self.calculate_risk_score(df)
            score = risk["risk_score"]
            level = risk["risk_level"]

            bar = "█" * int(score) + "░" * (10 - int(score))
            report += f"║  {symbol:<8} [{bar}] {score:>4.1f}/10  {level:<12}     ║\n"

        # Correlation analysis
        corr_matrix = self.calculate_correlation_matrix(data)
        corr_risk = self.check_correlation_risk(corr_matrix)

        report += "║                                                              ║\n"
        report += "║  🔗 CORRELATION ANALYSIS                                     ║\n"
        report += "║  ─────────────────────────────────                           ║\n"
        report += f"║  Average Correlation: {corr_risk['avg_correlation']:>8.3f}                        ║\n"
        report += f"║  Diversified:         {'YES ✅' if corr_risk['is_diversified'] else 'NO  ⚠️':>8}                        ║\n"

        if corr_risk["high_correlation_pairs"]:
            report += "║                                                              ║\n"
            report += "║  ⚠️ HIGH CORRELATION PAIRS:                                  ║\n"
            for pair in corr_risk["high_correlation_pairs"]:
                report += f"║    {pair['stock1']} ↔ {pair['stock2']}: {pair['correlation']:.3f}                          ║\n"

        report += "║                                                              ║\n"
        report += "╚══════════════════════════════════════════════════════════════╝\n"

        return report

    # ==========================================
    # POSITION SIZING RECOMMENDATION
    # ==========================================

    def recommend_position_size(self, capital: float, entry_price: float,
                                 atr: float, win_rate: float = 0.5,
                                 avg_win: float = 1.0,
                                 avg_loss: float = -1.0) -> dict:
        """
        Get position size recommendations from ALL methods.
        
        Returns recommendations from each method so you can compare.
        """
        # Fixed percentage stop-loss price
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)

        # Method 1: Fixed Fractional
        fixed_shares = self.fixed_fractional_size(
            capital, self.risk_per_trade, entry_price, stop_loss_price
        )

        # Method 2: Kelly Criterion
        kelly_shares = self.kelly_criterion_size(
            capital, entry_price, win_rate, avg_win, avg_loss
        )

        # Method 3: Volatility-Based
        vol_shares = self.volatility_based_size(capital, entry_price, atr)

        # Conservative recommendation (minimum of all methods)
        conservative = min(fixed_shares, kelly_shares, vol_shares) if all(
            [fixed_shares, kelly_shares, vol_shares]
        ) else max(fixed_shares, kelly_shares, vol_shares, 0)

        # Moderate recommendation (average)
        all_sizes = [s for s in [fixed_shares, kelly_shares, vol_shares] if s > 0]
        moderate = int(np.mean(all_sizes)) if all_sizes else 0

        recommendations = {
            "fixed_fractional": {
                "shares": fixed_shares,
                "value": fixed_shares * entry_price,
                "pct_of_capital": (fixed_shares * entry_price / capital * 100) if capital > 0 else 0
            },
            "kelly_criterion": {
                "shares": kelly_shares,
                "value": kelly_shares * entry_price,
                "pct_of_capital": (kelly_shares * entry_price / capital * 100) if capital > 0 else 0
            },
            "volatility_based": {
                "shares": vol_shares,
                "value": vol_shares * entry_price,
                "pct_of_capital": (vol_shares * entry_price / capital * 100) if capital > 0 else 0
            },
            "recommended_conservative": conservative,
            "recommended_moderate": moderate
        }

        return recommendations