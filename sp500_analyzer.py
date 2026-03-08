"""
S&P 500 COMPLETE STOCK ANALYZER - v5.4 Logic Upgrade
Technical + Fundamental Analysis with Email Delivery
Theme: Dark Slate (Redesigned)

=======================================================================
UPGRADES APPLIED FROM NIFTY50 v5.4 (ported to SP500):

  U01  WILDER'S RSI - Replaces simple rolling mean with ewm(com=period-1)
         Matches TradingView RSI values. Simple mean overstates RSI by
         5-10 points. All RSI signals now calculated correctly.

  U02  TREND VETO GATE - Hard block before rating is assigned.
         If 3+ of these bearish signals fire simultaneously:
           · SMA20 declining (today < 5 bars ago)
           · Death cross forming (SMA20 < SMA50)
           · MACD < Signal (bearish crossover)
           · RSI < 50 (momentum lost)
           · Price < SMA50
           · RSI falling fast (|slope| > 8)
         → Maximum rating is capped at HOLD, regardless of combined
         score. Fundamentals cannot rescue a stock in confirmed
         technical downtrend.

  U03  SMA200 SLOPE GUARD - SMA200 bonus now requires SMA200 itself
         to be rising (today > 10 bars ago). A rising price above a
         flat/falling SMA200 no longer earns +2 bonus.

  U04  ANALYST BONUS APPLIED TO SCORE - Analyst consensus was
         previously shown in the table but never applied to the score.
         Now adds/subtracts ±5 points, with a tech_score >= 2 gate
         to prevent analyst buy from rescuing a weak chart.

  U05  DYNAMIC WEIGHT SHIFT - When 2+ bearish tech signals fire,
         weight shifts from 50/50 to 40/60 (tech/fund → more tech
         weight in downtrend). At 3+ signals veto fires.

  U06  SMA20 DECLINING PENALTY (V53-1) - If SMA20 today < SMA20
         five bars ago, short-term trend is actively declining right
         now. Catches stocks rolling over before SMA50 lags down.

  U07  DEATH CROSS FORMING PENALTY (V53-2) - SMA20 < SMA50 adds
         additional -1 tech score as early warning of sustained
         bearish momentum shift.

  U08  ADX DIRECTION-AWARE (V52-1) - ADX bonus only when price >
         SMA50. A strong downtrend has high ADX but should NOT be
         rewarded. Strong downtrend now penalised with -1.

  U09  RSI WEAK MOMENTUM ZONE (V52-2) - RSI 30-45 = weak momentum
         zone adds -1 tech score. Was treated as neutral previously.

  U10  DOUBLE SMA PENALTY (V52-3) - Price below BOTH SMA20 AND
         SMA50 simultaneously = confirmed short-term downtrend, adds
         extra -1 penalty.

  U11  RSI DIVERGENCE DETECTION - Bearish divergence (price higher
         high + RSI lower high) and Bullish divergence (price lower
         low + RSI higher low) detected over a 20-bar window.

  U12  FULL RSI SLOPE OBJECT - calculate_rsi_slope() now returns
         direction (Rising/Falling/Flat), strong flag (|slope| > 8),
         and rsi_5bar for display. Replaces scalar-only version.

  U13  SECTOR-ADJUSTED PE (V52-4) - Financial sector stocks (Banks,
         Insurance, Financial Services) use PE < 15 thresholds.
         JPM, BAC, GS, WFC, MS, AXP etc scored correctly.

  U14  NEGATIVE GROWTH PENALTY + CAP (FIX-5 + CAL-4) - Negative
         revenue/earnings growth now penalised. Combined penalty
         capped at -10 to prevent cyclicals from being wiped out.

  U15  FCF WEIGHT RAISED - Free cash flow +15 (was +5).
         Cash generation matters especially in rate-hike environments.

  U16  D/E WEIGHT RAISED - Low debt/equity +15 (was +10).
         Highly leveraged companies collapse in credit crunches.

  U17  PARTIAL CREDIT FOR MISSING FIELDS (CAL-5) - Missing PEG,
         ROA, and Current Ratio now award partial credit instead of 0.
         Prevents unfair penalisation when yFinance returns None.

  U18  DATA SANITY CHECK - Detects >20% single-day price moves with
         normal volume as likely bad yFinance data. Skips stock
         rather than corrupting signals.

  U19  DATA FRESHNESS WARNING - Warns if last candle is >5 trading
         days old. Catches weekend/holiday stale data lag.

  U20  auto_adjust=False - Uses unadjusted prices to match
         TradingView RSI/MACD chart values.

  U21  VOLUME RATIO SMOOTHED - Uses 5-bar average for numerator
         instead of single last bar. Reduces noise from ex-dividend
         days and index rebalancing events.

  U22  SECTOR DIVERSITY CAP - Max 4 picks per sector in Top 20 Buy
         table. Prevents table from being flooded by one hot sector.

  U23  SELL GATE - Mirrors Nifty v5.4 sell validation: blocks
         oversold shorts (RSI < 35), RSI > 50 + MACD bullish stocks,
         and fast-recovering stocks from the sell table.

RETAINED FROM SP500 v9/v10/v11:
  - Dark Slate HTML design with grouped column headers
  - Quick View / Detail View toggle
  - RSI mini progress bar
  - Earnings row warning + pulsing badge
  - v8 MACD histogram slope penalties
  - v8 price extension above SMA20 penalty
  - v9 high sell-off volume filter in recommendations
  - Live Clock + Index Strip (DJI / NDX / SPX)
  - Mobile responsive layout
=======================================================================

Requirements:
    pip install yfinance pandas numpy pytz
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

warnings.filterwarnings('ignore')

# U22: Sector diversity cap
MAX_PICKS_PER_SECTOR = 4


class SP500CompleteAnalyzer:
    def __init__(self):
        self.sp500_stocks = {
            'NVDA':  'NVIDIA',
            'AAPL':  'Apple Inc.',
            'MSFT':  'Microsoft',
            'AMZN':  'Amazon',
            'GOOGL': 'Alphabet (Class A)',
            'GOOG':  'Alphabet (Class C)',
            'META':  'Meta Platforms',
            'TSLA':  'Tesla',
            'AVGO':  'Broadcom',
            'BRK-B': 'Berkshire Hathaway',
            'WMT':   'Walmart',
            'LLY':   'Eli Lilly',
            'JPM':   'JPMorgan Chase',
            'XOM':   'ExxonMobil',
            'V':     'Visa Inc.',
            'JNJ':   'Johnson & Johnson',
            'MA':    'Mastercard',
            'MU':    'Micron Technology',
            'ORCL':  'Oracle Corporation',
            'COST':  'Costco',
            'ABBV':  'AbbVie',
            'HD':    'Home Depot',
            'BAC':   'Bank of America',
            'PG':    'Procter & Gamble',
            'CVX':   'Chevron Corporation',
            'CAT':   'Caterpillar Inc.',
            'KO':    'Coca-Cola Company',
            'GE':    'GE Aerospace',
            'AMD':   'Advanced Micro Devices',
            'NFLX':  'Netflix',
            'PLTR':  'Palantir Technologies',
            'MRK':   'Merck & Co.',
            'CSCO':  'Cisco Systems',
            'PM':    'Philip Morris International',
            'LRCX':  'Lam Research',
            'AMAT':  'Applied Materials',
            'MS':    'Morgan Stanley',
            'WFC':   'Wells Fargo',
            'GS':    'Goldman Sachs',
            'RTX':   'RTX Corporation',
            'UNH':   'UnitedHealth Group',
            'TMUS':  'T-Mobile US',
            'IBM':   'IBM',
            'MCD':   "McDonald's",
            'AXP':   'American Express',
            'INTC':  'Intel',
            'PEP':   'PepsiCo',
            'LIN':   'Linde plc',
            'GEV':   'GE Vernova',
            'VZ':    'Verizon',
        }
        self.results = []

    # =========================================================================
    #  UTILITY
    # =========================================================================
    def get_est_time(self):
        return datetime.now(pytz.timezone('US/Eastern'))

    # U01: Wilder's RSI — matches TradingView. Simple rolling mean overstates
    # RSI by 5-10 points. ewm(com=period-1) = Wilder's smoothing (alpha=1/period).
    def calculate_rsi(self, prices, period=14):
        delta    = prices.diff()
        gain     = delta.where(delta > 0, 0)
        loss     = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return (100 - (100 / (1 + rs))).iloc[-1]

    # U12: Full RSI slope object — direction, strong flag, rsi_5bar for display.
    # Replaces scalar-only version. Backward-compat: callers extract ['slope'].
    def calculate_rsi_slope(self, prices, period=14, lookback=5):
        """
        Returns direction and strength of RSI movement.
        RSI rising from 35→50 = momentum building   → GOOD
        RSI falling from 65→46 = momentum fading    → BAD
        RSI flat at 50          = no signal          → NEUTRAL
        Returns dict: slope, direction, strong (|slope|>8), rsi_5bar
        """
        try:
            delta    = prices.diff()
            gain     = delta.where(delta > 0, 0)
            loss     = (-delta.where(delta < 0, 0))
            avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
            rsi_ser  = 100 - (100 / (1 + avg_gain / avg_loss))
            rsi_ser  = rsi_ser.dropna()
            if len(rsi_ser) < lookback + 2:
                return {'slope': 0, 'direction': 'Flat', 'strong': False,
                        'rsi_5bar': float(rsi_ser.iloc[-1]) if len(rsi_ser) else 50}
            rsi_now  = rsi_ser.iloc[-1]
            rsi_prev = rsi_ser.iloc[-(lookback + 1)]
            slope    = round(rsi_now - rsi_prev, 2)
            if slope > 3:
                direction = 'Rising'
            elif slope < -3:
                direction = 'Falling'
            else:
                direction = 'Flat'
            strong = abs(slope) > 8   # sharp move — e.g. RSI 71→46 in 5 bars
            return {'slope': slope, 'direction': direction,
                    'strong': strong, 'rsi_5bar': round(rsi_prev, 1)}
        except Exception:
            return {'slope': 0, 'direction': 'Flat', 'strong': False, 'rsi_5bar': 50}

    # U11: RSI Divergence detection (ported from Nifty50 v5.4 NEW-1)
    def detect_rsi_divergence(self, prices, window=14):
        """
        Bearish divergence: price makes HIGHER high, RSI makes LOWER high.
        Bullish divergence: price makes LOWER low,  RSI makes HIGHER low.
        Both measured over the last 20 bars vs prior 20 bars.
        """
        try:
            delta    = prices.diff()
            gain     = delta.where(delta > 0, 0)
            loss     = (-delta.where(delta < 0, 0))
            avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
            avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
            rsi_ser  = 100 - (100 / (1 + avg_gain / avg_loss))
            rsi_ser  = rsi_ser.dropna()
            lookback = 20
            if len(prices) < lookback + window or len(rsi_ser) < lookback:
                return 'None'
            recent_price = prices.iloc[-lookback:]
            prior_price  = prices.iloc[-(lookback * 2):-lookback]
            recent_rsi   = rsi_ser.iloc[-lookback:]
            prior_rsi    = rsi_ser.iloc[-(lookback * 2):-lookback]
            # Bearish: price higher high + RSI lower high
            if (recent_price.max() > prior_price.max() * 1.005 and
                    recent_rsi.max() < prior_rsi.max() * 0.97):
                return 'Bearish Divergence'
            # Bullish: price lower low + RSI higher low
            if (recent_price.min() < prior_price.min() * 0.995 and
                    recent_rsi.min() > prior_rsi.min() * 1.03):
                return 'Bullish Divergence'
            return 'None'
        except Exception:
            return 'None'

    def calculate_macd(self, prices):
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]

    # v8: MACD histogram slope — histogram shrinking = momentum loss
    def calculate_macd_hist_slope(self, prices):
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal
        if len(hist) < 2:
            return 0.0
        return round(hist.iloc[-1] - hist.iloc[-2], 4)

    def calculate_atr(self, df, period=14):
        high  = df['High']
        low   = df['Low']
        close = df['Close']
        tr    = pd.concat([high - low,
                           abs(high - close.shift(1)),
                           abs(low  - close.shift(1))], axis=1).max(axis=1)
        return round(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1], 2)

    def calculate_adx(self, df, period=14):
        high     = df['High']
        low      = df['Low']
        close    = df['Close']
        plus_dm  = high.diff()
        minus_dm = low.diff().abs()
        plus_dm[plus_dm < 0]   = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[plus_dm < minus_dm]  = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr      = pd.concat([high - low,
                             abs(high - close.shift(1)),
                             abs(low  - close.shift(1))], axis=1).max(axis=1)
        atr14    = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14)
        dx       = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx      = dx.ewm(alpha=1 / period, adjust=False).mean()
        return round(adx.iloc[-1], 1)

    # U21: Volume ratio uses 5-bar average for numerator (smooths ex-div noise)
    def calculate_volume_ratio(self, df):
        avg_vol = df['Volume'].tail(20).mean()
        if avg_vol == 0:
            return 1.0
        last_5_avg = df['Volume'].tail(5).mean()
        return round(last_5_avg / avg_vol, 2)

    def get_earnings_date(self, info):
        try:
            ts = (info.get('earningsTimestamp') or
                  info.get('earningsTimestampStart') or
                  info.get('earningsDate'))
            if ts:
                if isinstance(ts, (list, tuple)):
                    ts = ts[0]
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    return dt.strftime('%d %b %Y')
                if hasattr(ts, 'strftime'):
                    return ts.strftime('%d %b %Y')
        except Exception:
            pass
        return "N/A"

    def is_earnings_soon(self, earnings_date_str, days=14):
        if earnings_date_str == "N/A":
            return False
        try:
            ed    = datetime.strptime(earnings_date_str, '%d %b %Y')
            delta = (ed - datetime.now()).days
            return 0 <= delta <= days
        except Exception:
            return False

    def fetch_index_data(self):
        indices = {'DJI': '^DJI', 'NDX': '^NDX', 'SPX': '^GSPC'}
        result  = {}
        for label, sym in indices.items():
            try:
                d     = yf.Ticker(sym).history(period='2d')
                price = d['Close'].iloc[-1]
                prev  = d['Close'].iloc[-2]
                chg   = price - prev
                pct   = chg / prev * 100
                arrow = '▲' if chg >= 0 else '▼'
                cls   = 'up' if chg >= 0 else 'dn'
                sign  = '+' if chg >= 0 else ''
                result[label] = {'price': f"{price:,.2f}",
                                 'chg':   f"{arrow} {sign}{pct:.2f}%",
                                 'cls':   cls}
            except Exception:
                result[label] = {'price': 'N/A', 'chg': '—', 'cls': ''}
        return result

    # U18: Data sanity check — skip stocks with bad yFinance data
    def is_data_clean(self, df):
        """
        Returns (True, '') if data looks valid.
        Returns (False, reason) if a suspicious spike is detected.
        Any >20% single-day move with normal volume = likely data error.
        """
        try:
            close      = df['Close']
            volume     = df['Volume']
            pct_change = close.pct_change().abs()
            spike_days = pct_change[pct_change > 0.20]
            if spike_days.empty:
                return True, ''
            for spike_date, spike_val in spike_days.items():
                avg_vol   = volume.mean()
                spike_vol = volume.loc[spike_date] if spike_date in volume.index else 0
                if avg_vol > 0 and spike_vol < avg_vol * 3:
                    return False, f"Suspicious {spike_val*100:.0f}% move on {spike_date.date()} — possible bad data"
            return True, ''
        except Exception:
            return True, ''

    # =========================================================================
    #  RESISTANCE & SUPPORT
    # =========================================================================
    def find_resistance_levels(self, df, current_price, num_levels=5):
        window      = 5
        swing_highs = []
        for src_days in [180, 252]:
            highs = df.tail(src_days)['High'].values
            for i in range(window, len(highs) - window):
                if (highs[i] > max(highs[i-window:i]) and
                        highs[i] > max(highs[i+1:i+window+1])):
                    swing_highs.append(highs[i])
        high_52w = df['High'].tail(252).max()
        if high_52w > current_price * 1.005:
            swing_highs.append(high_52w)
        magnitude = 10 ** (len(str(int(current_price))) - 2)
        step      = magnitude * 5
        level     = current_price
        for _ in range(20):
            level += step
            if level <= current_price * 1.30:
                swing_highs.append(level)
        if not swing_highs:
            return []
        swing_highs = sorted(set([round(h, 2) for h in swing_highs]))
        clusters, cluster = [], [swing_highs[0]]
        for lv in swing_highs[1:]:
            if (lv - cluster[-1]) / cluster[-1] < 0.015:
                cluster.append(lv)
            else:
                clusters.append(cluster); cluster = [lv]
        clusters.append(cluster)
        res = [{'level': round(sum(c)/len(c), 2), 'strength': len(c)}
               for c in clusters if sum(c)/len(c) > current_price * 1.005]
        return sorted(res, key=lambda x: x['level'])[:num_levels]

    def find_support_levels(self, df, current_price, num_levels=5):
        window     = 5
        swing_lows = []
        for src_days in [180, 252]:
            lows = df.tail(src_days)['Low'].values
            for i in range(window, len(lows) - window):
                if (lows[i] < min(lows[i-window:i]) and
                        lows[i] < min(lows[i+1:i+window+1])):
                    swing_lows.append(lows[i])
        low_52w = df['Low'].tail(252).min()
        if low_52w < current_price * 0.995:
            swing_lows.append(low_52w)
        magnitude = 10 ** (len(str(int(current_price))) - 2)
        step      = magnitude * 5
        level     = current_price
        for _ in range(20):
            level -= step
            if level >= current_price * 0.70 and level > 0:
                swing_lows.append(level)
        if not swing_lows:
            return []
        swing_lows = sorted(set([round(l, 2) for l in swing_lows]))
        clusters, cluster = [], [swing_lows[0]]
        for lv in swing_lows[1:]:
            if (lv - cluster[-1]) / cluster[-1] < 0.015:
                cluster.append(lv)
            else:
                clusters.append(cluster); cluster = [lv]
        clusters.append(cluster)
        sup = [{'level': round(sum(c)/len(c), 2), 'strength': len(c)}
               for c in clusters if sum(c)/len(c) < current_price * 0.995]
        return sorted(sup, key=lambda x: x['level'], reverse=True)[:num_levels]

    # =========================================================================
    #  DYNAMIC TARGETS
    # =========================================================================
    def calculate_dynamic_targets(self, current_price, resistance_levels,
                                   support_levels, target_price, atr):
        valid      = [r['level'] for r in resistance_levels
                      if r['level'] > current_price * 1.005]
        min_target = current_price + (atr * 2)
        if len(valid) >= 2:
            t1, t2        = valid[0], valid[1]
            target_status = "Real S/R Levels"
        elif len(valid) == 1:
            t1 = valid[0]
            t2 = (round(target_price, 2)
                  if target_price and target_price > t1 * 1.01
                  else round(t1 * 1.04, 2))
            target_status = "Partial Real Levels"
        else:
            t1 = (round(target_price, 2)
                  if target_price and target_price > current_price * 1.005
                  else round(current_price * 1.03, 2))
            t2            = round(t1 * 1.04, 2)
            target_status = "ATH Zone — Projected"
        if t1 < min_target:
            t1            = round(min_target, 2)
            t2            = round(t1 * 1.04, 2)
            target_status += " (ATR Adj)"
        return round(t1, 2), round(t2, 2), 0, target_status

    # =========================================================================
    #  FUNDAMENTAL SCORE
    #  U13: Sector-adjusted PE (Financial sector uses PE < 15 thresholds)
    #  U14: Negative growth penalty capped at -10
    #  U15: FCF raised to +15
    #  U16: D/E raised to +15
    #  U17: Partial credit for missing fields
    # =========================================================================
    def get_fundamental_score(self, info, sector=''):
        score = 0

        pe  = info.get('trailingPE', info.get('forwardPE', 0))
        pb  = info.get('priceToBook', 0)
        peg = info.get('pegRatio', 0)

        # U13: Financial sector (banks, insurance) always trade at structurally
        # lower PE. Using PE < 25 for JPM/BAC rewards them for being "cheap"
        # when they are merely sector-typical. Use PE < 15 for financials.
        is_financial = sector in ('Financial Services', 'Banks', 'Banking',
                                  'Insurance', 'Financial', 'Capital Markets',
                                  'Asset Management')
        if is_financial:
            if pe and 0 < pe < 15:      score += 10   # genuinely cheap bank
            elif pe and 15 <= pe < 20:  score += 5    # fair value for a bank
            # PE > 20 for a financial = expensive for its sector → 0 pts
        else:
            if pe  and 0 < pe  < 25:    score += 10
            elif pe  and 25 <= pe < 35: score += 5

        if pb  and 0 < pb  < 3:      score += 5
        elif pb  and 3 <= pb  < 5:   score += 3

        if peg and 0 < peg < 1:      score += 10
        elif peg and 1 <= peg < 2:   score += 5
        else:                        score += 3   # U17: partial credit if PEG missing

        # Profitability
        roe = info.get('returnOnEquity', 0)
        roa = info.get('returnOnAssets', 0)
        pm  = info.get('profitMargins', 0)
        if roe and roe > 0.15:   score += 10
        elif roe and roe > 0.10: score += 5
        if roa and roa > 0.05:   score += 5
        elif roa and roa > 0.02: score += 3
        else:                    score += 2   # U17: partial credit if ROA missing
        if pm  and pm  > 0.10:   score += 10
        elif pm  and pm  > 0.05: score += 5

        # Growth (U14: penalise negative growth; cap total penalty at -10)
        rg = info.get('revenueGrowth', 0)
        eg = info.get('earningsGrowth', 0)
        growth_penalty = 0
        if rg and rg > 0.15:    score += 10
        elif rg and rg > 0.10:  score += 7
        elif rg and rg > 0.05:  score += 5
        elif rg and rg < 0:     growth_penalty += 10   # track separately for cap

        if eg and eg > 0.15:    score += 10
        elif eg and eg > 0.10:  score += 7
        elif eg and eg > 0.05:  score += 5
        elif eg and eg < 0:     growth_penalty += 10   # track separately for cap

        # Cap combined growth penalty at -10 to prevent cyclicals (energy,
        # materials, industrials) with one bad quarter from being wiped out.
        score -= min(growth_penalty, 10)

        # Balance sheet health
        de = info.get('debtToEquity', 0)
        cr = info.get('currentRatio', 0)
        fc = info.get('freeCashflow', 0)

        # U16: D/E weight raised to +15 (was +10). Highly leveraged companies
        # face outsized risk in rate-hike cycles.
        if de is not None:
            if de < 50:    score += 15   # was +10
            elif de < 100: score += 7    # was +5
        else:
            score += 5

        if cr and cr > 1.5:   score += 10
        elif cr and cr > 1.0: score += 5
        else:                  score += 3   # U17: partial credit if CR missing

        # U15: FCF weight raised to +15 (was +5). Cash generation = survival
        # in downturns and ability to buy back shares / pay dividends.
        if fc and fc > 0:     score += 15   # was +5

        return min(max(score, 0), 100)

    # =========================================================================
    #  MAIN ANALYSIS
    # =========================================================================
    def analyze_stock(self, symbol, name):
        try:
            stock = yf.Ticker(symbol)

            # U20: auto_adjust=False — use unadjusted prices to match TradingView.
            # Adjusted prices change RSI/MACD values vs what you see on charts.
            # U19: Fetch with explicit period — check freshness after.
            df = stock.history(period='1y', auto_adjust=False)

            if df.empty or len(df) < 200:
                return None

            # U19: Data freshness check — warn if last candle is stale
            today          = datetime.now(pytz.timezone('US/Eastern')).date()
            last_candle    = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            days_lag       = (today - last_candle).days
            if days_lag > 5:
                print(f"  ⚠ {symbol}: Data lag {days_lag} days (last candle: {last_candle})")

            # Ensure 'Close' column exists
            if 'Close' not in df.columns and 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'Close'})

            info = stock.info

            # U18: Data sanity check
            data_ok, data_warn = self.is_data_clean(df)
            if not data_ok:
                print(f"  ⚠ Skipping {symbol}: {data_warn}")
                return None

            current_price = df['Close'].iloc[-1]
            sma_20        = df['Close'].rolling(20).mean().iloc[-1]
            sma_50        = df['Close'].rolling(50).mean().iloc[-1]
            sma_200       = df['Close'].rolling(200).mean().iloc[-1]

            # U06: SMA20 slope — is the short-term trend actively declining?
            # Catches stocks rolling over BEFORE SMA50 lags down (key fix).
            sma_20_series    = df['Close'].rolling(20).mean()
            sma_20_5bar_ago  = sma_20_series.iloc[-6] if len(sma_20_series) >= 6 else sma_20
            sma_20_declining = sma_20 < sma_20_5bar_ago

            # U07: Death cross forming — SMA20 has crossed below SMA50.
            death_cross_forming = sma_20 < sma_50

            # U03: SMA200 slope guard — bonus only if SMA200 is rising.
            # Price above a flat/falling SMA200 does not confirm an uptrend.
            sma_200_series    = df['Close'].rolling(200).mean()
            sma_200_10bar_ago = sma_200_series.iloc[-11] if len(sma_200_series) >= 11 else sma_200
            sma_200_rising    = sma_200 > sma_200_10bar_ago

            rsi          = self.calculate_rsi(df['Close'])
            macd, signal = self.calculate_macd(df['Close'])
            atr          = self.calculate_atr(df)
            atr_pct      = round((atr / current_price) * 100, 2)
            adx          = self.calculate_adx(df)
            vol_ratio    = self.calculate_volume_ratio(df)

            # v8: MACD histogram slope for momentum-reversal detection
            macd_hist_slope = self.calculate_macd_hist_slope(df['Close'])

            # U11: RSI divergence detection
            rsi_divergence = self.detect_rsi_divergence(df['Close'])

            # U12: Full RSI slope object
            rsi_slope_data  = self.calculate_rsi_slope(df['Close'])
            rsi_slope       = rsi_slope_data['slope']        # scalar (backward compat)
            rsi_direction   = rsi_slope_data['direction']    # 'Rising'|'Falling'|'Flat'
            rsi_slope_strong = rsi_slope_data['strong']      # True if |slope| > 8
            rsi_5bar        = rsi_slope_data['rsi_5bar']

            high_52w = df['High'].tail(252).max()
            low_52w  = df['Low'].tail(252).min()

            resistance_levels = self.find_resistance_levels(df, current_price)
            support_levels    = self.find_support_levels(df, current_price)

            nearest_resistance = (resistance_levels[0]['level'] if resistance_levels
                                  else df.tail(60)['High'].quantile(0.90))
            nearest_support    = (support_levels[0]['level'] if support_levels
                                  else df.tail(60)['Low'].quantile(0.10))
            support_dist_pct   = round(((current_price - nearest_support) / current_price) * 100, 2)

            # ==================================================================
            #  TECHNICAL SCORE
            # ==================================================================
            tech_score = 0

            # SMA position
            tech_score += 1 if current_price > sma_20  else -1
            tech_score += 1 if current_price > sma_50  else -1

            # U03: SMA200 bonus only if SMA200 is actively rising
            if current_price > sma_200 and sma_200_rising:
                tech_score += 2   # confirmed long-term uptrend
            elif current_price > sma_200 and not sma_200_rising:
                tech_score += 0   # above SMA200 but trend flattening — no bonus
            else:
                tech_score -= 2   # below SMA200 — penalise

            # U10: Double SMA penalty — below BOTH SMA20 and SMA50 simultaneously
            if current_price < sma_20 and current_price < sma_50:
                tech_score -= 1

            # U06: SMA20 declining — short-term trend actively falling
            if sma_20_declining:
                tech_score -= 1

            # U07: Death cross forming — SMA20 crossed below SMA50
            if death_cross_forming:
                tech_score -= 1

            # RSI scoring — context-aware with direction (ported from Nifty v5.4)
            if rsi < 30:
                if current_price > sma_200:
                    tech_score += 2
                    rsi_signal = "Oversold ↑" if rsi_direction == 'Rising' else "Oversold"
                else:
                    tech_score -= 1
                    rsi_signal = "Oversold (Downtrend)"
            elif rsi > 70:
                if rsi_divergence == 'Bearish Divergence':
                    tech_score -= 3
                    rsi_signal = "Bearish Divergence ⚠"
                elif rsi_direction == 'Falling' and rsi_slope_strong:
                    tech_score -= 3
                    rsi_signal = f"Topping Out ⚠ ({rsi_5bar:.0f}→{rsi:.0f})"
                elif rsi_direction == 'Falling':
                    tech_score -= 2
                    rsi_signal = f"Fading ↓ ({rsi_5bar:.0f}→{rsi:.0f})"
                else:
                    tech_score -= 1
                    rsi_signal = "Overbought"
            elif 30 <= rsi <= 45:
                # U09: Weak momentum zone — no longer treated as neutral
                if rsi_direction == 'Falling':
                    tech_score -= 2
                    rsi_signal = f"Weak & Falling ↓ ({rsi_5bar:.0f}→{rsi:.0f})"
                else:
                    tech_score -= 1
                    rsi_signal = "Weak Momentum ⚠"
            elif 45 < rsi <= 55:
                if rsi_direction == 'Rising':
                    tech_score += 1
                    rsi_signal = f"Building ↑ ({rsi_5bar:.0f}→{rsi:.0f})"
                elif rsi_direction == 'Falling' and rsi_slope_strong:
                    tech_score -= 2
                    rsi_signal = f"Falling Fast ↓ ({rsi_5bar:.0f}→{rsi:.0f})"
                elif rsi_direction == 'Falling':
                    tech_score -= 1
                    rsi_signal = f"Fading ↓ ({rsi_5bar:.0f}→{rsi:.0f})"
                else:
                    rsi_signal = "Neutral →"
            else:
                # RSI 55-70: healthy zone — reward rising, penalise falling
                if rsi_direction == 'Rising' and rsi > 65:
                    rsi_signal = f"Near Overbought ⚠ ({rsi:.0f}↑)"
                elif rsi_direction == 'Rising':
                    tech_score = min(tech_score + 1, 6)
                    rsi_signal = f"Momentum ↑ ({rsi_5bar:.0f}→{rsi:.0f})"
                elif rsi_direction == 'Falling' and rsi_slope_strong:
                    tech_score -= 2
                    rsi_signal = f"Rolling Over ↓ ({rsi_5bar:.0f}→{rsi:.0f})"
                elif rsi_direction == 'Falling':
                    tech_score -= 1
                    rsi_signal = f"Softening ↓ ({rsi_5bar:.0f}→{rsi:.0f})"
                elif rsi_divergence == 'Bullish Divergence':
                    tech_score = min(tech_score + 1, 6)
                    rsi_signal = "Bullish Divergence ✅"
                else:
                    rsi_signal = f"Healthy ({rsi:.0f})"

            if macd > signal:
                tech_score += 1;  macd_signal = "Bullish"
            else:
                tech_score -= 1;  macd_signal = "Bearish"

            # U08: ADX direction-aware — bonus only when price > SMA50
            if adx > 25:
                if current_price > sma_50:
                    tech_score = min(tech_score + 1, 6)   # strong uptrend ✅
                else:
                    tech_score -= 1   # strong downtrend — penalise
            elif adx < 20:
                tech_score -= 1   # weak/no trend penalty

            # Volume influences tech score
            if vol_ratio > 1.5 and current_price > sma_20:
                tech_score = min(tech_score + 1, 6)
            elif vol_ratio < 0.7:
                tech_score -= 1

            # Near 52W high in uptrend bonus
            pct_from_52w_high = ((current_price - high_52w) / high_52w) * 100
            if pct_from_52w_high >= -5 and current_price > sma_200:
                tech_score = min(tech_score + 1, 6)

            # v8: RSI fading momentum penalties (retained from SP500 v8)
            if rsi > 55 and rsi_slope < -2:
                tech_score -= 2   # RSI peaked and falling fast
            elif rsi > 50 and rsi_slope < -1:
                tech_score -= 1   # mild RSI fade

            # v8: MACD histogram shrinking = momentum loss
            macd_hist_now = macd - signal
            if macd_hist_now > 0 and macd_hist_slope < -0.05:
                tech_score -= 1
            if macd_hist_now < 0 and macd_hist_slope < 0:
                tech_score -= 1

            # v8: Price extended above SMA20 AND RSI falling = stretched entry
            price_ext_pct = ((current_price - sma_20) / sma_20) * 100
            if price_ext_pct > 7 and rsi_slope < -1:
                tech_score -= 1
            # ==================================================================

            pe_ratio         = info.get('trailingPE', info.get('forwardPE', 0))
            pb_ratio         = info.get('priceToBook', 0)
            peg_ratio        = info.get('pegRatio', 0)
            market_cap       = info.get('marketCap', 0)
            dividend_yield   = info.get('dividendYield', 0)
            roe              = info.get('returnOnEquity', 0)
            roa              = info.get('returnOnAssets', 0)
            profit_margin    = info.get('profitMargins', 0)
            operating_margin = info.get('operatingMargins', 0)
            eps              = info.get('trailingEps', 0)
            revenue_growth   = info.get('revenueGrowth', 0)
            earnings_growth  = info.get('earningsGrowth', 0)
            debt_to_equity   = info.get('debtToEquity', 0)
            current_ratio    = info.get('currentRatio', 0)
            beta             = info.get('beta', 1.0)
            target_price     = info.get('targetMeanPrice', None)
            sector           = info.get('sector', 'N/A')
            analyst_key      = info.get('recommendationKey', 'N/A')
            analyst_map      = {
                'strongBuy': 'Strong Buy', 'buy': 'Buy',
                'hold': 'Hold', 'sell': 'Sell', 'strongSell': 'Strong Sell'
            }
            analyst_label    = analyst_map.get(
                analyst_key,
                analyst_key.title() if analyst_key else 'N/A')
            earnings_date    = self.get_earnings_date(info)
            earn_soon        = self.is_earnings_soon(earnings_date)

            fund_score = self.get_fundamental_score(info, sector)

            # U05: Dynamic weight shift — count active bearish signals
            bearish_signal_count = sum([
                bool(sma_20_declining),
                bool(death_cross_forming),
                bool(macd < signal),
                bool(rsi < 50),
                bool(current_price < sma_50),
                bool(rsi_direction == 'Falling' and rsi_slope_strong),
            ])

            # U05: Shift from 50/50 to 60/40 tech/fund when 2+ signals fire
            # (gives more weight to deteriorating technicals before veto kicks in)
            if bearish_signal_count >= 2:
                tech_weight  = 0.60
                fund_weight  = 0.40
                weight_label = "60/40 Tech (Downtrend Override)"
            else:
                tech_weight  = 0.50
                fund_weight  = 0.50
                weight_label = "50/50 (Normal)"

            # Combined score with dynamic weights
            tech_score_normalized = ((tech_score + 6) / 12) * 100
            combined_score        = (tech_score_normalized * tech_weight) + (fund_score * fund_weight)

            # U04: Analyst consensus ±5 applied to combined score.
            # Tech gate: tech_score >= 2 required before analyst buy counts.
            if analyst_key in ('strongBuy', 'buy'):
                if tech_score >= 2:
                    combined_score = min(combined_score + 5, 100)
            elif analyst_key in ('sell', 'strongSell'):
                combined_score = max(combined_score - 5, 0)

            # Rating thresholds
            if combined_score >= 75:
                rating = "⭐⭐⭐⭐⭐ STRONG BUY";  recommendation = "STRONG BUY"
            elif combined_score >= 55:
                rating = "⭐⭐⭐⭐ BUY";           recommendation = "BUY"
            elif combined_score >= 45:
                rating = "⭐⭐⭐ HOLD";            recommendation = "HOLD"
            elif combined_score >= 30:
                rating = "⭐⭐ SELL";              recommendation = "SELL"
            else:
                rating = "⭐ STRONG SELL";         recommendation = "STRONG SELL"

            # U02: TREND VETO GATE — hard cap BEFORE stop/target calculation.
            # 3+ confirmed bearish signals → maximum rating is HOLD.
            # Threshold = 3 (not 2) to avoid vetoing everything on a general
            # down day when MACD bear + RSI<50 fire across the whole market.
            veto_fired = bearish_signal_count >= 3 and recommendation in ("STRONG BUY", "BUY")
            if veto_fired:
                recommendation = "HOLD"
                rating         = "⭐⭐⭐ HOLD (Veto)"

            # Stop loss sizing by beta
            stock_beta = beta if beta else 1.0
            if stock_beta < 0.8:
                atr_multiplier = 1.0;  max_sl_pct = 5.0
            elif stock_beta < 1.2:
                atr_multiplier = 1.2;  max_sl_pct = 7.0
            elif stock_beta < 1.8:
                atr_multiplier = 1.5;  max_sl_pct = 10.0
            else:
                atr_multiplier = 2.0;  max_sl_pct = 12.0

            if recommendation in ("STRONG BUY", "BUY"):
                # ── LONG setup: stop below support, targets at resistance ──
                atr_stop       = nearest_support - (atr * atr_multiplier)
                min_allowed_sl = current_price * (1 - max_sl_pct / 100)
                stop_loss      = max(atr_stop, min_allowed_sl)
                sl_percentage  = ((current_price - stop_loss) / current_price) * 100
                stop_type      = "ATR Stop" if atr_stop >= min_allowed_sl else "Beta Cap"
                target_1, target_2, targets_hit, target_status = \
                    self.calculate_dynamic_targets(
                        current_price, resistance_levels,
                        support_levels, target_price, atr)
                if target_1 <= current_price * 1.005:
                    recommendation = "HOLD"; rating = "⭐⭐⭐ HOLD"
                upside = ((target_1 - current_price) / current_price) * 100

            elif recommendation == "HOLD":
                # ── HOLD setup: neutral — nearest support as soft floor,
                #    nearest resistance as soft ceiling. No directional bias.
                #    Stop = ATR below nearest support (risk management).
                #    Target = nearest resistance above price.
                #    Upside shown as % to resistance so user can judge entry.
                atr_stop       = nearest_support - (atr * atr_multiplier)
                min_allowed_sl = current_price * (1 - max_sl_pct / 100)
                stop_loss      = max(atr_stop, min_allowed_sl)
                sl_percentage  = ((current_price - stop_loss) / current_price) * 100
                stop_type      = "ATR Stop" if atr_stop >= min_allowed_sl else "Beta Cap"

                # Target: nearest resistance above price (or project +3%)
                valid_res = [r['level'] for r in resistance_levels
                             if r['level'] > current_price * 1.005]
                if len(valid_res) >= 2:
                    target_1, target_2 = valid_res[0], valid_res[1]
                    target_status = "Hold S/R Levels"
                elif len(valid_res) == 1:
                    target_1      = valid_res[0]
                    target_2      = round(target_1 * 1.03, 2)
                    target_status = "Hold Partial S/R"
                else:
                    target_1      = round(current_price * 1.03, 2)
                    target_2      = round(current_price * 1.06, 2)
                    target_status = "Hold Projected"
                targets_hit = 0
                upside      = ((target_1 - current_price) / current_price) * 100

            else:
                # ── SHORT/SELL setup: stop above resistance, targets at support ──
                atr_stop       = nearest_resistance + (atr * atr_multiplier)
                max_allowed_sl = current_price * (1 + max_sl_pct / 100)
                stop_loss      = min(atr_stop, max_allowed_sl)
                sl_percentage  = ((stop_loss - current_price) / current_price) * 100
                stop_type      = "ATR Stop" if atr_stop <= max_allowed_sl else "Beta Cap"
                valid_sups     = [s['level'] for s in support_levels
                                  if s['level'] < current_price * 0.995]
                if len(valid_sups) >= 2:
                    target_1, target_2 = valid_sups[0], valid_sups[1]
                    target_status      = "Real S/R Levels"
                elif len(valid_sups) == 1:
                    target_1      = valid_sups[0]
                    target_2      = round(target_1 * 0.96, 2)
                    target_status = "Partial Real Levels"
                else:
                    target_1      = round(current_price * 0.96, 2)
                    target_2      = round(current_price * 0.92, 2)
                    target_status = "Projected"
                targets_hit = 0
                upside      = ((current_price - target_1) / current_price) * 100

            risk        = abs(current_price - stop_loss)
            reward      = abs(target_1 - current_price)
            risk_reward = round(reward / risk, 2) if risk > 0 else 0

            # FIX-4: STRONG BUY requires R:R ≥ 1.5.
            # If R:R is between 1.0-1.5 the trade is fine but not exceptional —
            # demote to BUY so the user knows it's not a premium setup.
            if recommendation == "STRONG BUY" and risk_reward < 1.5:
                recommendation = "BUY"
                rating         = "⭐⭐⭐⭐ BUY"

            if fund_score >= 80:   quality = "Excellent"
            elif fund_score >= 60: quality = "Good"
            elif fund_score >= 40: quality = "Average"
            else:                  quality = "Poor"

            return {
                'Symbol':           symbol,
                'Name':             name,
                'Price':            round(current_price, 2),
                'Sector':           sector,
                'RSI':              round(rsi, 2),
                'RSI_Signal':       rsi_signal,
                'RSI_Slope':        rsi_slope,          # scalar — used by v8/v9 filters
                'RSI_Direction':    rsi_direction,       # U12: direction label
                'RSI_5Bar':         rsi_5bar,            # RSI 5 bars ago — for display
                'RSI_Divergence':   rsi_divergence,      # U11
                'MACD':             macd_signal,
                'MACD_Hist_Slope':  round(macd_hist_slope, 4),
                'ADX':              adx,
                'Vol_Ratio':        vol_ratio,
                'SMA_20':           round(sma_20, 2),
                'SMA_50':           round(sma_50, 2),
                'SMA_200':          round(sma_200, 2),
                'SMA_20_Declining': sma_20_declining,
                'Death_Cross':      death_cross_forming,
                'SMA_200_Rising':   sma_200_rising,
                'Bearish_Signals':  bearish_signal_count,
                'Weight_Mode':      weight_label,
                'Veto_Fired':       veto_fired,
                'Support':          round(nearest_support, 2),
                'Resistance':       round(nearest_resistance, 2),
                'Support_Dist_Pct': support_dist_pct,
                '52W_High':         round(high_52w, 2),
                '52W_Low':          round(low_52w, 2),
                'Tech_Score':       tech_score,
                'Tech_Score_Norm':  round(tech_score_normalized, 1),
                'ATR':              atr,
                'ATR_Pct':          atr_pct,
                'ATR_Multiplier':   atr_multiplier,
                'Stop_Type':        stop_type,
                'PE_Ratio':         round(pe_ratio, 2)           if pe_ratio else 0,
                'PB_Ratio':         round(pb_ratio, 2)           if pb_ratio else 0,
                'PEG_Ratio':        round(peg_ratio, 2)          if peg_ratio else 0,
                'ROE':              round(roe * 100, 2)          if roe else 0,
                'ROA':              round(roa * 100, 2)          if roa else 0,
                'Profit_Margin':    round(profit_margin * 100, 2)    if profit_margin else 0,
                'Operating_Margin': round(operating_margin * 100, 2) if operating_margin else 0,
                'EPS':              round(eps, 2)                if eps else 0,
                'Dividend_Yield':   round(dividend_yield * 100, 2)   if dividend_yield else 0,
                'Revenue_Growth':   round(revenue_growth * 100, 2)   if revenue_growth else 0,
                'Earnings_Growth':  round(earnings_growth * 100, 2)  if earnings_growth else 0,
                'Debt_to_Equity':   round(debt_to_equity, 2)    if debt_to_equity else 0,
                'Current_Ratio':    round(current_ratio, 2)     if current_ratio else 0,
                'Market_Cap':       round(market_cap / 1e9, 2)  if market_cap else 0,
                'Beta':             round(beta, 2)               if beta else 1.0,
                'Fund_Score':       round(fund_score, 1),
                'Quality':          quality,
                'Combined_Score':   round(combined_score, 1),
                'Rating':           rating,
                'Recommendation':   recommendation,
                'Stop_Loss':        round(stop_loss, 2),
                'SL_Percentage':    round(sl_percentage, 2),
                'Target_1':         round(target_1, 2),
                'Target_2':         round(target_2, 2),
                'Target_Price':     round(target_price, 2) if target_price else 0,
                'Upside':           round(upside, 2),
                'Risk_Reward':      risk_reward,
                'Targets_Hit':      targets_hit,
                'Target_Status':    target_status,
                'Analyst':          analyst_label,
                'Earnings_Date':    earnings_date,
                'Earn_Soon':        earn_soon,
            }
        except Exception:
            return None

    # =========================================================================
    #  ANALYZE ALL
    # =========================================================================
    def analyze_all_stocks(self):
        print(f"🔍 Analyzing {len(self.sp500_stocks)} US stocks...")
        print("⏳ ~2-3 minutes...\n")
        for idx, (symbol, name) in enumerate(self.sp500_stocks.items(), 1):
            result = self.analyze_stock(symbol, name)
            if result:
                self.results.append(result)
            if idx % 10 == 0:
                print(f"  [{idx}/{len(self.sp500_stocks)}] {name}")
        print(f"\n✅ {len(self.results)} stocks analyzed\n")

    # =========================================================================
    #  TOP RECOMMENDATIONS
    # =========================================================================
    def get_top_recommendations(self):
        df = pd.DataFrame(self.results)

        # == BUY side ===========================================================
        all_buys = df[df['Recommendation'].isin(['STRONG BUY', 'BUY'])]
        print(f"\n📊 BUY Filter Debug ({len(all_buys)} rated BUY/STRONG BUY):")

        f1 = all_buys[all_buys['Upside'] > 0.5]
        f2 = f1[f1['Risk_Reward'] >= 0.5]
        f3 = f2[f2['Target_1'] > f2['Price']]

        # U12: RSI safety gate — mirrors Nifty v5.4
        def rsi_is_safe_to_buy(row):
            rsi_val = row.get('RSI', 50)
            rsi_dir = row.get('RSI_Direction', 'Flat')
            rsi_slp = row.get('RSI_Slope', 0)
            if rsi_val > 70:                              return False
            if rsi_val > 65 and rsi_dir == 'Falling':    return False
            if rsi_val > 60 and rsi_slp < -8:            return False
            return True

        f4 = f3[f3.apply(rsi_is_safe_to_buy, axis=1)]

        # v8: drop stocks where MACD histogram is shrinking while RSI is elevated
        f5 = f4[~((f4['MACD_Hist_Slope'] < -0.05) & (f4['RSI'] > 58))]

        # v9: drop stocks with high sell-off volume + falling RSI
        f6 = f5[~((f5['Vol_Ratio'] > 2.0) & (f5['RSI_Slope'] < 0))]

        print(f"   {len(all_buys)} → {len(f1)} upside → {len(f2)} RR → "
              f"{len(f3)} T1>Price → {len(f4)} RSI-safe → "
              f"{len(f5)} MACD → {len(f6)} vol+momentum")

        # Log v9 filtered stocks
        v9_filtered = f5[((f5['Vol_Ratio'] > 2.0) & (f5['RSI_Slope'] < 0))]
        if not v9_filtered.empty:
            print(f"   ⚠  v9 removed (high sell-off volume — check tomorrow):")
            for _, r in v9_filtered.iterrows():
                print(f"      {r['Symbol']:6s}  Vol×{r['Vol_Ratio']:.1f}  "
                      f"RSI slope {r['RSI_Slope']:+.1f}  RSI {r['RSI']:.0f}")

        sorted_buys = f6.sort_values('Combined_Score', ascending=False)

        # U22: Sector diversity cap — max 4 per sector in top 20
        top_buys_rows = []
        sector_counts = {}
        for _, row in sorted_buys.iterrows():
            sec = row.get('Sector', 'N/A')
            sector_counts[sec] = sector_counts.get(sec, 0)
            if sector_counts[sec] < MAX_PICKS_PER_SECTOR:
                top_buys_rows.append(row)
                sector_counts[sec] += 1
            if len(top_buys_rows) >= 20:
                break
        top_buys = pd.DataFrame(top_buys_rows)

        # == SELL side ==========================================================
        all_sells = df[df['Recommendation'].isin(['STRONG SELL', 'SELL'])]
        s1 = all_sells[all_sells['Upside'] > 0.5]
        s2 = s1[s1['Risk_Reward'] >= 0.5]
        s3 = s2[s2['Target_1'] < s2['Price']]

        # U23: Sell gate — mirrors Nifty v5.4 sell validation
        def sell_is_valid(row):
            rsi_val = row.get('RSI', 50)
            rsi_dir = row.get('RSI_Direction', 'Flat')
            rsi_slp = row.get('RSI_Slope', 0)
            macd    = row.get('MACD', 'Bearish')
            # Block: Oversold — bounce risk, not a short entry
            if rsi_val < 35:
                return False
            # Block: RSI healthy + MACD Bullish — chart says uptrend
            if rsi_val > 50 and macd == 'Bullish':
                return False
            # Block: RSI rising fast from low base — recovery in progress
            if rsi_val < 45 and rsi_dir == 'Rising' and rsi_slp > 8:
                return False
            return True

        top_sells = s3[s3.apply(sell_is_valid, axis=1)].nsmallest(20, 'Combined_Score')
        print(f"   SELL: {len(all_sells)} → {len(s3)} filtered → "
              f"{len(top_sells)} final\n")
        return top_buys, top_sells

    # =========================================================================
    #  HTML — Nifty50-style compact always-detail no-scroll layout v5.4
    # =========================================================================
    # =========================================================================
    #  HTML — Nifty50-style: sticky header, all columns, no toggle, no h-scroll
    # =========================================================================
    def generate_email_html(self):
        df = pd.DataFrame(self.results)
        top_buys, top_sells = self.get_top_recommendations()

        now         = self.get_est_time()
        idx_data    = self.fetch_index_data()
        time_of_day = "Morning" if now.hour < 12 else "Evening"
        next_update = "4:30 PM" if now.hour < 12 else "9:30 AM (Next Day)"

        strong_buy_count  = len(df[df['Recommendation'] == 'STRONG BUY'])
        buy_count         = len(df[df['Recommendation'] == 'BUY'])
        hold_count        = len(df[df['Recommendation'] == 'HOLD'])
        sell_count        = len(df[df['Recommendation'] == 'SELL'])
        strong_sell_count = len(df[df['Recommendation'] == 'STRONG SELL'])

        if not top_buys.empty:
            sector_summary = top_buys['Sector'].value_counts().head(4)
            sector_kpi = ' · '.join([f"{s}: {c}" for s, c in sector_summary.items()])
        else:
            sector_kpi = 'No buys'

        ticker_items = ''
        for t in self.results[:12]:
            pct  = ((t['Price'] - t['SMA_20']) / t['SMA_20']) * 100
            cls  = 'tick-up' if pct >= 0 else 'tick-dn'
            sign = '+' if pct >= 0 else ''
            ticker_items += (
                f'<span class="tick">'
                f'<span class="tick-sym">{t["Symbol"]}</span>'
                f'<span class="tick-px">${t["Price"]:,.2f}</span>'
                f'<span class="{cls}">{sign}{pct:.1f}%</span>'
                f'</span>'
            )
        ticker_html = ticker_items + ticker_items   # doubled for seamless loop

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0">
<title>US Market Influencers - {time_of_day} · {now.strftime('%d %b %Y')}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#04080f; --bg2:#060d18; --surf:#0a1628; --surf2:#0c1a2e;
  --bdr:#1e3a5a; --bdr2:#2a4a6a;
  --accent:#00f5ff; --green:#00ff88; --red:#ff4466;
  --gold:#ffcc00; --purple:#cc99ff; --text:#ddeeff; --muted:#aaccee;
}}
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#04080f;color:#ddeeff;font-family:'Space Grotesk',sans-serif;font-size:12px;min-height:100vh;}}

/* ── STICKY HEADER ── */
header{{background:#060d18;border-bottom:2px solid #00f5ff;position:sticky;top:0;z-index:100;box-shadow:0 4px 24px rgba(0,0,0,0.85);}}
.h-top{{display:flex;align-items:center;justify-content:space-between;padding:8px 18px;gap:12px;flex-wrap:wrap;}}
.brand{{display:flex;align-items:center;gap:10px;}}
.brand-gem{{width:36px;height:36px;flex-shrink:0;background:linear-gradient(135deg,#00d4ff,#7c4dff);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 16px rgba(0,212,255,0.3);}}
.brand-name{{font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#fff;letter-spacing:-0.5px;}}
.brand-sub{{font-size:9px;color:#aaddff;letter-spacing:2px;text-transform:uppercase;margin-top:2px;font-weight:700;}}
.idx-strip{{display:flex;align-items:center;background:rgba(0,0,0,0.4);border:1px solid var(--bdr2);border-radius:8px;overflow:hidden;}}
.idx-item{{display:flex;flex-direction:column;align-items:center;padding:5px 16px;border-right:1px solid var(--bdr);gap:1px;}}
.idx-item:last-child{{border-right:none;}}
.idx-name{{font-size:9px;font-weight:800;letter-spacing:2px;color:#aaddff;text-transform:uppercase;}}
.idx-price{{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:800;color:#fff;}}
.idx-chg{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:800;}}
.up{{color:#00ff88;}} .dn{{color:#ff4466;}}
.clock-box{{display:flex;flex-direction:column;align-items:flex-end;gap:2px;}}
.clock-time{{font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:800;color:#00ff88;text-shadow:0 0 12px rgba(0,255,136,0.6);letter-spacing:1px;}}
.clock-meta{{font-size:10px;color:#fff;letter-spacing:1px;font-weight:700;}}
.clock-next{{font-size:10px;color:#aaddff;font-weight:700;}}

/* ── TICKER ── */
.ticker{{background:rgba(0,0,0,0.6);border-top:1px solid var(--bdr);overflow:hidden;}}
.ticker-inner{{display:flex;white-space:nowrap;animation:scroll 50s linear infinite;padding:4px 0;}}
@keyframes scroll{{0%{{transform:translateX(0);}}100%{{transform:translateX(-50%);}}}}
.tick{{display:inline-flex;align-items:center;gap:6px;padding:0 16px;border-right:1px solid #1e3a5a;font-family:'IBM Plex Mono',monospace;font-size:11px;}}
.tick-sym{{color:#00f5ff;font-weight:800;}} .tick-px{{color:#fff;font-weight:600;}}
.tick-up{{color:#00ff88;font-weight:700;}} .tick-dn{{color:#ff4466;font-weight:700;}}

/* ── KPI BAND ── */
.kpi-band{{display:flex;align-items:center;background:#080f1e;border-bottom:2px solid #1e3a5a;}}
.kpi-item{{display:flex;flex-direction:column;align-items:center;padding:12px 20px;border-right:1px solid #1e3a5a;flex:1;}}
.kpi-item:last-child{{border-right:none;}}
.kpi-num{{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;line-height:1;}}
.kpi-label{{font-size:9px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#aaddff;margin-top:4px;}}
.kpi-bar{{height:3px;width:40px;border-radius:2px;margin-top:6px;}}
.kpi-sub{{font-size:8px;color:#6699bb;margin-top:3px;font-weight:600;letter-spacing:0.5px;text-align:center;}}

/* ── MAIN ── */
.main{{padding:14px 18px;}}
.sec-hdr{{display:flex;align-items:center;gap:10px;margin-bottom:10px;}}
.sec-pill{{display:flex;align-items:center;gap:6px;padding:6px 16px;border-radius:100px;font-size:11px;font-weight:800;letter-spacing:0.5px;}}
.pill-buy{{background:#004d25;color:#00ff88;border:2px solid #00cc66;}}
.pill-sell{{background:#4d0010;color:#ff4466;border:2px solid #cc0033;}}
.sec-line{{flex:1;height:1px;background:#1e3a5a;}}
.sec-note{{font-size:9px;color:#88aacc;letter-spacing:1.5px;white-space:nowrap;font-weight:800;text-transform:uppercase;}}

/* ── TABLE — no min-width: fills screen, compact cells ── */
.tbl-wrap{{width:100%;overflow-x:auto;border:1px solid #1e3a5a;border-radius:10px;margin-bottom:22px;box-shadow:0 8px 40px rgba(0,0,0,0.6);background:#080f1e;-webkit-overflow-scrolling:touch;}}
table{{width:100%;border-collapse:collapse;table-layout:auto;}}
.grp-row th{{font-size:9px;font-weight:800;letter-spacing:2px;text-transform:uppercase;padding:6px 6px;text-align:center;border-bottom:1px solid rgba(255,255,255,0.08);white-space:nowrap;}}
.grp-stock{{background:#0d3a42;color:#00f5ff;}}
.grp-trade{{background:#0a3320;color:#00ff88;}}
.grp-tech{{background:#0a2a40;color:#40c8ff;}}
.grp-fund{{background:#3a2a00;color:#ffcc00;}}
.grp-meta{{background:#28124a;color:#cc99ff;}}
.col-row th{{font-size:9px;font-weight:800;letter-spacing:0.5px;text-transform:uppercase;padding:7px 6px;color:#fff;background:#0c1a2e;border-bottom:2px solid #1e3a5a;white-space:nowrap;text-align:left;}}
.ch-stock{{border-top:3px solid #00f5ff;color:#b0f0ff;}}
.ch-trade{{border-top:3px solid #00ff88;color:#b0ffe0;}}
.ch-tech{{border-top:3px solid #40c8ff;color:#c0e8ff;}}
.ch-fund{{border-top:3px solid #ffcc00;color:#fff0a0;}}
.ch-meta{{border-top:3px solid #cc99ff;color:#e8d0ff;}}
.gsep{{border-left:2px solid rgba(255,255,255,0.08)!important;}}
td{{padding:8px 6px;border-bottom:1px solid #0e2040;vertical-align:middle;white-space:nowrap;}}
tr:last-child td{{border-bottom:none;}}
tr:nth-child(even) td{{background:rgba(255,255,255,0.02);}}
tr:hover td{{background:rgba(0,212,255,0.06);transition:background 0.12s;}}

/* ── STOCK ── */
.sname{{font-size:11px;font-weight:700;color:#fff;line-height:1.2;}}
.ssym{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:#00f5ff;font-weight:700;letter-spacing:1px;margin-top:2px;}}
.ssec{{font-size:8px;color:#88bbdd;margin-top:1px;max-width:100px;overflow:hidden;text-overflow:ellipsis;font-weight:600;}}
.pval{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;color:#ffcc00;}}

/* ── RATING / SCORE ── */
.badge{{display:inline-flex;align-items:center;font-size:8px;font-weight:800;padding:3px 7px;border-radius:4px;letter-spacing:0.5px;white-space:nowrap;}}
.b-sb{{background:#004d25;color:#00ff88;border:1px solid #00ff88;}}
.b-b {{background:#003a4d;color:#00f5ff;border:1px solid #00f5ff;}}
.b-h {{background:#1a2a3a;color:#aabbcc;border:1px solid #445566;}}
.b-s {{background:#4d0010;color:#ff4466;border:1px solid #ff4466;}}
.b-ss{{background:#5a0015;color:#ff7788;border:1px solid #ff7788;}}
.sc-wrap{{display:flex;flex-direction:column;align-items:center;gap:3px;margin-top:3px;}}
.sc-num{{font-family:'Syne',sans-serif;font-size:18px;font-weight:800;line-height:1;}}
.sc-bar{{width:36px;height:3px;background:#1a2a3a;border-radius:2px;}}
.sc-fill{{height:100%;border-radius:2px;}}
.veto{{margin-top:2px;display:inline-block;padding:1px 5px;border-radius:3px;background:#2a1500;color:#ff8c00;border:1px solid #ff8c00;font-size:8px;font-weight:700;font-family:'IBM Plex Mono',monospace;}}
.warn2{{color:#f59e0b;border-color:#f59e0b;background:#1a1000;}}

/* ── TRADE SETUP ── */
.tbadge{{font-size:8px;font-weight:800;padding:2px 5px;border-radius:3px;letter-spacing:0.3px;display:block;margin-bottom:3px;white-space:nowrap;}}
.tb-r{{background:#004d25;color:#00ff88;border:1px solid #00ff88;}}
.tb-p{{background:#4d3300;color:#ffcc00;border:1px solid #ffcc00;}}
.tb-a{{background:#003a4d;color:#00f5ff;border:1px solid #00f5ff;}}
.t1{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;color:#fff;}}
.t2{{font-size:9px;color:#88bbdd;margin-top:1px;font-weight:700;}}
.slv{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;color:#ff4466;}}
.slvs{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;color:#ffcc00;}}
.slp{{font-size:9px;color:#88bbdd;margin-top:1px;font-weight:700;}}
.slt{{font-size:8px;font-weight:800;padding:2px 5px;border-radius:3px;margin-top:2px;display:inline-block;}}
.slt-a{{background:#004d25;color:#00ff88;border:1px solid #00cc66;}}
.slt-b{{background:#4d3300;color:#ffcc00;border:1px solid #cc9900;}}
.upv{{font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:800;}}
.upv.up{{color:#00ff88;}} .upv.dn{{color:#ff4466;}}
.rrv{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:800;}}
.atrv{{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;color:#00f5ff;}}
.atrs{{font-size:8px;color:#88bbdd;margin-top:1px;font-weight:700;}}

/* ── TECHNICALS ── */
.rsiv{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;}}
.rsis{{font-size:8px;color:#88bbdd;margin-top:1px;font-weight:700;max-width:95px;white-space:normal;line-height:1.3;}}
.divb{{font-size:8px;font-weight:800;padding:2px 5px;border-radius:3px;white-space:nowrap;display:inline-block;margin-top:2px;}}
.db-bear{{background:#4d0010;color:#ff4466;border:1px solid #ff4466;}}
.db-bull{{background:#004d25;color:#00ff88;border:1px solid #00ff88;}}
.rsi-slope{{font-family:'IBM Plex Mono',monospace;font-size:9px;display:inline-block;margin-top:2px;}}
.adxv{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;}}
.adxl{{font-size:8px;color:#88bbdd;margin-top:1px;font-weight:700;}}
.adx-s{{color:#00ff88;}} .adx-m{{color:#ffcc00;}} .adx-w{{color:#aabbcc;}}
.volv{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;}}
.voll{{font-size:8px;color:#88bbdd;margin-top:1px;font-weight:700;}}
.vol-h{{color:#00ff88;}} .vol-n{{color:#ddeeff;}} .vol-l{{color:#aabbcc;}}
.sdv{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;}}
.sd-c{{color:#00ff88;}} .sd-m{{color:#ffcc00;}} .sd-f{{color:#ff4466;}}
.mono{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;}}
.macd-bull{{color:#00ff88;font-weight:800;font-size:10px;}}
.macd-bear{{color:#ff4466;font-weight:800;font-size:10px;}}

/* ── FUNDAMENTALS ── */
.qb{{font-size:8px;font-weight:800;padding:3px 6px;border-radius:4px;}}
.q-ex{{background:#004d25;color:#00ff88;border:1px solid #00cc66;}}
.q-gd{{background:#003a4d;color:#00f5ff;border:1px solid #0099bb;}}
.q-av{{background:#4d3300;color:#ffcc00;border:1px solid #cc9900;}}
.q-po{{background:#4d0010;color:#ff4466;border:1px solid #cc0033;}}

/* ── META ── */
.ab{{font-size:8px;font-weight:800;padding:3px 6px;border-radius:4px;white-space:nowrap;}}
.ab-sb{{background:#004d25;color:#00ff88;border:1px solid #00cc66;}}
.ab-b {{background:#003a4d;color:#00f5ff;border:1px solid #0099bb;}}
.ab-h {{background:#1a2a3a;color:#aabbcc;border:1px solid #334455;}}
.ab-s {{background:#4d0010;color:#ff4466;border:1px solid #cc0033;}}
.earn{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#00f5ff;font-weight:700;}}
.earn-soon{{color:#ff4466;animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:0.4;}}}}
.actn{{display:inline-block;padding:3px 8px;border-radius:4px;font-size:9px;font-weight:800;letter-spacing:0.3px;white-space:nowrap;}}
.a-sb{{background:#004d25;color:#00ff88;border:1px solid #00ff88;}}
.a-b {{background:#003a4d;color:#00f5ff;border:1px solid #00f5ff;}}
.a-h {{background:#2a2200;color:#ffab00;border:1px solid #ffab00;}}
.a-s {{background:#4d0010;color:#ff4466;border:1px solid #ff4466;}}
.a-ss{{background:#5a0015;color:#ff7788;border:1px solid #ff7788;}}
.rnum{{font-size:10px;color:#88bbdd;font-weight:700;}}
.vol-warn{{font-size:8px;font-weight:700;padding:1px 4px;border-radius:2px;background:rgba(249,115,22,0.15);color:#fb923c;border:1px solid rgba(249,115,22,0.3);margin-left:3px;}}

/* ── FOOTER ── */
.disc{{background:#0c1a2e;border:1px solid #1e3a5a;border-left:4px solid #ff4466;padding:10px 14px;border-radius:8px;font-size:10px;color:#aaccee;line-height:1.8;margin:12px 0;}}
footer{{text-align:center;padding:12px;background:#080f1e;border-top:1px solid #1e3a5a;font-size:10px;color:#88aacc;letter-spacing:1px;}}
footer strong{{color:#00f5ff;}}

/* ── RESPONSIVE ── */
@media(max-width:900px){{.idx-strip{{display:none;}}.kpi-item{{padding:8px 10px;}}.kpi-num{{font-size:22px;}}}}
@media(max-width:600px){{.h-top{{padding:6px 10px;}}.main{{padding:8px;}}.kpi-band{{flex-wrap:wrap;}}.kpi-item{{flex:0 0 50%;border-bottom:1px solid var(--bdr);}}.brand-name{{font-size:12px;}}}}
</style>
</head>
<body>

<header>
  <div class="h-top">
    <div class="brand">
      <div class="brand-gem">🌅</div>
      <div>
        <div class="brand-name">US Market Influencers · NASDAQ &amp; S&amp;P 500</div>
        <div class="brand-sub">12M S/R · Trend Veto · Wilder RSI · Dynamic Weights · Sector PE · v5.4</div>
      </div>
    </div>
    <div class="idx-strip">
      <div class="idx-item"><span class="idx-name">DJI</span><span class="idx-price">{idx_data['DJI']['price']}</span><span class="idx-chg {idx_data['DJI']['cls']}">{idx_data['DJI']['chg']}</span></div>
      <div class="idx-item"><span class="idx-name">NDX</span><span class="idx-price">{idx_data['NDX']['price']}</span><span class="idx-chg {idx_data['NDX']['cls']}">{idx_data['NDX']['chg']}</span></div>
      <div class="idx-item"><span class="idx-name">SPX</span><span class="idx-price">{idx_data['SPX']['price']}</span><span class="idx-chg {idx_data['SPX']['cls']}">{idx_data['SPX']['chg']}</span></div>
    </div>
    <div class="clock-box">
      <div class="clock-time" id="liveClock">--:--:-- --</div>
      <div class="clock-meta" id="liveDate">{now.strftime('%d %b %Y')} · EST</div>
      <div class="clock-next">Report: {now.strftime('%d %b %Y %I:%M %p')} EST</div>
      <div class="clock-next">Next Update: <strong style="color:#00f5ff">{next_update}</strong></div>
    </div>
  </div>
  <div class="ticker"><div class="ticker-inner">{ticker_html}</div></div>
</header>

<div class="kpi-band">
  <div class="kpi-item"><div class="kpi-num" style="color:#00f5ff">{len(self.results)}</div><div class="kpi-label">Analyzed</div><div class="kpi-bar" style="background:#00f5ff"></div></div>
  <div class="kpi-item"><div class="kpi-num" style="color:#00ff88">{strong_buy_count}</div><div class="kpi-label">Strong Buy</div><div class="kpi-bar" style="background:#00ff88"></div></div>
  <div class="kpi-item"><div class="kpi-num" style="color:#00d4ff">{buy_count}</div><div class="kpi-label">Buy</div><div class="kpi-bar" style="background:#00d4ff"></div></div>
  <div class="kpi-item"><div class="kpi-num" style="color:#ff4466">{sell_count + strong_sell_count}</div><div class="kpi-label">Sell / Strong Sell</div><div class="kpi-bar" style="background:#ff4466"></div></div>
  <div class="kpi-item"><div class="kpi-num" style="color:#60a5fa">{hold_count}</div><div class="kpi-label">Hold</div><div class="kpi-bar" style="background:#60a5fa"></div><div class="kpi-sub">{sector_kpi}</div></div>
</div>

<div class="main">
"""

        # ── HELPERS ──────────────────────────────────────────────────────────
        def badge(rec):
            m = {'STRONG BUY':'b-sb','BUY':'b-b','HOLD':'b-h','SELL':'b-s','STRONG SELL':'b-ss'}
            labels = {'STRONG BUY':'⭐⭐⭐⭐⭐ STRONG BUY','BUY':'⭐⭐⭐⭐ BUY',
                      'HOLD':'⭐⭐⭐ HOLD','SELL':'⭐⭐ SELL','STRONG SELL':'⭐ STRONG SELL'}
            return f'<span class="badge {m.get(rec,"b-h")}">{labels.get(rec,rec)}</span>'

        def action_btn(rec):
            m = {'STRONG BUY':('a-sb','⭐⭐ STRONG BUY'),'BUY':('a-b','▲ BUY'),
                 'HOLD':('a-h','◆ HOLD'),'SELL':('a-s','▼ SELL'),'STRONG SELL':('a-ss','⚠ STRONG SELL')}
            cls, lbl = m.get(rec, ('a-h','◆ HOLD'))
            return f'<span class="actn {cls}">{lbl}</span>'

        def score_cell(val, col, bar):
            pct = min(int(val), 100)
            return (f'<div class="sc-wrap">'
                    f'<div class="sc-num" style="color:{col}">{val:.0f}</div>'
                    f'<div class="sc-bar"><div class="sc-fill" style="width:{pct}%;background:{bar}"></div></div>'
                    f'</div>')

        def veto_pill(bs):
            if bs >= 3:
                return f'<div class="veto">🚫 Veto ({bs}/6)</div>'
            if bs >= 2:
                return f'<div class="veto warn2">⚠ {bs} signals</div>'
            return ''

        def tbadge(ts):
            if 'ATH' in ts:      return 'tb-a', '🚀 ATH Zone'
            elif 'Partial' in ts or 'Hold' in ts: return 'tb-p', '⚡ Partial S/R'
            return 'tb-r', '📍 Real S/R'

        def div_badge(div):
            if div == 'Bearish Divergence':
                return '<span class="divb db-bear">⚠ Bear Div</span>'
            elif div == 'Bullish Divergence':
                return '<span class="divb db-bull">✅ Bull Div</span>'
            return ''

        def slope_html(direction, slope):
            if direction == 'Rising':
                return f'<span class="rsi-slope" style="color:#00ff88">↑ +{slope:.0f}</span>'
            elif direction == 'Falling':
                c = '#ff4466' if abs(slope) > 8 else '#ffcc00'
                return f'<span class="rsi-slope" style="color:{c}">↓ {slope:.0f}</span>'
            return '<span class="rsi-slope" style="color:#4a6080">→</span>'

        def adx_cell(v):
            if v >= 30:   c,l = 'adx-s','Strong'
            elif v >= 20: c,l = 'adx-m','Moderate'
            else:         c,l = 'adx-w','Weak'
            return f'<div class="adxv {c}">{v:.0f}</div><div class="adxl">{l}</div>'

        def vol_cell(v):
            c = 'vol-h' if v >= 1.5 else ('vol-l' if v < 0.7 else 'vol-n')
            l = 'High' if v >= 1.5 else ('Low' if v < 0.7 else 'Avg')
            return f'<div class="volv {c}">{v:.1f}×</div><div class="voll">{l} Vol</div>'

        def sdist(v):
            c = 'sd-c' if v <= 3 else ('sd-m' if v <= 8 else 'sd-f')
            return f'<span class="sdv {c}">{v:.1f}%</span>'

        def ab(label):
            m = {'Strong Buy':'ab-sb','Buy':'ab-b','Hold':'ab-h','Sell':'ab-s','Strong Sell':'ab-s'}
            return f'<span class="ab {m.get(label,"ab-h")}">{label}</span>'

        def qb(q):
            m = {'Excellent':'q-ex','Good':'q-gd','Average':'q-av','Poor':'q-po'}
            return f'<span class="qb {m.get(q,"q-av")}">{q}</span>'

        def rr_c(v):   return '#00e676' if v >= 2 else ('#00d4ff' if v >= 1.5 else ('#ffab00' if v >= 1 else '#ff3d57'))
        def pe_c(v,d='b'): 
            if v <= 0: return '#4a6080'
            return ('#00e676' if v < 25 else ('#ffab00' if v < 40 else '#ff3d57')) if d=='b' else ('#ff3d57' if v > 40 else ('#ffab00' if v > 25 else '#00e676'))
        def w52_c(p):  return '#ff3d57' if p >= -5 else ('#ffab00' if p >= -20 else '#00e676')
        def beta_c(v): return '#ff3d57' if v > 1.5 else ('#ffab00' if v > 1.0 else '#00e676')
        def sc_c(v):   return '#00e676' if v >= 75 else ('#00d4ff' if v >= 55 else '#ffab00')
        def sc_b(v):   return '#00c853' if v >= 75 else ('#0099cc' if v >= 55 else '#f59e0b')

        # ── BUY TABLE ─────────────────────────────────────────────────────────
        if not top_buys.empty:
            html += """
  <div class="sec-hdr">
    <div class="sec-pill pill-buy">▲ Top Buy Recommendations — Sector Diversified</div>
    <div class="sec-line"></div>
    <div class="sec-note">STOCK INFO · TRADE SETUP · TECHNICALS · FUNDAMENTALS · META</div>
  </div>
  <div class="tbl-wrap"><table>
    <thead>
      <tr class="grp-row">
        <th class="grp-stock" colspan="3">STOCK INFO</th>
        <th class="grp-trade gsep" colspan="6">TRADE SETUP</th>
        <th class="grp-tech gsep"  colspan="6">TECHNICALS</th>
        <th class="grp-fund gsep"  colspan="4">FUNDAMENTALS</th>
        <th class="grp-meta gsep"  colspan="3">META</th>
      </tr>
      <tr class="col-row">
        <th class="ch-stock">#</th>
        <th class="ch-stock">Stock / Sector</th>
        <th class="ch-stock">Price</th>
        <th class="ch-trade gsep">Rating / Score</th>
        <th class="ch-trade">Upside</th>
        <th class="ch-trade">Target S/R</th>
        <th class="ch-trade">Stop Loss</th>
        <th class="ch-trade">ATR</th>
        <th class="ch-trade">R:R</th>
        <th class="ch-tech gsep">RSI / Div</th>
        <th class="ch-tech">ADX</th>
        <th class="ch-tech">Vol</th>
        <th class="ch-tech">Sup Dist</th>
        <th class="ch-tech">52W Hi</th>
        <th class="ch-tech">MACD</th>
        <th class="ch-fund gsep">P/E</th>
        <th class="ch-fund">Beta</th>
        <th class="ch-fund">Div%</th>
        <th class="ch-fund">Quality</th>
        <th class="ch-meta gsep">Analyst</th>
        <th class="ch-meta">Earnings</th>
        <th class="ch-meta">Action</th>
      </tr>
    </thead>
    <tbody>
"""
            for i, (_, row) in enumerate(top_buys.iterrows(), 1):
                rec     = row['Recommendation']
                w52     = ((row['Price'] - row['52W_High']) / row['52W_High']) * 100
                tbc,tbt = tbadge(row.get('Target_Status',''))
                st      = row.get('Stop_Type','ATR Stop')
                sltcls  = 'slt-a' if st == 'ATR Stop' else 'slt-b'
                sltlbl  = '📐 ATR' if st == 'ATR Stop' else '🔒 Beta'
                dy      = f"{row['Dividend_Yield']:.2f}%" if row['Dividend_Yield'] > 0 else '-'
                dyc     = '#00e676' if row['Dividend_Yield'] > 0 else '#4a6080'
                rr      = row['Risk_Reward']
                bs      = row.get('Bearish_Signals', 0)
                ec      = 'earn-soon' if row.get('Earn_Soon', False) else ''
                vw      = row.get('Vol_Ratio', 1.0) > 2.0 and row.get('RSI_Slope', 0) < 0
                vwb     = '<span class="vol-warn">⚠ Vol×' + f"{row.get('Vol_Ratio',1.0):.1f}" + '</span>' if vw else ''
                html += f"""      <tr>
        <td><span class="rnum">{i}</span></td>
        <td><div class="sname">{row['Name']}{vwb}</div><div class="ssym">{row['Symbol']}</div><div class="ssec">{row.get('Sector','N/A')}</div></td>
        <td><div class="pval">${row['Price']:,.2f}</div></td>
        <td class="gsep">{badge(rec)}{score_cell(row['Combined_Score'],sc_c(row['Combined_Score']),sc_b(row['Combined_Score']))}{veto_pill(bs)}</td>
        <td><span class="upv {'up' if row['Upside']>=0 else 'dn'}">{row['Upside']:+.1f}%</span></td>
        <td><span class="tbadge {tbc}">{tbt}</span><div class="t1">${row['Target_1']:,.2f}</div><div class="t2">T2: ${row['Target_2']:,.2f}</div></td>
        <td><div class="slv">${row['Stop_Loss']:,.2f}</div><div class="slp">-{row['SL_Percentage']:.1f}%</div><span class="slt {sltcls}">{sltlbl}</span></td>
        <td><div class="atrv">${row['ATR']:,.2f}</div><div class="atrs">{row['ATR_Pct']:.1f}% · {row['ATR_Multiplier']}×</div></td>
        <td><span class="rrv" style="color:{rr_c(rr)}">{rr:.1f}×</span></td>
        <td class="gsep"><div class="rsiv" style="color:{'#ff3d57' if row['RSI']>70 else ('#00e676' if row['RSI']<30 else '#60a5fa')}">{row['RSI']:.0f}</div><div class="rsis">{row['RSI_Signal']}</div>{div_badge(row.get('RSI_Divergence','None'))}{slope_html(row.get('RSI_Direction','Flat'),row.get('RSI_Slope',0))}</td>
        <td>{adx_cell(row.get('ADX',0))}</td>
        <td>{vol_cell(row.get('Vol_Ratio',1.0))}</td>
        <td>{sdist(row.get('Support_Dist_Pct',0))}</td>
        <td><span class="mono" style="color:{w52_c(w52)}">{w52:+.1f}%</span></td>
        <td><span class="{'macd-bull' if row['MACD']=='Bullish' else 'macd-bear'}">{row['MACD']}</span></td>
        <td class="gsep"><span class="mono" style="color:{pe_c(row['PE_Ratio'],'b')}">{f"{row['PE_Ratio']:.1f}" if row['PE_Ratio']>0 else 'N/A'}</span></td>
        <td><span class="mono" style="color:{beta_c(row['Beta'])}">{row['Beta']:.2f}</span></td>
        <td><span class="mono" style="color:{dyc}">{dy}</span></td>
        <td>{qb(row['Quality'])}</td>
        <td class="gsep">{ab(row.get('Analyst','N/A'))}</td>
        <td><div class="earn {ec}">{row.get('Earnings_Date','N/A')}</div></td>
        <td>{action_btn(rec)}</td>
      </tr>
"""
            html += "    </tbody></table></div>\n"

        # ── SELL TABLE ────────────────────────────────────────────────────────
        if not top_sells.empty:
            html += """
  <div class="sec-hdr">
    <div class="sec-pill pill-sell">▼ Top 20 Sell Recommendations</div>
    <div class="sec-line"></div>
    <div class="sec-note">STOCK INFO · TRADE SETUP · TECHNICALS · FUNDAMENTALS · META</div>
  </div>
  <div class="tbl-wrap"><table>
    <thead>
      <tr class="grp-row">
        <th class="grp-stock" colspan="3">STOCK INFO</th>
        <th class="grp-trade gsep" colspan="6">TRADE SETUP</th>
        <th class="grp-tech gsep"  colspan="5">TECHNICALS</th>
        <th class="grp-fund gsep"  colspan="4">FUNDAMENTALS</th>
        <th class="grp-meta gsep"  colspan="3">META</th>
      </tr>
      <tr class="col-row">
        <th class="ch-stock">#</th>
        <th class="ch-stock">Stock / Sector</th>
        <th class="ch-stock">Price</th>
        <th class="ch-trade gsep">Rating / Score</th>
        <th class="ch-trade">Downside</th>
        <th class="ch-trade">Target S/R</th>
        <th class="ch-trade">Stop Loss</th>
        <th class="ch-trade">ATR</th>
        <th class="ch-trade">R:R</th>
        <th class="ch-tech gsep">RSI / Signal</th>
        <th class="ch-tech">ADX</th>
        <th class="ch-tech">Vol</th>
        <th class="ch-tech">52W Hi</th>
        <th class="ch-tech">MACD</th>
        <th class="ch-fund gsep">P/E</th>
        <th class="ch-fund">Beta</th>
        <th class="ch-fund">Div%</th>
        <th class="ch-fund">Quality</th>
        <th class="ch-meta gsep">Analyst</th>
        <th class="ch-meta">Earnings</th>
        <th class="ch-meta">Action</th>
      </tr>
    </thead>
    <tbody>
"""
            for i, (_, row) in enumerate(top_sells.iterrows(), 1):
                rec     = row['Recommendation']
                w52     = ((row['Price'] - row['52W_High']) / row['52W_High']) * 100
                tbc,tbt = tbadge(row.get('Target_Status',''))
                st      = row.get('Stop_Type','ATR Stop')
                sltcls  = 'slt-a' if st == 'ATR Stop' else 'slt-b'
                sltlbl  = '📐 ATR' if st == 'ATR Stop' else '🔒 Beta'
                dy      = f"{row['Dividend_Yield']:.2f}%" if row['Dividend_Yield'] > 0 else '-'
                dyc     = '#00e676' if row['Dividend_Yield'] > 0 else '#4a6080'
                rr      = row['Risk_Reward']
                ec      = 'earn-soon' if row.get('Earn_Soon', False) else ''
                html += f"""      <tr>
        <td><span class="rnum">{i}</span></td>
        <td><div class="sname">{row['Name']}</div><div class="ssym">{row['Symbol']}</div><div class="ssec">{row.get('Sector','N/A')}</div></td>
        <td><div class="pval">${row['Price']:,.2f}</div></td>
        <td class="gsep">{badge(rec)}{score_cell(row['Combined_Score'],'#ff4466','#cc2244')}</td>
        <td><span class="upv {'dn' if row['Upside']>=0 else 'up'}">{row['Upside']:+.1f}%</span></td>
        <td><span class="tbadge {tbc}">{tbt}</span><div class="t1">${row['Target_1']:,.2f}</div><div class="t2">T2: ${row['Target_2']:,.2f}</div></td>
        <td><div class="slvs">${row['Stop_Loss']:,.2f}</div><div class="slp">+{row['SL_Percentage']:.1f}%</div><span class="slt {sltcls}">{sltlbl}</span></td>
        <td><div class="atrv">${row['ATR']:,.2f}</div><div class="atrs">{row['ATR_Pct']:.1f}% · {row['ATR_Multiplier']}×</div></td>
        <td><span class="rrv" style="color:{rr_c(rr)}">{rr:.1f}×</span></td>
        <td class="gsep"><div class="rsiv" style="color:{'#ff3d57' if row['RSI']>70 else ('#00e676' if row['RSI']<30 else '#60a5fa')}">{row['RSI']:.0f}</div><div class="rsis">{row['RSI_Signal']}</div>{slope_html(row.get('RSI_Direction','Flat'),row.get('RSI_Slope',0))}</td>
        <td>{adx_cell(row.get('ADX',0))}</td>
        <td>{vol_cell(row.get('Vol_Ratio',1.0))}</td>
        <td><span class="mono" style="color:{w52_c(w52)}">{w52:+.1f}%</span></td>
        <td><span class="{'macd-bull' if row['MACD']=='Bullish' else 'macd-bear'}">{row['MACD']}</span></td>
        <td class="gsep"><span class="mono" style="color:{pe_c(row['PE_Ratio'],'s')}">{f"{row['PE_Ratio']:.1f}" if row['PE_Ratio']>0 else 'N/A'}</span></td>
        <td><span class="mono" style="color:{beta_c(row['Beta'])}">{row['Beta']:.2f}</span></td>
        <td><span class="mono" style="color:{dyc}">{dy}</span></td>
        <td>{qb(row['Quality'])}</td>
        <td class="gsep">{ab(row.get('Analyst','N/A'))}</td>
        <td><div class="earn {ec}">{row.get('Earnings_Date','N/A')}</div></td>
        <td>{action_btn(rec)}</td>
      </tr>
"""
            html += "    </tbody></table></div>\n"

        html += f"""
  <div class="disc">
    <strong style="color:#ff4466">⚠ DISCLAIMER:</strong>
    For <strong>EDUCATIONAL PURPOSES ONLY</strong>. Not financial advice.
    Stop losses ATR-based near real 12-month S/R zones. Targets from swing highs/lows and 52W extremes.
    Earnings dates are estimates. RSI divergence is algorithmic — not guaranteed.
    Trend Veto caps BUY to HOLD when 3+ bearish signals fire. Consult a registered financial advisor.
  </div>
</div>

<footer>
  <strong>US Market Influencers · NASDAQ &amp; S&amp;P 500</strong>
  · Wilder RSI · Trend Veto · Dynamic Weights · SMA200 Slope · Sector PE · RSI Divergence · v5.4
  · Next Update: <strong>{next_update} EST</strong> · {now.strftime('%d %b %Y')}
</footer>

<script>
function tick(){{
  var d=new Date(),e=new Date(d.toLocaleString('en-US',{{timeZone:'America/New_York'}}));
  var h=e.getHours(),m=e.getMinutes(),s=e.getSeconds(),ap=h>=12?'PM':'AM';
  h=h%12||12;
  var p=n=>String(n).padStart(2,'0');
  document.getElementById('liveClock').textContent=p(h)+':'+p(m)+':'+p(s)+' '+ap+' EST';
  var mo=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  document.getElementById('liveDate').textContent=p(e.getDate())+' '+mo[e.getMonth()]+' '+e.getFullYear()+' · EST';
}}
tick(); setInterval(tick,1000);
</script>
</body></html>"""
        return html

    # =========================================================================
    #  EMAIL
    # =========================================================================
    def send_email(self, to_email):
        try:
            from_email = os.environ.get('GMAIL_USER')
            password   = os.environ.get('GMAIL_APP_PASSWORD')
            if not from_email or not password:
                print("❌ Set GMAIL_USER and GMAIL_APP_PASSWORD env vars"); return False
            now = self.get_est_time()
            tod = "Morning" if now.hour < 12 else "Evening"
            msg = MIMEMultipart('alternative')
            msg['From']    = from_email
            msg['To']      = to_email
            msg['Subject'] = f"🌅 US Market Report v5.4 — {tod} {now.strftime('%d %b %Y')}"
            msg.attach(MIMEText(self.generate_email_html(), 'html'))
            srv = smtplib.SMTP('smtp.gmail.com', 587)
            srv.starttls(); srv.login(from_email, password)
            srv.send_message(msg); srv.quit()
            print(f"✅ Email sent to {to_email}"); return True
        except Exception as e:
            print(f"❌ Email error: {e}"); return False

    # =========================================================================
    #  ENTRY POINT
    # =========================================================================
    def generate_complete_report(self, send_email_flag=True, recipient_email=None,
                                  output_file='index.html'):
        now = self.get_est_time()
        print("=" * 70)
        print("📊 S&P 500 ANALYZER v5.4 — Trend Veto · Wilder RSI · Dynamic Weights · Sector PE")
        print(f"   {now.strftime('%d %b %Y, %I:%M %p EST')}")
        print("=" * 70)
        self.analyze_all_stocks()
        html = self.generate_email_html()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ HTML report saved: {output_file}")
        if send_email_flag and recipient_email:
            self.send_email(recipient_email)
        print("=" * 70)
        print("✅ DONE")
        print("=" * 70)


# =============================================================================
#  RUN
# =============================================================================
if __name__ == "__main__":
    analyzer  = SP500CompleteAnalyzer()
    recipient = os.environ.get('RECIPIENT_EMAIL')
    analyzer.generate_complete_report(
        send_email_flag=bool(recipient),
        recipient_email=recipient,
        output_file=os.environ.get('OUTPUT_FILE', 'index.html')
    )
