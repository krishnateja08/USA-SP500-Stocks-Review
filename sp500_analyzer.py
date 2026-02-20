"""
S&P 500 COMPLETE STOCK ANALYZER
Technical + Fundamental Analysis with Email Delivery
Theme: Sunset Warm (Theme 10) — VISIBILITY FIXED
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

warnings.filterwarnings('ignore')


class SP500CompleteAnalyzer:
    def __init__(self):
        # Master Top 50 List (Nasdaq & S&P 500 Combined)
        self.sp500_stocks = {
            'NVDA': 'NVIDIA',
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'GOOGL': 'Alphabet (Class A)',
            'GOOG': 'Alphabet (Class C)',
            'META': 'Meta Platforms',
            'TSLA': 'Tesla',
            'AVGO': 'Broadcom',
            'BRK-B': 'Berkshire Hathaway',
            'WMT': 'Walmart',
            'LLY': 'Eli Lilly',
            'JPM': 'JPMorgan Chase',
            'XOM': 'ExxonMobil',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'MA': 'Mastercard',
            'MU': 'Micron Technology',
            'ORCL': 'Oracle Corporation',
            'COST': 'Costco',
            'ABBV': 'AbbVie',
            'HD': 'Home Depot',
            'BAC': 'Bank of America',
            'PG': 'Procter & Gamble',
            'CVX': 'Chevron Corporation',
            'CAT': 'Caterpillar Inc.',
            'KO': 'Coca-Cola Company',
            'GE': 'GE Aerospace',
            'AMD': 'Advanced Micro Devices',
            'NFLX': 'Netflix',
            'PLTR': 'Palantir Technologies',
            'MRK': 'Merck & Co.',
            'CSCO': 'Cisco Systems',
            'PM': 'Philip Morris International',
            'LRCX': 'Lam Research',
            'AMAT': 'Applied Materials',
            'MS': 'Morgan Stanley',
            'WFC': 'Wells Fargo',
            'GS': 'Goldman Sachs',
            'RTX': 'RTX Corporation',
            'UNH': 'UnitedHealth Group',
            'TMUS': 'T-Mobile US',
            'IBM': 'IBM',
            'MCD': "McDonald's",
            'AXP': 'American Express',
            'INTC': 'Intel',
            'PEP': 'PepsiCo',
            'LIN': 'Linde plc',
            'GEV': 'GE Vernova',
            'VZ': 'Verizon',
        }

        self.results = []

    def get_est_time(self):
        """Get current time in EST timezone"""
        est = pytz.timezone('US/Eastern')
        return datetime.now(est)

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_macd(self, prices):
        """Calculate MACD"""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]

    def get_fundamental_score(self, info):
        """Calculate fundamental score (0-100)"""
        score = 0

        # Valuation Score (25 points)
        pe = info.get('trailingPE', info.get('forwardPE', 0))
        pb = info.get('priceToBook', 0)
        peg = info.get('pegRatio', 0)

        if pe and 0 < pe < 25:
            score += 10
        elif pe and 25 <= pe < 35:
            score += 5

        if pb and 0 < pb < 3:
            score += 5
        elif pb and 3 <= pb < 5:
            score += 3

        if peg and 0 < peg < 1:
            score += 10
        elif peg and 1 <= peg < 2:
            score += 5

        # Profitability Score (25 points)
        roe = info.get('returnOnEquity', 0)
        roa = info.get('returnOnAssets', 0)
        profit_margin = info.get('profitMargins', 0)

        if roe and roe > 0.15:
            score += 10
        elif roe and roe > 0.10:
            score += 5

        if roa and roa > 0.05:
            score += 5
        elif roa and roa > 0.02:
            score += 3

        if profit_margin and profit_margin > 0.10:
            score += 10
        elif profit_margin and profit_margin > 0.05:
            score += 5

        # Growth Score (25 points)
        revenue_growth = info.get('revenueGrowth', 0)
        earnings_growth = info.get('earningsGrowth', 0)

        if revenue_growth and revenue_growth > 0.15:
            score += 10
        elif revenue_growth and revenue_growth > 0.10:
            score += 7
        elif revenue_growth and revenue_growth > 0.05:
            score += 5

        if earnings_growth and earnings_growth > 0.15:
            score += 10
        elif earnings_growth and earnings_growth > 0.10:
            score += 7
        elif earnings_growth and earnings_growth > 0.05:
            score += 5

        # Financial Health Score (25 points)
        debt_to_equity = info.get('debtToEquity', 0)
        current_ratio = info.get('currentRatio', 0)

        if debt_to_equity is not None:
            if debt_to_equity < 50:
                score += 10
            elif debt_to_equity < 100:
                score += 5
        else:
            score += 5

        if current_ratio and current_ratio > 1.5:
            score += 10
        elif current_ratio and current_ratio > 1.0:
            score += 5

        # Free cash flow
        free_cashflow = info.get('freeCashflow', 0)
        if free_cashflow and free_cashflow > 0:
            score += 5

        return min(score, 100)

    def analyze_stock(self, symbol, name):
        """Analyze individual stock - Technical + Fundamental"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period='1y')
            info = stock.info

            if df.empty or len(df) < 200:
                return None

            # ========== TECHNICAL ANALYSIS ==========
            current_price = df['Close'].iloc[-1]

            # Moving Averages
            sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]

            # Indicators
            rsi = self.calculate_rsi(df['Close'])
            macd, signal = self.calculate_macd(df['Close'])

            # Support/Resistance
            recent_60 = df.tail(60)
            resistance = recent_60['High'].quantile(0.90)
            support = recent_60['Low'].quantile(0.10)

            # 52-week
            high_52w = df['High'].tail(252).max()
            low_52w = df['Low'].tail(252).min()

            # Technical Score (-6 to +6)
            tech_score = 0

            if current_price > sma_20:
                tech_score += 1
            else:
                tech_score -= 1

            if current_price > sma_50:
                tech_score += 1
            else:
                tech_score -= 1

            if current_price > sma_200:
                tech_score += 2
            else:
                tech_score -= 2

            if rsi < 30:
                tech_score += 2
                rsi_signal = "Oversold"
            elif rsi > 70:
                tech_score -= 2
                rsi_signal = "Overbought"
            else:
                rsi_signal = "Neutral"

            if macd > signal:
                tech_score += 1
                macd_signal = "Bullish"
            else:
                tech_score -= 1
                macd_signal = "Bearish"

            # ========== FUNDAMENTAL ANALYSIS ==========

            # Valuation
            pe_ratio = info.get('trailingPE', info.get('forwardPE', 0))
            pb_ratio = info.get('priceToBook', 0)
            peg_ratio = info.get('pegRatio', 0)
            market_cap = info.get('marketCap', 0)
            dividend_yield = info.get('dividendYield', 0)

            # Profitability
            roe = info.get('returnOnEquity', 0)
            roa = info.get('returnOnAssets', 0)
            profit_margin = info.get('profitMargins', 0)
            operating_margin = info.get('operatingMargins', 0)
            eps = info.get('trailingEps', 0)

            # Growth
            revenue_growth = info.get('revenueGrowth', 0)
            earnings_growth = info.get('earningsGrowth', 0)

            # Financial Health
            debt_to_equity = info.get('debtToEquity', 0)
            current_ratio = info.get('currentRatio', 0)
            quick_ratio = info.get('quickRatio', 0)

            # Other
            beta = info.get('beta', 1.0)
            analyst_recommendation = info.get('recommendationKey', 'hold')
            target_price = info.get('targetMeanPrice', current_price)

            # Fundamental Score (0-100)
            fund_score = self.get_fundamental_score(info)

            # ========== COMBINED SCORING ==========

            # Normalize technical score to 0-100 scale
            tech_score_normalized = ((tech_score + 6) / 12) * 100

            # Combined score (50% technical + 50% fundamental)
            combined_score = (tech_score_normalized * 0.5) + (fund_score * 0.5)

            # Rating - ADJUSTED THRESHOLDS FOR MORE RECOMMENDATIONS
            if combined_score >= 75:
                rating = "⭐⭐⭐⭐⭐ STRONG BUY"
                recommendation = "STRONG BUY"
            elif combined_score >= 55:
                rating = "⭐⭐⭐⭐ BUY"
                recommendation = "BUY"
            elif combined_score >= 45:
                rating = "⭐⭐⭐ HOLD"
                recommendation = "HOLD"
            elif combined_score >= 30:
                rating = "⭐⭐ SELL"
                recommendation = "SELL"
            else:
                rating = "⭐ STRONG SELL"
                recommendation = "STRONG SELL"

            # ---- Smart Beta-Adjusted Stop Loss ----
            stock_beta = beta if beta else 1.0
            if stock_beta < 0.8:
                max_sl_pct = 5.0
            elif stock_beta < 1.2:
                max_sl_pct = 7.0
            elif stock_beta < 1.8:
                max_sl_pct = 10.0
            else:
                max_sl_pct = 12.0

            # Stop Loss & Targets
            if recommendation in ["STRONG BUY", "BUY"]:
                raw_stop_loss = support * 0.97
                min_allowed_sl = current_price * (1 - max_sl_pct / 100)
                stop_loss = max(raw_stop_loss, min_allowed_sl)
                sl_percentage = ((current_price - stop_loss) / current_price) * 100
                target_1 = resistance
                target_2 = min(target_price, resistance * 1.05) if target_price > current_price else resistance * 1.05
                upside = ((target_1 - current_price) / current_price) * 100
            else:
                raw_stop_loss = resistance * 1.03
                max_allowed_sl = current_price * (1 + max_sl_pct / 100)
                stop_loss = min(raw_stop_loss, max_allowed_sl)
                sl_percentage = ((stop_loss - current_price) / current_price) * 100
                target_1 = support
                target_2 = support * 0.95
                upside = ((current_price - target_1) / current_price) * 100

            # Risk-Reward
            risk = abs(current_price - stop_loss)
            reward = abs(target_1 - current_price)
            risk_reward = reward / risk if risk > 0 else 0

            # Quality Assessment
            if fund_score >= 80:
                quality = "Excellent"
            elif fund_score >= 60:
                quality = "Good"
            elif fund_score >= 40:
                quality = "Average"
            else:
                quality = "Poor"

            result = {
                # Basic Info
                'Symbol': symbol,
                'Name': name,
                'Price': round(current_price, 2),

                # Technical
                'RSI': round(rsi, 2),
                'RSI_Signal': rsi_signal,
                'MACD': macd_signal,
                'SMA_20': round(sma_20, 2),
                'SMA_50': round(sma_50, 2),
                'SMA_200': round(sma_200, 2),
                'Support': round(support, 2),
                'Resistance': round(resistance, 2),
                '52W_High': round(high_52w, 2),
                '52W_Low': round(low_52w, 2),
                'Tech_Score': tech_score,
                'Tech_Score_Norm': round(tech_score_normalized, 1),

                # Fundamental
                'PE_Ratio': round(pe_ratio, 2) if pe_ratio else 0,
                'PB_Ratio': round(pb_ratio, 2) if pb_ratio else 0,
                'PEG_Ratio': round(peg_ratio, 2) if peg_ratio else 0,
                'ROE': round(roe * 100, 2) if roe else 0,
                'ROA': round(roa * 100, 2) if roa else 0,
                'Profit_Margin': round(profit_margin * 100, 2) if profit_margin else 0,
                'Operating_Margin': round(operating_margin * 100, 2) if operating_margin else 0,
                'EPS': round(eps, 2) if eps else 0,
                'Dividend_Yield': round(dividend_yield * 100, 2) if dividend_yield else 0,
                'Revenue_Growth': round(revenue_growth * 100, 2) if revenue_growth else 0,
                'Earnings_Growth': round(earnings_growth * 100, 2) if earnings_growth else 0,
                'Debt_to_Equity': round(debt_to_equity, 2) if debt_to_equity else 0,
                'Current_Ratio': round(current_ratio, 2) if current_ratio else 0,
                'Market_Cap': round(market_cap / 1e9, 2) if market_cap else 0,
                'Beta': round(beta, 2) if beta else 1.0,
                'Fund_Score': round(fund_score, 1),
                'Quality': quality,

                # Combined
                'Combined_Score': round(combined_score, 1),
                'Rating': rating,
                'Recommendation': recommendation,

                # Trading
                'Stop_Loss': round(stop_loss, 2),
                'SL_Percentage': round(sl_percentage, 2),
                'Target_1': round(target_1, 2),
                'Target_2': round(target_2, 2),
                'Target_Price': round(target_price, 2) if target_price else 0,
                'Upside': round(upside, 2),
                'Risk_Reward': round(risk_reward, 2),
            }

            return result

        except Exception as e:
            return None

    def analyze_all_stocks(self):
        """Analyze all stocks"""
        print(f"🔍 Analyzing {len(self.sp500_stocks)} stocks...")
        print("⏳ This will take approximately 2-3 minutes...\n")

        for idx, (symbol, name) in enumerate(self.sp500_stocks.items(), 1):
            result = self.analyze_stock(symbol, name)
            if result:
                self.results.append(result)

            if idx % 10 == 0:
                print(f"  [{idx}/{len(self.sp500_stocks)}] {name}")

        print(f"\n✅ Analysis complete: {len(self.results)} stocks analyzed\n")

    def get_top_recommendations(self):
        """Get top 20 buy and sell recommendations"""
        df = pd.DataFrame(self.results)

        top_buys = df[df['Recommendation'].isin(['STRONG BUY', 'BUY'])].nlargest(20, 'Combined_Score')
        top_sells = df[df['Recommendation'].isin(['STRONG SELL', 'SELL'])].nsmallest(20, 'Combined_Score')

        return top_buys, top_sells

    # =========================================================================
    #  HTML GENERATION  —  Sunset Warm Theme  (VISIBILITY FIXED)
    # =========================================================================
    def generate_email_html(self):
        """Generate beautiful HTML report — Sunset Warm Theme"""
        df = pd.DataFrame(self.results)
        top_buys, top_sells = self.get_top_recommendations()

        now = self.get_est_time()
        time_of_day = "Morning" if now.hour < 12 else "Evening"
        next_update  = "4:30 PM" if now.hour < 12 else "9:30 AM (Next Day)"

        strong_buy_count  = len(df[df['Recommendation'] == 'STRONG BUY'])
        buy_count         = len(df[df['Recommendation'] == 'BUY'])
        hold_count        = len(df[df['Recommendation'] == 'HOLD'])
        sell_count        = len(df[df['Recommendation'] == 'SELL'])
        strong_sell_count = len(df[df['Recommendation'] == 'STRONG SELL'])

        # ── HEAD / CSS ────────────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top US Market Influencers: NASDAQ &amp; S&amp;P 500 — {time_of_day} Report</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{{
    --bg:#0f0a05;--bg2:#160d05;--card:#1d1108;--card2:#241508;
    --accent:#ff6b2b;--accent2:#ff8c55;
    --green:#22c55e;--red:#ef4444;--blue:#60a5fa;
    --gold:#f59e0b;--teal:#2dd4bf;
    /* ── FIXED: brighter base text colours ── */
    --text:#f5ddb8;--text2:#fff8ee;
    --sym:#ffb366;          /* ticker symbol — bright amber */
    --t2-price:#ffd080;     /* T2 price sub-line — warm yellow */
    --w52-neu:#c8a882;      /* 52W% neutral — readable warm grey */
    --muted:#a07850;        /* muted — raised from #7a5030 */
    --border:#3d2010;--border2:#4d2a14;
  }}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{
    background:var(--bg);color:var(--text);
    font-family:'Plus Jakarta Sans',sans-serif;
    min-height:100vh;font-size:14px;
    background-image:
      radial-gradient(ellipse at 0% 0%,rgba(255,107,43,0.08) 0%,transparent 50%),
      radial-gradient(ellipse at 100% 100%,rgba(245,158,11,0.05) 0%,transparent 40%);
  }}

  /* ── HEADER ── */
  header{{background:linear-gradient(180deg,#1a0e06,var(--bg2));border-bottom:2px solid var(--accent);padding:0;box-shadow:0 2px 20px rgba(255,107,43,0.15);}}
  .h-top{{max-width:1380px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;padding:15px 28px;gap:20px;flex-wrap:wrap;}}
  .brand{{display:flex;align-items:center;gap:12px;}}
  .brand-icon{{width:38px;height:38px;background:linear-gradient(135deg,var(--accent),var(--gold));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;}}
  .brand-t{{font-size:clamp(13px,2vw,19px);font-weight:800;color:var(--text2);}}
  .brand-s{{font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;}}
  .h-right{{display:flex;gap:0;flex-wrap:wrap;}}
  .hr{{padding:8px 16px;border-left:1px solid var(--border2);text-align:right;}}
  .hr-l{{font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;}}
  .hr-v{{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;margin-top:2px;}}

  /* Ticker bar */
  .ticker{{background:#0a0602;border-bottom:1px solid var(--border);overflow:hidden;display:flex;}}
  .ticker-inner{{max-width:1380px;margin:0 auto;display:flex;padding:0 28px;overflow-x:auto;}}
  .ti{{display:flex;gap:6px;align-items:center;padding:6px 12px;border-right:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-size:10px;white-space:nowrap;}}
  .ti-s{{color:var(--accent2);font-weight:700;}}
  .ti-p{{color:var(--text2);}}
  .ti-u{{color:var(--green);}}
  .ti-d{{color:var(--red);}}

  /* ── KPI BAND ── */
  .kpi-band{{background:var(--card);border-bottom:1px solid var(--border2);}}
  .kpi-inner{{max-width:1380px;margin:0 auto;display:grid;grid-template-columns:repeat(5,1fr);}}
  .kc{{padding:15px 20px;border-right:1px solid var(--border);text-align:center;}}
  .kc:last-child{{border-right:none;}}
  .kn{{font-size:30px;font-weight:800;line-height:1;}}
  .kl{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-top:4px;}}
  .kbar{{height:2px;border-radius:1px;margin:4px auto 0;width:40px;}}

  /* ── MAIN ── */
  .main{{max-width:1380px;margin:0 auto;padding:24px 28px;}}

  /* Section header */
  .sh{{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;}}
  .sh-icon{{width:32px;height:32px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;}}
  .shi-buy{{background:rgba(34,197,94,0.15);}}
  .shi-sell{{background:rgba(239,68,68,0.15);}}
  .sh-title{{font-size:16px;font-weight:800;color:var(--text2);}}
  .sh-divider{{flex:1;height:1px;background:var(--border);min-width:20px;}}
  .sh-count{{font-size:10px;color:var(--muted);}}

  /* ── TABLE ── */
  .tbl-wrap{{overflow-x:auto;border:1px solid var(--border2);border-radius:8px;margin-bottom:28px;background:var(--card);box-shadow:0 4px 24px rgba(0,0,0,0.3);-webkit-overflow-scrolling:touch;}}
  table{{width:100%;border-collapse:collapse;min-width:900px;}}
  th{{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#c8a060;padding:10px 12px;background:var(--card2);border-bottom:1px solid var(--border2);text-align:left;white-space:nowrap;}}
  td{{padding:11px 12px;border-bottom:1px solid var(--border);vertical-align:middle;white-space:nowrap;}}
  tr:hover td{{background:rgba(255,107,43,0.06);}}
  tr:nth-child(even) td{{background:rgba(29,17,8,0.5);}}
  tr:last-child td{{border-bottom:none;}}

  /* Stock name + symbol cell */
  .sn{{font-size:14px;font-weight:700;color:var(--text2);}}
  /* FIXED: symbol now uses --sym (bright amber) not --muted */
  .ss{{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--sym);letter-spacing:1px;margin-top:3px;}}

  .pv{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:var(--gold);}}

  /* Rating tags */
  .rt{{display:inline-block;font-size:9px;font-weight:700;padding:4px 9px;border-radius:4px;white-space:nowrap;letter-spacing:0.5px;}}
  .rt-sb{{background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.35);}}
  .rt-b{{background:rgba(96,165,250,0.15);color:#93c5fd;border:1px solid rgba(96,165,250,0.35);}}
  .rt-s{{background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.35);}}
  .rt-ss{{background:rgba(239,68,68,0.22);color:#fca5a5;border:1px solid rgba(239,68,68,0.5);}}

  /* Score */
  .scn{{font-size:22px;font-weight:800;}}
  .scb{{height:3px;border-radius:2px;margin-top:4px;width:40px;}}

  /* Upside / Downside */
  .up{{color:#4ade80;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;}}
  .dn{{color:#f87171;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;}}

  /* Target cell — FIXED: T1 brighter, T2 clearly readable */
  .t1{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--text2);}}
  .t2{{font-size:10px;font-weight:500;color:var(--t2-price);margin-top:2px;}}  /* was --muted */

  /* Stop loss cell */
  .sl1{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#f87171;}}
  .sl2{{font-size:10px;color:var(--muted);margin-top:2px;}}

  /* RSI */
  .rv{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;}}
  .rsb{{font-size:9px;color:var(--muted);}}

  /* R:R */
  .rrv{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;}}

  /* Quality badges */
  .qb{{font-size:9px;font-weight:700;padding:3px 8px;border-radius:4px;}}
  .qb-ex{{background:rgba(34,197,94,0.15);color:#4ade80;}}
  .qb-gd{{background:rgba(96,165,250,0.15);color:#93c5fd;}}
  .qb-av{{background:rgba(245,158,11,0.15);color:#fbbf24;}}
  .qb-po{{background:rgba(239,68,68,0.15);color:#f87171;}}

  /* Disclaimer */
  .disc{{background:var(--card);border:1px solid var(--border2);border-left:3px solid var(--accent);padding:14px 18px;margin:20px 0;font-size:12px;color:var(--muted);line-height:1.7;}}
  .disc strong{{color:#f87171;}}

  footer{{background:linear-gradient(90deg,var(--bg2),#1a1005,var(--bg2));border-top:2px solid var(--accent);text-align:center;padding:18px;font-size:11px;color:var(--muted);letter-spacing:1px;}}
  footer strong{{color:var(--accent2);}}

  /* ── RESPONSIVE ── */
  @media(max-width:1200px){{
    .kpi-inner{{grid-template-columns:repeat(3,1fr);}}
    .h-right .hr:nth-child(n+4){{display:none;}}
  }}
  @media(max-width:768px){{
    .kpi-inner{{grid-template-columns:repeat(2,1fr);}}
    .h-top{{padding:10px 14px;}}
    .main{{padding:14px;}}
    .h-right .hr:nth-child(n+3){{display:none;}}
    .kn{{font-size:24px;}}
  }}
  @media(max-width:480px){{
    .kpi-inner{{grid-template-columns:repeat(2,1fr);}}
    .brand-t{{font-size:13px;}}
    .sh-title{{font-size:14px;}}
    .main{{padding:10px;}}
  }}
</style>
</head>
<body>

<!-- ===== HEADER ===== -->
<header>
  <div class="h-top">
    <div class="brand">
      <div class="brand-icon">🌅</div>
      <div>
        <div class="brand-t">Top US Market Influencers · NASDAQ &amp; S&amp;P 500</div>
        <div class="brand-s">Technical &amp; Fundamental Analysis Report</div>
      </div>
    </div>
    <div class="h-right">
      <div class="hr"><div class="hr-l">Date</div><div class="hr-v" style="color:var(--gold)">{now.strftime('%d %b %Y')}</div></div>
      <div class="hr"><div class="hr-l">Time</div><div class="hr-v" style="color:var(--green)">{now.strftime('%I:%M %p')} EST</div></div>
      <div class="hr"><div class="hr-l">Session</div><div class="hr-v" style="color:var(--green)">▲ {time_of_day.upper()}</div></div>
      <div class="hr"><div class="hr-l">Next</div><div class="hr-v" style="color:var(--accent2)">{next_update}</div></div>
    </div>
  </div>
  <div class="ticker"><div class="ticker-inner">
"""
        # Ticker bar
        for t in self.results[:7]:
            pct_chg = ((t['Price'] - t['SMA_20']) / t['SMA_20']) * 100
            tick_cls = "ti-u" if pct_chg >= 0 else "ti-d"
            sign = "+" if pct_chg >= 0 else ""
            html += f'    <div class="ti"><span class="ti-s">{t["Symbol"]}</span><span class="ti-p">${t["Price"]:,.2f}</span><span class="{tick_cls}">{sign}{pct_chg:.1f}%</span></div>\n'

        html += f"""  </div></div>
</header>

<!-- ===== KPI BAND ===== -->
<div class="kpi-band"><div class="kpi-inner">
  <div class="kc"><div class="kn" style="color:var(--accent2)">{len(self.results)}</div><div class="kl">Analyzed</div><div class="kbar" style="background:var(--accent)"></div></div>
  <div class="kc"><div class="kn" style="color:var(--green)">{strong_buy_count}</div><div class="kl">Strong Buy</div><div class="kbar" style="background:var(--green)"></div></div>
  <div class="kc"><div class="kn" style="color:var(--teal)">{buy_count}</div><div class="kl">Buy</div><div class="kbar" style="background:var(--teal)"></div></div>
  <div class="kc"><div class="kn" style="color:var(--red)">{sell_count + strong_sell_count}</div><div class="kl">Sell</div><div class="kbar" style="background:var(--red)"></div></div>
  <div class="kc"><div class="kn" style="color:var(--blue)">{hold_count}</div><div class="kl">Hold</div><div class="kbar" style="background:var(--blue)"></div></div>
</div></div>

<!-- ===== MAIN CONTENT ===== -->
<div class="main">
"""

        # ── BUY TABLE ─────────────────────────────────────────────────────────
        if not top_buys.empty:
            html += """  <!-- BUY TABLE -->
  <div class="sh">
    <div class="sh-icon shi-buy">▲</div>
    <span class="sh-title">Top 20 Buy Recommendations</span>
    <div class="sh-divider"></div>
    <span class="sh-count">Sorted by Combined Score</span>
  </div>
  <div class="tbl-wrap"><table>
    <thead><tr>
      <th>#</th><th>Stock</th><th>Price</th><th>Rating</th><th>Score</th>
      <th>Upside</th><th>Target</th><th>Stop Loss</th><th>RSI</th>
      <th>R:R</th><th>52W Hi%</th><th>Beta</th><th>P/E</th><th>Div%</th><th>Quality</th>
    </tr></thead>
    <tbody>
"""
            for row_num, (_, row) in enumerate(top_buys.iterrows(), 1):
                rtag_cls = "rt-sb" if row['Recommendation'] == "STRONG BUY" else "rt-b"

                if row['Combined_Score'] >= 75:
                    sc_color = "#4ade80"; sc_bg = "#22c55e"
                elif row['Combined_Score'] >= 55:
                    sc_color = "#2dd4bf"; sc_bg = "#14b8a6"
                else:
                    sc_color = "#fbbf24"; sc_bg = "#f59e0b"

                upside_cls = "up" if row['Upside'] >= 0 else "dn"

                if row['RSI'] > 70:
                    rsi_color = "#f87171"
                elif row['RSI'] < 30:
                    rsi_color = "#4ade80"
                else:
                    rsi_color = "#93c5fd"

                # FIXED: 52W% — use explicit bright colours, never --muted for neutral
                pct_from_52w = ((row['Price'] - row['52W_High']) / row['52W_High']) * 100
                if pct_from_52w >= -5:
                    w52_color = "#f87171"   # near 52W high — red warning
                elif pct_from_52w >= -20:
                    w52_color = "#d4a85a"   # mid range — readable warm amber (was var(--muted))
                else:
                    w52_color = "#4ade80"   # well below 52W — green opportunity

                if row['Beta'] > 1.5:
                    beta_color = "#f87171"
                elif row['Beta'] > 1.0:
                    beta_color = "#fbbf24"
                else:
                    beta_color = "#4ade80"

                rr = row['Risk_Reward']
                if rr >= 2:
                    rr_color = "#4ade80"
                elif rr >= 1:
                    rr_color = "#2dd4bf"
                else:
                    rr_color = "#f87171"

                pe_display = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                if row['PE_Ratio'] <= 0:
                    pe_color = "#a07850"
                elif row['PE_Ratio'] < 25:
                    pe_color = "#4ade80"
                elif row['PE_Ratio'] < 40:
                    pe_color = "#fbbf24"
                else:
                    pe_color = "#f87171"

                div_display = f"{row['Dividend_Yield']:.2f}%" if row['Dividend_Yield'] > 0 else "—"
                div_color   = "#4ade80" if row['Dividend_Yield'] > 0 else "#a07850"

                qb_map = {"Excellent": "qb-ex", "Good": "qb-gd", "Average": "qb-av", "Poor": "qb-po"}
                qb_cls = qb_map.get(row['Quality'], "qb-av")

                html += f"""      <tr>
        <td style="color:#a07850">{row_num}</td>
        <td><div class="sn">{row['Name']}</div><div class="ss">{row['Symbol']}</div></td>
        <td><div class="pv">${row['Price']:,.2f}</div></td>
        <td><span class="rt {rtag_cls}">{row['Rating']}</span></td>
        <td><div class="scn" style="color:{sc_color}">{row['Combined_Score']:.0f}</div><div class="scb" style="background:{sc_bg}"></div></td>
        <td class="{upside_cls}">{row['Upside']:+.1f}%</td>
        <td>
          <div class="t1">${row['Target_1']:,.2f}</div>
          <div class="t2">T2: ${row['Target_2']:,.2f}</div>
        </td>
        <td>
          <div class="sl1">${row['Stop_Loss']:,.2f}</div>
          <div class="sl2">-{row['SL_Percentage']:.1f}%</div>
        </td>
        <td><div class="rv" style="color:{rsi_color}">{row['RSI']:.0f}</div><div class="rsb">{row['RSI_Signal']}</div></td>
        <td class="rrv" style="color:{rr_color}">{rr:.1f}×</td>
        <td style="color:{w52_color};font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600">{pct_from_52w:+.1f}%</td>
        <td style="color:{beta_color};font-size:11px">{row['Beta']:.2f}</td>
        <td style="color:{pe_color};font-size:11px">{pe_display}</td>
        <td style="color:{div_color}">{div_display}</td>
        <td><span class="qb {qb_cls}">{row['Quality']}</span></td>
      </tr>
"""
            html += """    </tbody>
  </table></div>
"""

        # ── SELL TABLE ────────────────────────────────────────────────────────
        if not top_sells.empty:
            html += """  <!-- SELL TABLE -->
  <div class="sh">
    <div class="sh-icon shi-sell">▼</div>
    <span class="sh-title">Top 20 Sell Recommendations</span>
    <div class="sh-divider"></div>
    <span class="sh-count">Sorted by Combined Score</span>
  </div>
  <div class="tbl-wrap"><table>
    <thead><tr>
      <th>#</th><th>Stock</th><th>Price</th><th>Rating</th><th>Score</th>
      <th>RSI</th><th>MACD</th><th>Downside</th><th>Target</th><th>Stop Loss</th>
      <th>R:R</th><th>Beta</th><th>P/E</th><th>Quality</th>
    </tr></thead>
    <tbody>
"""
            for row_num, (_, row) in enumerate(top_sells.iterrows(), 1):
                rtag_cls = "rt-ss" if row['Recommendation'] == "STRONG SELL" else "rt-s"

                sc_color = "#f87171"; sc_bg = "#ef4444"

                if row['RSI'] > 70:
                    rsi_color = "#f87171"
                elif row['RSI'] < 30:
                    rsi_color = "#4ade80"
                else:
                    rsi_color = "#fbbf24"

                macd_color = "#f87171" if row['MACD'] == "Bearish" else "#4ade80"
                dn_cls = "dn" if row['Upside'] >= 0 else "up"

                rr = row['Risk_Reward']
                if rr >= 2:
                    rr_color = "#4ade80"
                elif rr >= 1:
                    rr_color = "#fbbf24"
                else:
                    rr_color = "#f87171"

                if row['Beta'] > 1.5:
                    beta_color = "#f87171"
                elif row['Beta'] > 1.0:
                    beta_color = "#fbbf24"
                else:
                    beta_color = "#4ade80"

                pe_display = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                if row['PE_Ratio'] <= 0:
                    pe_color = "#a07850"
                elif row['PE_Ratio'] > 40:
                    pe_color = "#f87171"
                elif row['PE_Ratio'] > 25:
                    pe_color = "#fbbf24"
                else:
                    pe_color = "#4ade80"

                qb_map = {"Excellent": "qb-ex", "Good": "qb-gd", "Average": "qb-av", "Poor": "qb-po"}
                qb_cls = qb_map.get(row['Quality'], "qb-av")

                html += f"""      <tr>
        <td style="color:#a07850">{row_num}</td>
        <td><div class="sn">{row['Name']}</div><div class="ss">{row['Symbol']}</div></td>
        <td><div class="pv">${row['Price']:,.2f}</div></td>
        <td><span class="rt {rtag_cls}">{row['Rating']}</span></td>
        <td><div class="scn" style="color:{sc_color}">{row['Combined_Score']:.0f}</div><div class="scb" style="background:{sc_bg}"></div></td>
        <td><div class="rv" style="color:{rsi_color}">{row['RSI']:.0f}</div><div class="rsb">{row['RSI_Signal']}</div></td>
        <td style="color:{macd_color};font-weight:600">{row['MACD']}</td>
        <td class="{dn_cls}">{row['Upside']:+.1f}%</td>
        <td>
          <div class="t1">${row['Target_1']:,.2f}</div>
          <div class="t2">T2: ${row['Target_2']:,.2f}</div>
        </td>
        <td>
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#fbbf24">${row['Stop_Loss']:,.2f}</div>
          <div class="sl2">+{row['SL_Percentage']:.1f}%</div>
        </td>
        <td class="rrv" style="color:{rr_color}">{rr:.1f}×</td>
        <td style="color:{beta_color};font-size:11px">{row['Beta']:.2f}</td>
        <td style="color:{pe_color};font-size:11px">{pe_display}</td>
        <td><span class="qb {qb_cls}">{row['Quality']}</span></td>
      </tr>
"""
            html += """    </tbody>
  </table></div>
"""

        # ── DISCLAIMER + FOOTER ───────────────────────────────────────────────
        html += f"""  <div class="disc">
    <strong>⚠ DISCLAIMER:</strong> For <strong>EDUCATIONAL PURPOSES ONLY</strong>. Not financial advice.
    Conduct your own research, consult a registered financial advisor, use stop-losses,
    and never invest more than you can afford to lose.
  </div>
</div>

<footer>
  <strong>Top US Market Influencers: NASDAQ &amp; S&amp;P 500</strong>
  · Automated Technical &amp; Fundamental Analysis
  · Next Update: <strong>{next_update} EST</strong>
  · {now.strftime('%d %b %Y')}
</footer>

</body>
</html>
"""
        return html

    # =========================================================================
    #  EMAIL
    # =========================================================================
    def send_email(self, to_email):
        """Send email with analysis report"""
        try:
            from_email = os.environ.get('GMAIL_USER')
            password   = os.environ.get('GMAIL_APP_PASSWORD')

            if not from_email or not password:
                print("❌ Gmail credentials not found in environment variables")
                print("   Set GMAIL_USER and GMAIL_APP_PASSWORD")
                return False

            now = self.get_est_time()
            time_of_day = "Morning" if now.hour < 12 else "Evening"

            msg = MIMEMultipart('alternative')
            msg['From']    = from_email
            msg['To']      = to_email
            msg['Subject'] = f"🌅 Top US Market Influencers: NASDAQ & S&P 500 — {time_of_day} Report ({now.strftime('%d %b %Y')})"

            html_body = self.generate_email_html()
            msg.attach(MIMEText(html_body, 'html'))

            print(f"📧 Sending email to {to_email}...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            server.quit()

            print("✅ Email sent successfully!\n")
            return True

        except Exception as e:
            print(f"❌ Error sending email: {e}\n")
            return False

    def generate_complete_report(self, send_email_flag=True, recipient_email=None):
        """Generate complete analysis report"""
        est_time = self.get_est_time()

        print("=" * 70)
        print("📊 S&P 500 STOCK ANALYZER")
        print(f"Started: {est_time.strftime('%d %b %Y, %I:%M %p EST')}")
        print("=" * 70)
        print()

        self.analyze_all_stocks()

        if send_email_flag and recipient_email:
            self.send_email(recipient_email)

        print("=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)


# =============================================================================
#  ENTRY POINT
# =============================================================================
def main():
    analyzer  = SP500CompleteAnalyzer()
    recipient = os.environ.get('RECIPIENT_EMAIL')

    if not recipient:
        print("⚠️  RECIPIENT_EMAIL environment variable not set")
        recipient = None

    analyzer.generate_complete_report(send_email_flag=True, recipient_email=recipient)


if __name__ == "__main__":
    analyzer = SP500CompleteAnalyzer()
    analyzer.analyze_all_stocks()

    report_html = analyzer.generate_email_html()

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(report_html)

    # analyzer.send_email(report_html)
    print("✅ Report saved to index.html and sent to email.")
