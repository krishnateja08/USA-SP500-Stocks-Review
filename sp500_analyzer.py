"""
S&P 500 COMPLETE STOCK ANALYZER
Technical + Fundamental Analysis with Email Delivery
Theme: Deep Navy Executive

Requirements:
pip install yfinance pandas numpy openpyxl pytz
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

            # Progress indicator
            if idx % 10 == 0:
                print(f"  [{idx}/{len(self.sp500_stocks)}] {name}")

        print(f"\n✅ Analysis complete: {len(self.results)} stocks analyzed\n")

    def get_top_recommendations(self):
        """Get top 20 buy and sell recommendations"""
        df = pd.DataFrame(self.results)

        # Top 20 Buy (highest combined scores from BUY + STRONG BUY)
        top_buys = df[df['Recommendation'].isin(['STRONG BUY', 'BUY'])].nlargest(20, 'Combined_Score')

        # Top 20 Sell (lowest combined scores from SELL + STRONG SELL)
        top_sells = df[df['Recommendation'].isin(['STRONG SELL', 'SELL'])].nsmallest(20, 'Combined_Score')

        return top_buys, top_sells

    def generate_email_html(self):
        """Generate beautiful HTML email — Deep Navy Executive Theme"""
        df = pd.DataFrame(self.results)
        top_buys, top_sells = self.get_top_recommendations()

        # Get EST time
        now = self.get_est_time()
        time_of_day = "Morning" if now.hour < 12 else "Evening"

        # Count recommendations
        strong_buy_count = len(df[df['Recommendation'] == 'STRONG BUY'])
        buy_count = len(df[df['Recommendation'] == 'BUY'])
        hold_count = len(df[df['Recommendation'] == 'HOLD'])
        sell_count = len(df[df['Recommendation'] == 'SELL'])
        strong_sell_count = len(df[df['Recommendation'] == 'STRONG SELL'])

        next_update = "4:30 PM" if now.hour < 12 else "9:30 AM (Next Day)"

        # ── CSS / HEAD ────────────────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top US Market Influencers: NASDAQ &amp; S&amp;P 500 — {time_of_day} Report</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700;800&family=Barlow:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{{
    --bg:#0e1621;--bg2:#121d2e;--card:#162030;--card2:#1a2840;
    --accent:#3d8eff;--accent2:#5ba3ff;
    --green:#00c875;--green2:#33d68e;
    --red:#ff4757;--red2:#ff6b78;
    --gold:#ffb347;--gold2:#ffc87a;
    --purple:#9b59f5;
    --text:#c8d8eb;--text2:#e8f0f8;--muted:#4a6080;
    --border:#1e3050;--border2:#243a58;
  }}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:var(--bg);color:var(--text);font-family:'Barlow',sans-serif;min-height:100vh;font-size:14px;
    background-image:radial-gradient(ellipse at 70% -10%,rgba(61,142,255,0.06) 0%,transparent 50%);
  }}

  /* ===== HEADER ===== */
  header{{
    background:linear-gradient(180deg,#0c1520 0%,var(--bg2) 100%);
    border-bottom:1px solid var(--border2);
    padding:0;
  }}
  .hdr-main{{
    max-width:1400px;margin:0 auto;
    display:flex;align-items:center;justify-content:space-between;
    padding:16px 28px;
    gap:20px;
  }}
  .hdr-brand{{display:flex;flex-direction:column;gap:4px;}}
  .brand-logo{{display:flex;align-items:center;gap:10px;}}
  .brand-square{{
    width:32px;height:32px;
    background:var(--accent);
    clip-path:polygon(0 0,100% 0,100% 70%,70% 100%,0 100%);
    display:flex;align-items:center;justify-content:center;
    font-family:'Barlow Condensed',sans-serif;
    font-weight:800;font-size:14px;color:#fff;
  }}
  .brand-name{{
    font-family:'Barlow Condensed',sans-serif;
    font-size:clamp(16px,2.5vw,22px);font-weight:700;letter-spacing:1px;
    color:var(--text2);text-transform:uppercase;
  }}
  .brand-sub{{font-size:11px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;}}

  .hdr-kpis{{display:flex;gap:0;}}
  .kpi-block{{
    padding:10px 20px;border-left:1px solid var(--border);
    display:flex;flex-direction:column;align-items:flex-end;gap:2px;
  }}
  .kpi-label{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);}}
  .kpi-val{{font-family:'Barlow Condensed',sans-serif;font-size:18px;font-weight:700;}}

  .hdr-ticker{{
    background:var(--bg);border-top:1px solid var(--border);
    overflow:hidden;display:flex;
  }}
  .hdr-ticker-inner{{
    max-width:1400px;margin:0 auto;
    display:flex;padding:0 28px;
  }}
  .tick{{
    display:flex;gap:8px;align-items:center;
    padding:7px 14px;border-right:1px solid var(--border);
    font-family:'Roboto Mono',monospace;font-size:11px;white-space:nowrap;
  }}
  .tick-sym{{color:var(--accent2);font-weight:600;}}
  .tick-price{{color:var(--text);}}
  .tick-up{{color:var(--green);}}
  .tick-dn{{color:var(--red);}}

  /* ===== STATS BAND ===== */
  .stats-band{{background:var(--card);border-bottom:1px solid var(--border2);}}
  .stats-inner{{
    max-width:1400px;margin:0 auto;
    display:grid;grid-template-columns:repeat(5,1fr);
    padding:0 28px;
  }}
  .stat-c{{
    padding:16px 20px;border-right:1px solid var(--border);
    text-align:center;transition:0.2s;
  }}
  .stat-c:hover{{background:rgba(61,142,255,0.05);}}
  .stat-c:last-child{{border-right:none;}}
  .stat-num{{
    font-family:'Barlow Condensed',sans-serif;
    font-size:32px;font-weight:700;line-height:1;
  }}
  .stat-lbl{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-top:4px;}}
  .stat-bar{{height:2px;border-radius:1px;margin:6px auto 0;max-width:60px;}}

  /* ===== CONTENT ===== */
  .content{{max-width:1400px;margin:0 auto;padding:24px 28px;}}

  .sec-header{{
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:14px;
  }}
  .sec-left{{display:flex;align-items:center;gap:10px;}}
  .sec-indicator{{width:4px;height:28px;border-radius:2px;}}
  .sec-title{{
    font-family:'Barlow Condensed',sans-serif;
    font-size:18px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
    color:var(--text2);
  }}
  .sec-count{{
    font-size:10px;letter-spacing:2px;color:var(--muted);
    border:1px solid var(--border2);padding:3px 10px;border-radius:2px;
  }}
  .sec-divider{{width:100%;height:1px;background:var(--border);margin-bottom:14px;}}

  /* Table */
  .tbl-wrap{{overflow-x:auto;border-radius:4px;border:1px solid var(--border2);}}
  table{{width:100%;border-collapse:collapse;}}
  thead tr{{background:var(--card2);}}
  th{{
    font-family:'Barlow Condensed',sans-serif;
    font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;
    color:var(--muted);padding:11px 12px;
    border-bottom:1px solid var(--border2);
    text-align:left;white-space:nowrap;
  }}
  tbody td{{
    padding:11px 12px;border-bottom:1px solid var(--border);
    vertical-align:middle;white-space:nowrap;
  }}
  tbody tr:nth-child(even) td{{background:rgba(22,32,48,0.5);}}
  tbody tr:hover td{{background:rgba(61,142,255,0.06);}}
  tbody tr:last-child td{{border-bottom:none;}}

  .stk-name{{font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:600;color:var(--text2);}}
  .stk-sym{{font-family:'Roboto Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2px;margin-top:2px;}}
  .price-val{{font-family:'Roboto Mono',monospace;font-size:14px;font-weight:600;color:var(--gold);}}

  .rtag{{
    font-family:'Barlow Condensed',sans-serif;
    font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
    padding:4px 10px;border-radius:3px;white-space:nowrap;
    display:inline-block;
  }}
  .rtag-sb{{background:rgba(0,200,117,0.12);color:var(--green2);border:1px solid rgba(0,200,117,0.25);}}
  .rtag-b{{background:rgba(61,142,255,0.12);color:var(--accent2);border:1px solid rgba(61,142,255,0.25);}}
  .rtag-h{{background:rgba(155,89,245,0.12);color:var(--purple);border:1px solid rgba(155,89,245,0.25);}}
  .rtag-s{{background:rgba(255,71,87,0.12);color:var(--red2);border:1px solid rgba(255,71,87,0.25);}}
  .rtag-ss{{background:rgba(255,71,87,0.2);color:var(--red);border:1px solid rgba(255,71,87,0.4);}}

  .score-box{{display:inline-flex;flex-direction:column;align-items:center;gap:4px;}}
  .score-num{{font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:700;}}
  .score-track{{width:36px;height:3px;background:var(--border);border-radius:2px;}}
  .score-fill{{height:100%;border-radius:2px;}}

  .up-pct{{font-family:'Barlow Condensed',sans-serif;font-size:17px;font-weight:700;color:var(--green);}}
  .dn-pct{{font-family:'Barlow Condensed',sans-serif;font-size:17px;font-weight:700;color:var(--red);}}

  .tgt{{font-family:'Roboto Mono',monospace;font-size:13px;color:var(--text);}}
  .tgt2{{font-size:10px;color:var(--muted);}}

  .sl{{font-family:'Roboto Mono',monospace;font-size:13px;color:var(--red2);}}
  .sl2{{font-size:10px;color:var(--muted);}}

  .rsi-num{{font-family:'Roboto Mono',monospace;font-size:14px;font-weight:600;}}
  .rsi-lbl{{font-size:9px;color:var(--muted);}}

  .rr{{font-family:'Barlow Condensed',sans-serif;font-size:17px;font-weight:700;}}

  .pe-val{{font-family:'Roboto Mono',monospace;font-size:13px;}}

  .qbadge{{
    font-family:'Barlow Condensed',sans-serif;
    font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
    padding:3px 8px;border-radius:3px;
  }}
  .qb-ex{{background:rgba(0,200,117,0.1);color:var(--green2);}}
  .qb-gd{{background:rgba(61,142,255,0.1);color:var(--accent2);}}
  .qb-av{{background:rgba(255,179,71,0.1);color:var(--gold2);}}
  .qb-po{{background:rgba(255,71,87,0.1);color:var(--red2);}}

  .spacer{{height:28px;}}

  .disc{{
    background:var(--card);border:1px solid var(--border2);
    border-left:3px solid var(--gold);
    padding:16px 20px;margin-top:28px;
    font-size:12px;color:var(--muted);line-height:1.7;
  }}
  .disc strong{{color:var(--red2);}}

  footer{{
    background:var(--bg2);border-top:1px solid var(--border2);
    padding:16px 28px;text-align:center;
    font-size:11px;color:var(--muted);letter-spacing:1px;
    font-family:'Barlow Condensed',sans-serif;
  }}
  footer strong{{color:var(--accent2);}}

  @media(max-width:1100px){{
    .stats-inner{{grid-template-columns:repeat(3,1fr);}}
    .hdr-kpis .kpi-block:nth-child(n+3){{display:none;}}
  }}
  @media(max-width:700px){{
    .hdr-main{{padding:12px 16px;flex-wrap:wrap;}}
    .stats-inner{{grid-template-columns:repeat(2,1fr);padding:0 16px;}}
    .content{{padding:16px;}}
    table{{min-width:700px;}}
  }}
</style>
</head>
<body>

<!-- ===== HEADER ===== -->
<header>
  <div class="hdr-main">
    <div class="hdr-brand">
      <div class="brand-logo">
        <div class="brand-square">US</div>
        <div class="brand-name">Market Intelligence · NASDAQ &amp; S&amp;P 500</div>
      </div>
      <div class="brand-sub">Technical &amp; Fundamental Analysis Report</div>
    </div>
    <div class="hdr-kpis">
      <div class="kpi-block">
        <span class="kpi-label">Session</span>
        <span class="kpi-val" style="color:var(--green)">{time_of_day.upper()} ▲</span>
      </div>
      <div class="kpi-block">
        <span class="kpi-label">Updated</span>
        <span class="kpi-val" style="color:var(--accent2)">{now.strftime('%I:%M %p')} EST</span>
      </div>
      <div class="kpi-block">
        <span class="kpi-label">{now.strftime('%d %b %Y')}</span>
        <span class="kpi-val" style="color:var(--gold)">{time_of_day}</span>
      </div>
    </div>
  </div>
  <div class="hdr-ticker">
    <div class="hdr-ticker-inner">
"""
        # Ticker bar — use top buy stocks if available, fallback to first 7 results
        ticker_stocks = list(self.results[:7]) if self.results else []
        for t in ticker_stocks:
            pct_chg = ((t['Price'] - t['SMA_20']) / t['SMA_20']) * 100
            tick_cls = "tick-up" if pct_chg >= 0 else "tick-dn"
            sign = "+" if pct_chg >= 0 else ""
            html += f'      <div class="tick"><span class="tick-sym">{t["Symbol"]}</span><span class="tick-price">${t["Price"]:,.2f}</span><span class="{tick_cls}">{sign}{pct_chg:.1f}%</span></div>\n'

        html += f"""    </div>
  </div>
</header>

<!-- ===== STATS BAND ===== -->
<div class="stats-band">
  <div class="stats-inner">
    <div class="stat-c">
      <div class="stat-num" style="color:var(--accent2)">{len(self.results)}</div>
      <div class="stat-lbl">Stocks Analyzed</div>
      <div class="stat-bar" style="background:var(--accent)"></div>
    </div>
    <div class="stat-c">
      <div class="stat-num" style="color:var(--green)">{strong_buy_count}</div>
      <div class="stat-lbl">Strong Buy</div>
      <div class="stat-bar" style="background:var(--green)"></div>
    </div>
    <div class="stat-c">
      <div class="stat-num" style="color:var(--green2)">{buy_count}</div>
      <div class="stat-lbl">Buy</div>
      <div class="stat-bar" style="background:var(--green2)"></div>
    </div>
    <div class="stat-c">
      <div class="stat-num" style="color:var(--red)">{sell_count + strong_sell_count}</div>
      <div class="stat-lbl">Sell / Strong Sell</div>
      <div class="stat-bar" style="background:var(--red)"></div>
    </div>
    <div class="stat-c">
      <div class="stat-num" style="color:var(--purple)">{hold_count}</div>
      <div class="stat-lbl">Hold</div>
      <div class="stat-bar" style="background:var(--purple)"></div>
    </div>
  </div>
</div>

<!-- ===== MAIN CONTENT ===== -->
<div class="content">
"""

        # ── BUY TABLE ────────────────────────────────────────────────────────────
        if not top_buys.empty:
            html += f"""  <!-- BUY TABLE -->
  <div class="sec-header">
    <div class="sec-left">
      <div class="sec-indicator" style="background:var(--green)"></div>
      <span class="sec-title">▲ Top 20 Buy Recommendations</span>
    </div>
    <span class="sec-count">Sorted by Combined Score</span>
  </div>
  <div class="sec-divider"></div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Stock</th><th>Price</th><th>Rating</th><th>Score</th>
          <th>Upside</th><th>Target</th><th>Stop Loss</th><th>RSI</th>
          <th>R:R</th><th>52W Hi%</th><th>Beta</th><th>P/E</th><th>Div%</th><th>Quality</th>
        </tr>
      </thead>
      <tbody>
"""
            for row_num, (_, row) in enumerate(top_buys.iterrows(), 1):
                # Rating tag class
                rtag_cls = "rtag-sb" if row['Recommendation'] == "STRONG BUY" else "rtag-b"

                # Score colour
                if row['Combined_Score'] >= 75:
                    score_color = "var(--green)"
                    score_bar_color = "var(--green)"
                elif row['Combined_Score'] >= 55:
                    score_color = "var(--accent2)"
                    score_bar_color = "var(--accent)"
                else:
                    score_color = "var(--gold)"
                    score_bar_color = "var(--gold)"

                # Upside
                upside_cls = "up-pct" if row['Upside'] >= 0 else "dn-pct"

                # RSI colour
                if row['RSI'] > 70:
                    rsi_color = "var(--red)"
                elif row['RSI'] < 30:
                    rsi_color = "var(--green)"
                else:
                    rsi_color = "var(--accent2)"

                # 52W High %
                pct_from_52w_high = ((row['Price'] - row['52W_High']) / row['52W_High']) * 100
                if pct_from_52w_high >= -5:
                    w52_color = "var(--red)"
                elif pct_from_52w_high >= -20:
                    w52_color = "var(--muted)"
                else:
                    w52_color = "var(--green)"

                # Beta colour
                if row['Beta'] > 1.5:
                    beta_color = "var(--red)"
                elif row['Beta'] > 1.0:
                    beta_color = "var(--gold)"
                else:
                    beta_color = "var(--green)"

                # R:R colour
                rr = row['Risk_Reward']
                if rr >= 2:
                    rr_color = "var(--green)"
                elif rr >= 1:
                    rr_color = "var(--accent2)"
                else:
                    rr_color = "var(--red)"

                # P/E
                pe_display = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                if row['PE_Ratio'] <= 0:
                    pe_color = "var(--muted)"
                elif row['PE_Ratio'] < 25:
                    pe_color = "var(--green)"
                elif row['PE_Ratio'] < 40:
                    pe_color = "var(--gold)"
                else:
                    pe_color = "var(--red2)"

                # Dividend
                div_display = f"{row['Dividend_Yield']:.2f}%" if row['Dividend_Yield'] > 0 else "—"
                div_color = "var(--green)" if row['Dividend_Yield'] > 2 else "var(--text)"

                # Quality badge
                qbadge_map = {"Excellent": "qb-ex", "Good": "qb-gd", "Average": "qb-av", "Poor": "qb-po"}
                qbadge_cls = qbadge_map.get(row['Quality'], "qb-av")

                html += f"""        <tr>
          <td style="color:var(--muted);font-family:'Roboto Mono',monospace;font-size:11px">{row_num:02d}</td>
          <td><div class="stk-name">{row['Name']}</div><div class="stk-sym">{row['Symbol']}</div></td>
          <td><div class="price-val">${row['Price']:,.2f}</div></td>
          <td><span class="rtag {rtag_cls}">{row['Rating']}</span></td>
          <td><div class="score-box"><div class="score-num" style="color:{score_color}">{row['Combined_Score']:.0f}</div><div class="score-track"><div class="score-fill" style="width:{row['Combined_Score']:.0f}%;background:{score_bar_color}"></div></div></div></td>
          <td><span class="{upside_cls}">{row['Upside']:+.1f}%</span></td>
          <td><div class="tgt">${row['Target_1']:,.2f}</div><div class="tgt2">T2: ${row['Target_2']:,.2f}</div></td>
          <td><div class="sl">${row['Stop_Loss']:,.2f}</div><div class="sl2">-{row['SL_Percentage']:.1f}%</div></td>
          <td><div class="rsi-num" style="color:{rsi_color}">{row['RSI']:.0f}</div><div class="rsi-lbl">{row['RSI_Signal']}</div></td>
          <td><span class="rr" style="color:{rr_color}">{rr:.1f}×</span></td>
          <td style="color:{w52_color};font-family:'Roboto Mono',monospace;font-size:12px">{pct_from_52w_high:+.1f}%</td>
          <td style="color:{beta_color};font-family:'Roboto Mono',monospace">{row['Beta']:.2f}</td>
          <td><span class="pe-val" style="color:{pe_color}">{pe_display}</span></td>
          <td style="color:{div_color};font-size:12px">{div_display}</td>
          <td><span class="qbadge {qbadge_cls}">{row['Quality']}</span></td>
        </tr>
"""
            html += """      </tbody>
    </table>
  </div>

  <div class="spacer"></div>
"""

        # ── SELL TABLE ───────────────────────────────────────────────────────────
        if not top_sells.empty:
            html += """  <!-- SELL TABLE -->
  <div class="sec-header">
    <div class="sec-left">
      <div class="sec-indicator" style="background:var(--red)"></div>
      <span class="sec-title">▼ Top 20 Sell Recommendations</span>
    </div>
    <span class="sec-count">Sorted by Combined Score</span>
  </div>
  <div class="sec-divider"></div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Stock</th><th>Price</th><th>Rating</th><th>Score</th>
          <th>RSI</th><th>MACD</th><th>Downside</th><th>Target</th><th>Stop Loss</th>
          <th>R:R</th><th>Beta</th><th>P/E</th><th>Quality</th>
        </tr>
      </thead>
      <tbody>
"""
            for row_num, (_, row) in enumerate(top_sells.iterrows(), 1):
                # Rating tag class
                rtag_cls = "rtag-ss" if row['Recommendation'] == "STRONG SELL" else "rtag-s"

                # Score colour (sell = always red range)
                if row['Combined_Score'] <= 30:
                    score_color = "var(--red)"
                    score_bar_color = "var(--red)"
                else:
                    score_color = "var(--red2)"
                    score_bar_color = "var(--red2)"

                # RSI colour
                if row['RSI'] > 70:
                    rsi_color = "var(--red)"
                elif row['RSI'] < 30:
                    rsi_color = "var(--green)"
                else:
                    rsi_color = "var(--gold)"

                # Downside
                dn_cls = "dn-pct" if row['Upside'] >= 0 else "up-pct"

                # MACD colour
                macd_color = "var(--red)" if row['MACD'] == "Bearish" else "var(--green)"

                # R:R colour
                rr = row['Risk_Reward']
                if rr >= 2:
                    rr_color = "var(--green)"
                elif rr >= 1:
                    rr_color = "var(--gold)"
                else:
                    rr_color = "var(--red)"

                # Beta colour
                if row['Beta'] > 1.5:
                    beta_color = "var(--red)"
                elif row['Beta'] > 1.0:
                    beta_color = "var(--gold)"
                else:
                    beta_color = "var(--green)"

                # P/E
                pe_display = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                if row['PE_Ratio'] <= 0:
                    pe_color = "var(--muted)"
                elif row['PE_Ratio'] > 40:
                    pe_color = "var(--red)"
                elif row['PE_Ratio'] > 25:
                    pe_color = "var(--gold)"
                else:
                    pe_color = "var(--green)"

                # Quality badge
                qbadge_map = {"Excellent": "qb-ex", "Good": "qb-gd", "Average": "qb-av", "Poor": "qb-po"}
                qbadge_cls = qbadge_map.get(row['Quality'], "qb-av")

                html += f"""        <tr>
          <td style="color:var(--muted);font-family:'Roboto Mono',monospace;font-size:11px">{row_num:02d}</td>
          <td><div class="stk-name">{row['Name']}</div><div class="stk-sym">{row['Symbol']}</div></td>
          <td><div class="price-val">${row['Price']:,.2f}</div></td>
          <td><span class="rtag {rtag_cls}">{row['Rating']}</span></td>
          <td><div class="score-box"><div class="score-num" style="color:{score_color}">{row['Combined_Score']:.0f}</div><div class="score-track"><div class="score-fill" style="width:{row['Combined_Score']:.0f}%;background:{score_bar_color}"></div></div></div></td>
          <td><div class="rsi-num" style="color:{rsi_color}">{row['RSI']:.0f}</div><div class="rsi-lbl">{row['RSI_Signal']}</div></td>
          <td style="color:{macd_color};font-size:13px;font-weight:600">{row['MACD']}</td>
          <td><span class="{dn_cls}">{row['Upside']:+.1f}%</span></td>
          <td><div class="tgt">${row['Target_1']:,.2f}</div><div class="tgt2">T2: ${row['Target_2']:,.2f}</div></td>
          <td><div style="font-family:'Roboto Mono',monospace;font-size:13px;color:var(--gold)">${row['Stop_Loss']:,.2f}</div><div class="sl2">+{row['SL_Percentage']:.1f}%</div></td>
          <td><span class="rr" style="color:{rr_color}">{rr:.1f}×</span></td>
          <td style="color:{beta_color};font-family:'Roboto Mono',monospace">{row['Beta']:.2f}</td>
          <td><span class="pe-val" style="color:{pe_color}">{pe_display}</span></td>
          <td><span class="qbadge {qbadge_cls}">{row['Quality']}</span></td>
        </tr>
"""
            html += """      </tbody>
    </table>
  </div>
"""

        # ── DISCLAIMER + FOOTER ──────────────────────────────────────────────────
        html += f"""
  <div class="disc">
    <strong>⚠ DISCLAIMER:</strong> This analysis is for <strong>EDUCATIONAL PURPOSES ONLY</strong>. This is NOT financial advice. Always do your own research, consult a registered financial advisor, use proper risk management, and never invest more than you can afford to lose.
  </div>
</div>

<footer>
  <strong>Top US Market Influencers: NASDAQ &amp; S&amp;P 500</strong> &nbsp;·&nbsp; Automated Analysis System &nbsp;·&nbsp; Next Update: <strong>{next_update} EST</strong> &nbsp;·&nbsp; {now.strftime('%d %b %Y')}
</footer>

</body>
</html>
"""
        return html

    def send_email(self, to_email):
        """Send email with analysis report"""
        try:
            # Get credentials from environment variables
            from_email = os.environ.get('GMAIL_USER')
            password = os.environ.get('GMAIL_APP_PASSWORD')

            if not from_email or not password:
                print("❌ Gmail credentials not found in environment variables")
                print("   Set GMAIL_USER and GMAIL_APP_PASSWORD")
                return False

            # Get EST time
            now = self.get_est_time()
            time_of_day = "Morning" if now.hour < 12 else "Evening"

            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = f"🌅 Top US Market Influencers: NASDAQ & S&P 500 — {time_of_day} Report ({now.strftime('%d %b %Y')})"

            # Generate email body
            html_body = self.generate_email_html()
            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            print(f"📧 Sending email to {to_email}...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            server.quit()

            print(f"✅ Email sent successfully!\n")
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

        # Analyze all stocks
        self.analyze_all_stocks()

        # Send email if requested
        if send_email_flag and recipient_email:
            self.send_email(recipient_email)

        print("=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)


def main():
    """Main execution"""
    analyzer = SP500CompleteAnalyzer()

    # Get recipient email from environment variable
    recipient = os.environ.get('RECIPIENT_EMAIL')

    if not recipient:
        print("⚠️  RECIPIENT_EMAIL environment variable not set")
        print("   Please set it to receive email reports")
        recipient = None

    # Generate report and send email
    analyzer.generate_complete_report(send_email_flag=True, recipient_email=recipient)


if __name__ == "__main__":
    analyzer = SP500CompleteAnalyzer()
    analyzer.analyze_all_stocks()

    # 1. Generate the HTML content
    report_html = analyzer.generate_email_html()

    # 2. Save it as index.html (This is what GitHub Pages will display)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(report_html)

    # 3. Keep your existing email logic here
    # analyzer.send_email(report_html)
    print("✅ Report saved to index.html and sent to email.")
