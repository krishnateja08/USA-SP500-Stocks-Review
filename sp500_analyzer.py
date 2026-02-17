"""
S&P 500 COMPLETE STOCK ANALYZER
Technical + Fundamental Analysis with Email Delivery

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
            'MCD': 'McDonald\'s',
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
            
            # Stop Loss & Targets
            if recommendation in ["STRONG BUY", "BUY"]:
                stop_loss = support * 0.97
                sl_percentage = ((current_price - stop_loss) / current_price) * 100
                target_1 = resistance
                target_2 = min(target_price, resistance * 1.05) if target_price > current_price else resistance * 1.05
                upside = ((target_1 - current_price) / current_price) * 100
            else:
                stop_loss = resistance * 1.03
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
        """Generate beautiful HTML email"""
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
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body bgcolor="#1a0a00" style="margin:0; padding:0; font-family: Arial, sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#1a0a00">
        <tr>
            <td align="center" style="padding: 20px;">
                <table width="1200" cellpadding="0" cellspacing="0" border="0" style="max-width:1200px; width:100%;">
                    <!-- Header -->
                    <tr>
                        <td align="center" style="background: linear-gradient(135deg, #7b2d00 0%, #c0392b 55%, #e67e22 100%); border-radius: 12px 12px 0 0; padding: 0;">
                            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                                <tr>
                                    <td align="center" style="padding: 12px 30px 4px;">
                                        <p style="color: #ffd5b0; margin: 0; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; opacity: 0.9;">📅 {time_of_day} Update &nbsp;|&nbsp; {now.strftime('%d %b %Y, %I:%M %p')} EST</p>
                                    </td>
                                </tr>
                                <tr>
                                    <td align="center" style="padding: 8px 30px 6px;">
                                        <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: 0.5px; line-height: 1.3;">🌅 Top US Market Influencers: NASDAQ &amp; S&amp;P 500</h1>
                                    </td>
                                </tr>
                                <tr>
                                    <td align="center" style="padding: 4px 30px 18px;">
                                        <p style="color: #ffd5b0; margin: 0; font-size: 13px; opacity: 0.85;">Technical &amp; Fundamental Analysis Report</p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td bgcolor="#1f0d00" style="padding: 28px 30px;">
                            
                            <!-- Summary Box -->
                            <table width="100%" cellpadding="15" cellspacing="0" border="0" style="border-radius: 10px; margin-bottom: 30px; background: linear-gradient(135deg, #7b2d00, #c0392b); border: 1px solid #e67e2266;">
                                <tr>
                                    <td>
                                        <h2 style="color: #ffffff; margin: 0 0 15px 0; font-size: 18px; letter-spacing: 0.5px;">📈 Market Summary</h2>
                                        <table width="100%" cellpadding="10" cellspacing="10" border="0">
                                            <tr>
                                                <td width="25%" bgcolor="#e67e22" align="center" style="border-radius: 8px;">
                                                    <strong style="color: #ffffff; font-size: 30px; display: block;">{len(self.results)}</strong>
                                                    <span style="color: #fff3e0; font-size: 12px; font-weight: 600;">STOCKS ANALYZED</span>
                                                </td>
                                                <td width="25%" bgcolor="#1e8449" align="center" style="border-radius: 8px;">
                                                    <strong style="color: #ffffff; font-size: 30px; display: block;">{strong_buy_count}</strong>
                                                    <span style="color: #d5f5e3; font-size: 12px; font-weight: 600;">STRONG BUY</span>
                                                </td>
                                                <td width="25%" bgcolor="#2471a3" align="center" style="border-radius: 8px;">
                                                    <strong style="color: #ffffff; font-size: 30px; display: block;">{buy_count}</strong>
                                                    <span style="color: #d6eaf8; font-size: 12px; font-weight: 600;">BUY</span>
                                                </td>
                                                <td width="25%" bgcolor="#6c3483" align="center" style="border-radius: 8px;">
                                                    <strong style="color: #ffffff; font-size: 30px; display: block;">{hold_count}</strong>
                                                    <span style="color: #e8daef; font-size: 12px; font-weight: 600;">HOLD</span>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
"""
        
        # Top 20 Buy Recommendations
        if not top_buys.empty:
            html += """
                            <!-- BUY Section -->
                            <h2 style="color: #e67e22; border-bottom: 3px solid #e67e22; padding-bottom: 10px; margin-top: 40px; font-size: 18px; letter-spacing: 0.5px;">🟢 TOP 20 BUY RECOMMENDATIONS</h2>
                            <table width="100%" cellpadding="12" cellspacing="0" border="1" bordercolor="#3d1500" style="border-collapse: collapse; margin: 20px 0;">
                                <tr style="background: linear-gradient(90deg, #1e8449, #145a32);">
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">STOCK</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">PRICE</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">RATING</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">SCORE</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">UPSIDE %</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">TARGET</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">STOP LOSS</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">RSI</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">R:R</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">52W HIGH %</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">BETA</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">P/E</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">DIV %</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">QUALITY</th>
                                </tr>
"""
            row_num = 0
            for idx, row in top_buys.iterrows():
                row_num += 1
                row_bg = "#1f0d00" if row_num % 2 == 1 else "#2a1000"

                # Upside color
                if row['Upside'] > 0:
                    upside_color = "#27ae60"
                elif row['Upside'] < 0:
                    upside_color = "#e74c3c"
                else:
                    upside_color = "#e8c090"

                # RSI color
                if row['RSI'] > 70:
                    rsi_color = "#e74c3c"
                elif row['RSI'] < 30:
                    rsi_color = "#27ae60"
                else:
                    rsi_color = "#f39c12"

                # 52W High % — how far below the 52-week high
                pct_from_52w_high = ((row['Price'] - row['52W_High']) / row['52W_High']) * 100
                if pct_from_52w_high >= -5:
                    w52_color = "#e74c3c"   # near high — caution
                elif pct_from_52w_high >= -20:
                    w52_color = "#f39c12"   # moderate pullback
                else:
                    w52_color = "#27ae60"   # deep pullback — opportunity

                # Beta color
                if row['Beta'] > 1.5:
                    beta_color = "#e74c3c"
                elif row['Beta'] > 1.0:
                    beta_color = "#f39c12"
                else:
                    beta_color = "#27ae60"

                # R:R color
                rr = row['Risk_Reward']
                if rr >= 2:
                    rr_color = "#27ae60"
                elif rr >= 1:
                    rr_color = "#f39c12"
                else:
                    rr_color = "#e74c3c"

                # P/E display
                pe_display = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                pe_color = "#27ae60" if 0 < row['PE_Ratio'] < 25 else ("#f39c12" if row['PE_Ratio'] < 40 else "#e74c3c")

                # Dividend display
                div_display = f"{row['Dividend_Yield']:.2f}%" if row['Dividend_Yield'] > 0 else "—"
                div_color = "#27ae60" if row['Dividend_Yield'] > 2 else "#e8c090"

                # Quality badge
                if row['Quality'] == 'Excellent':
                    badge_color = "#1e8449"
                elif row['Quality'] == 'Good':
                    badge_color = "#2471a3"
                elif row['Quality'] == 'Average':
                    badge_color = "#d68910"
                else:
                    badge_color = "#922b21"

                html += f"""
                                <tr bgcolor="{row_bg}">
                                    <td style="color: #f0d0a0; font-weight: 700; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">{row['Name']}<br><span style="color:#8a6040; font-size:11px; font-weight:400;">{row['Symbol']}</span></td>
                                    <td style="color: #e8c090; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px; font-weight: 600;">${row['Price']:,.2f}</td>
                                    <td style="color: #e8c090; padding: 12px 10px; border: 1px solid #3d1500; font-size: 11px; font-weight: bold;">{row['Rating']}</td>
                                    <td style="color: #e67e22; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 18px;">{row['Combined_Score']:.0f}</td>
                                    <td style="color: {upside_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 15px;">{row['Upside']:+.1f}%</td>
                                    <td style="color: #e8c090; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">${row['Target_1']:,.2f}</td>
                                    <td style="color: #e74c3c; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">${row['Stop_Loss']:,.2f}<br><span style="font-size:11px; color:#8a6040;">-{row['SL_Percentage']:.1f}%</span></td>
                                    <td style="color: {rsi_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 15px;">{row['RSI']:.0f}<br><span style="font-size:10px; color:#8a6040;">{row['RSI_Signal']}</span></td>
                                    <td style="color: {rr_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 15px;">{rr:.1f}x</td>
                                    <td style="color: {w52_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">{pct_from_52w_high:+.1f}%<br><span style="font-size:10px; color:#8a6040;">${row['52W_High']:,.0f} hi</span></td>
                                    <td style="color: {beta_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">{row['Beta']:.2f}</td>
                                    <td style="color: {pe_color}; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px; font-weight: 600;">{pe_display}</td>
                                    <td style="color: {div_color}; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px; font-weight: 600;">{div_display}</td>
                                    <td style="padding: 12px 10px; border: 1px solid #3d1500;"><span style="background-color: {badge_color}; color: #ffffff; padding: 5px 8px; border-radius: 5px; font-size: 11px; font-weight: bold; white-space: nowrap;">{row['Quality']}</span></td>
                                </tr>
"""
            html += """
                            </table>
"""
        
        # Top 20 Sell Recommendations
        if not top_sells.empty:
            html += """
                            <!-- SELL Section -->
                            <h2 style="color: #e74c3c; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; margin-top: 40px; font-size: 18px; letter-spacing: 0.5px;">🔴 TOP 20 SELL RECOMMENDATIONS</h2>
                            <table width="100%" cellpadding="12" cellspacing="0" border="1" bordercolor="#3d1500" style="border-collapse: collapse; margin: 20px 0;">
                                <tr style="background: linear-gradient(90deg, #922b21, #c0392b);">
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">STOCK</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">PRICE</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">RATING</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">SCORE</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">RSI</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">MACD</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">DOWNSIDE %</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">TARGET</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">STOP LOSS</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">R:R</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">BETA</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">P/E</th>
                                    <th style="color: #ffffff; text-align: left; padding: 14px 10px; font-size: 12px; letter-spacing: 0.5px;">QUALITY</th>
                                </tr>
"""
            row_num = 0
            for idx, row in top_sells.iterrows():
                row_num += 1
                row_bg = "#1f0d00" if row_num % 2 == 1 else "#2a1000"

                # RSI color
                if row['RSI'] > 70:
                    rsi_color = "#e74c3c"
                elif row['RSI'] < 30:
                    rsi_color = "#27ae60"
                else:
                    rsi_color = "#f39c12"

                # Downside color
                downside_color = "#e74c3c" if row['Upside'] > 0 else "#27ae60"

                # R:R color
                rr = row['Risk_Reward']
                if rr >= 2:
                    rr_color = "#27ae60"
                elif rr >= 1:
                    rr_color = "#f39c12"
                else:
                    rr_color = "#e74c3c"

                # Beta color
                if row['Beta'] > 1.5:
                    beta_color = "#e74c3c"
                elif row['Beta'] > 1.0:
                    beta_color = "#f39c12"
                else:
                    beta_color = "#27ae60"

                # P/E display
                pe_display = f"{row['PE_Ratio']:.1f}" if row['PE_Ratio'] > 0 else "N/A"
                pe_color = "#e74c3c" if row['PE_Ratio'] > 40 else ("#f39c12" if row['PE_Ratio'] > 25 else "#27ae60")

                # Quality badge
                if row['Quality'] == 'Excellent':
                    badge_color = "#1e8449"
                elif row['Quality'] == 'Good':
                    badge_color = "#2471a3"
                elif row['Quality'] == 'Average':
                    badge_color = "#d68910"
                else:
                    badge_color = "#922b21"

                html += f"""
                                <tr bgcolor="{row_bg}">
                                    <td style="color: #f0d0a0; font-weight: 700; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">{row['Name']}<br><span style="color:#8a6040; font-size:11px; font-weight:400;">{row['Symbol']}</span></td>
                                    <td style="color: #e8c090; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px; font-weight: 600;">${row['Price']:,.2f}</td>
                                    <td style="color: #e8c090; padding: 12px 10px; border: 1px solid #3d1500; font-size: 11px; font-weight: bold;">{row['Rating']}</td>
                                    <td style="color: #e67e22; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 18px;">{row['Combined_Score']:.0f}</td>
                                    <td style="color: {rsi_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 15px;">{row['RSI']:.0f}<br><span style="font-size:10px; color:#8a6040;">{row['RSI_Signal']}</span></td>
                                    <td style="color: #e8c090; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px; font-weight: 600;">{row['MACD']}</td>
                                    <td style="color: {downside_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 15px;">{row['Upside']:+.1f}%</td>
                                    <td style="color: #e8c090; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">${row['Target_1']:,.2f}</td>
                                    <td style="color: #e74c3c; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">${row['Stop_Loss']:,.2f}<br><span style="font-size:11px; color:#8a6040;">+{row['SL_Percentage']:.1f}%</span></td>
                                    <td style="color: {rr_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 15px;">{rr:.1f}x</td>
                                    <td style="color: {beta_color}; font-weight: bold; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px;">{row['Beta']:.2f}</td>
                                    <td style="color: {pe_color}; padding: 12px 10px; border: 1px solid #3d1500; font-size: 13px; font-weight: 600;">{pe_display}</td>
                                    <td style="padding: 12px 10px; border: 1px solid #3d1500;"><span style="background-color: {badge_color}; color: #ffffff; padding: 5px 8px; border-radius: 5px; font-size: 11px; font-weight: bold; white-space: nowrap;">{row['Quality']}</span></td>
                                </tr>
"""
            html += """
                            </table>
"""
        
        # Disclaimer and Footer
        next_update = "4:30 PM" if now.hour < 12 else "9:30 AM (Next Day)"
        html += f"""
                            <!-- Disclaimer -->
                            <table width="100%" cellpadding="20" cellspacing="0" border="0" style="margin: 30px 0; border-radius: 8px; background: #2a1000; border-left: 4px solid #e67e22;">
                                <tr>
                                    <td>
                                        <p style="color: #f0d0a0; margin: 0 0 10px 0;"><strong style="color: #e74c3c;">⚠️ DISCLAIMER:</strong> This analysis is for <strong style="color: #e67e22;">EDUCATIONAL PURPOSES ONLY</strong>. This is NOT financial advice. Always:</p>
                                        <ul style="color: #d4a080; margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                                            <li>Do your own research</li>
                                            <li>Consult a registered financial advisor</li>
                                            <li>Use proper risk management and stop losses</li>
                                            <li>Never invest more than you can afford to lose</li>
                                        </ul>
                                    </td>
                                </tr>
                            </table>
                            
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td align="center" style="padding: 25px; background: linear-gradient(135deg, #7b2d00, #c0392b); border-radius: 0 0 12px 12px;">
                            <p style="color: #ffffff; margin: 0 0 6px 0; font-size: 14px; font-weight: 700; letter-spacing: 0.5px;">🌅 Top US Market Influencers: NASDAQ &amp; S&amp;P 500</p>
                            <p style="color: #ffd5b0; margin: 0; font-size: 12px; opacity: 0.85;">Automated Stock Analysis System &nbsp;|&nbsp; Next Update: {next_update} EST</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
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
    main()


###############################
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
