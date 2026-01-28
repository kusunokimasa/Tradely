"""
Tradely - Professional Trading Journal
======================================
Tradely is a professional trading journal and performance analytics platform.
Designed for traders who want to track, analyse, and improve their trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import calendar
import json
import os

# Page configuration
st.set_page_config(
    page_title="Tradely - Trading Journal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for TradeZella-like styling
st.markdown("""
<style>
    .main {
        background-color: #0F1419;
        color: white;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: 1px solid #2EA043;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2EA043;
        border-color: #3FB950;
    }
    .metric-card {
        background-color: #1E2329;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #30363D;
    }
    .profit-text {
        color: #00D084 !important;
    }
    .loss-text {
        color: #FF4757 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trades' not in st.session_state:
    if os.path.exists('trades_data.csv'):
        st.session_state.trades = pd.read_csv('trades_data.csv')
    else:
        st.session_state.trades = pd.DataFrame(columns=[
            'Trade_ID', 'Date', 'Time', 'Direction', 'Entry_Price', 'Exit_Price',
            'Stop_Loss', 'Take_Profit', 'Lots', 'Pips', 'PnL', 'Strategy',
            'Timeframe', 'Session', 'Exit_Reason', 'Risk_Reward', 'Confluence',
            'Mood', 'Market_Condition', 'Journal', 'Followed_Plan'
        ])

def save_trades():
    """Save trades to CSV"""
    st.session_state.trades.to_csv('trades_data.csv', index=False)

def delete_trade(trade_id):
    """Delete a trade by ID"""
    st.session_state.trades = st.session_state.trades[st.session_state.trades['Trade_ID'] != trade_id]
    save_trades()
    st.success(f"✅ Trade {trade_id} deleted!")
    st.rerun()

def calculate_pnl(direction, entry, exit_price, lots):
    """Calculate P&L and pips"""
    if direction == 'BUY':
        pips = (exit_price - entry) * 10000
        pnl = (exit_price - entry) * lots * 100000
    else:
        pips = (entry - exit_price) * 10000
        pnl = (entry - exit_price) * lots * 100000
    return round(pips, 1), round(pnl, 2)

def get_stats():
    """Calculate performance statistics"""
    trades = st.session_state.trades
    if len(trades) == 0:
        return {}
    
    total = len(trades)
    wins = len(trades[trades['PnL'] > 0])
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0
    
    total_pnl = trades['PnL'].sum()
    avg_win = trades[trades['PnL'] > 0]['PnL'].mean() if wins > 0 else 0
    avg_loss = trades[trades['PnL'] < 0]['PnL'].mean() if losses > 0 else 0
    
    gross_profit = trades[trades['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(trades[trades['PnL'] < 0]['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_pips': trades['Pips'].sum()
    }

# Sidebar navigation
st.sidebar.title("📊 Tradely")
page = st.sidebar.radio("Navigation", ["Dashboard", "Add Trade", "Trade History", "Manage Trades", "Analytics", "Calendar View"])

# ==================== DASHBOARD PAGE ====================
if page == "Dashboard":
    st.title("📈 Trading Dashboard")
    
    stats = get_stats()
    
    if stats:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total P&L", f"${stats['total_pnl']:,.2f}", 
                     delta=f"{stats['total_pnl']/100:.1f}%")
        with col2:
            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        with col3:
            st.metric("Total Trades", stats['total_trades'])
        with col4:
            pf_display = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "∞"
            st.metric("Profit Factor", pf_display)
        with col5:
            st.metric("Avg Win", f"${stats['avg_win']:,.2f}")
        with col6:
            st.metric("Avg Loss", f"${stats['avg_loss']:,.2f}")
        
        st.markdown("---")
        
        st.subheader("Recent Trades")
        recent = st.session_state.trades.tail(10).iloc[::-1]
        
        display_df = recent[['Trade_ID', 'Date', 'Time', 'Direction', 'Entry_Price', 'Exit_Price', 
                            'PnL', 'Pips', 'Strategy', 'Confluence']].copy()
        display_df['PnL'] = display_df['PnL'].apply(lambda x: f"${x:,.2f}" if x > 0 else f"-${abs(x):,.2f}")
        display_df['Entry_Price'] = display_df['Entry_Price'].apply(lambda x: f"{x:.5f}")
        display_df['Exit_Price'] = display_df['Exit_Price'].apply(lambda x: f"{x:.5f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Win/Loss Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[stats['wins'], stats['losses']],
                hole=.4,
                marker_colors=['#00D084', '#FF4757']
            )])
            fig.update_layout(
                paper_bgcolor='#0F1419',
                plot_bgcolor='#0F1419',
                font_color='white',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("P&L by Strategy")
            if len(st.session_state.trades) > 0:
                strategy_pnl = st.session_state.trades.groupby('Strategy')['PnL'].sum().reset_index()
                fig = px.bar(strategy_pnl, x='Strategy', y='PnL',
                            color='PnL', 
                            color_continuous_scale=[(0, '#FF4757'), (0.5, '#FFD700'), (1, '#00D084')])
                fig.update_layout(
                    paper_bgcolor='#0F1419',
                    plot_bgcolor='#0F1419',
                    font_color='white',
                    xaxis_gridcolor='#30363D',
                    yaxis_gridcolor='#30363D'
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet. Go to 'Add Trade' to start journaling!")

# ==================== ADD TRADE PAGE ====================
elif page == "Add Trade":
    st.title("➕ Add New Trade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trade Details")
        date = st.date_input("Date", datetime.now())
        time = st.time_input("Time", datetime.now().time())
        direction = st.selectbox("Direction", ["BUY", "SELL"])
        
        entry_price = st.number_input("Entry Price", min_value=0.0, value=1.19450, format="%.5f")
        exit_price = st.number_input("Exit Price", min_value=0.0, value=1.19800, format="%.5f")
        stop_loss = st.number_input("Stop Loss", min_value=0.0, value=1.19200, format="%.5f")
        # Take profit with 1 decimal place
        take_profit = st.number_input("Take Profit", min_value=0.0, value=1.2, format="%.1f")
        lots = st.number_input("Lots", min_value=0.01, value=1.0, step=0.1)
    
    with col2:
        st.subheader("Analysis & Journal")
        
        # Custom strategy input
        strategy_option = st.selectbox("Strategy", [
            "Custom", "Trend Following", "Breakout", "Mean Reversion", 
            "Scalping", "Swing Trading", "News Trading"
        ])
        
        if strategy_option == "Custom":
            strategy = st.text_input("Enter Strategy Name", placeholder="e.g., My Strategy")
        else:
            strategy = strategy_option
        
        timeframe = st.selectbox("Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        session = st.selectbox("Session", ["Asian", "London", "NY", "London/NY Overlap"])
        exit_reason = st.selectbox("Exit Reason", [
            "Take Profit", "Stop Loss", "Manual Close", 
            "Trailing Stop", "Time Stop", "Breakeven"
        ])
        confluence = st.selectbox("Setup Quality", ["A+", "B", "C"])
        mood = st.selectbox("Your Mood", ["Confident", "Neutral", "Hesitant", "FOMO", "Revenge"])
        market_condition = st.selectbox("Market Condition", ["Trending", "Ranging", "Volatile", "Quiet"])
        
        risk_reward = st.text_input("Risk:Reward", "1:2")
        followed_plan = st.checkbox("Followed Trading Plan?", value=True)
    
    st.subheader("Trade Journal")
    journal = st.text_area("Notes", placeholder="Describe your trade setup, emotions, lessons learned...")
    
    if st.button("💾 SAVE TRADE", key="save_trade"):
        pips, pnl = calculate_pnl(direction, entry_price, exit_price, lots)
        
        new_trade = {
            'Trade_ID': f"EUR{len(st.session_state.trades)+1:03d}",
            'Date': date.strftime('%Y-%m-%d'),
            'Time': time.strftime('%H:%M'),
            'Direction': direction,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Stop_Loss': stop_loss,
            'Take_Profit': take_profit,
            'Lots': lots,
            'Pips': pips,
            'PnL': pnl,
            'Strategy': strategy,
            'Timeframe': timeframe,
            'Session': session,
            'Exit_Reason': exit_reason,
            'Risk_Reward': risk_reward,
            'Confluence': confluence,
            'Mood': mood,
            'Market_Condition': market_condition,
            'Journal': journal,
            'Followed_Plan': followed_plan
        }
        
        st.session_state.trades = pd.concat([
            st.session_state.trades, 
            pd.DataFrame([new_trade])
        ], ignore_index=True)
        
        save_trades()
        
        st.success(f"✅ Trade saved! P&L: ${pnl:,.2f} ({pips:+.1f} pips)")
        st.metric("Total Trades", len(st.session_state.trades))

# ==================== TRADE HISTORY PAGE ====================
elif page == "Trade History":
    st.title("📜 Trade History")
    
    if len(st.session_state.trades) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            strategy_filter = st.multiselect("Strategy", 
                st.session_state.trades['Strategy'].unique())
        with col2:
            direction_filter = st.multiselect("Direction", ["BUY", "SELL"])
        with col3:
            confluence_filter = st.multiselect("Setup Quality", ["A+", "B", "C"])
        
        filtered = st.session_state.trades.copy()
        if strategy_filter:
            filtered = filtered[filtered['Strategy'].isin(strategy_filter)]
        if direction_filter:
            filtered = filtered[filtered['Direction'].isin(direction_filter)]
        if confluence_filter:
            filtered = filtered[filtered['Confluence'].isin(confluence_filter)]
        
        st.dataframe(filtered, use_container_width=True)
        
        if st.button("📥 Export to CSV"):
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='trade_history.csv',
                mime='text/csv'
            )
    else:
        st.info("No trades recorded yet.")

# ==================== MANAGE TRADES PAGE (DELETE) ====================
elif page == "Manage Trades":
    st.title("🗑️ Manage Trades")
    
    if len(st.session_state.trades) > 0:
        st.warning("⚠️ Deleting trades is permanent! Export your data first if needed.")
        
        for idx, trade in st.session_state.trades.iterrows():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{trade['Trade_ID']}** - {trade['Date']} {trade['Time']}")
                st.write(f"{trade['Direction']} @ {trade['Entry_Price']:.5f} → {trade['Exit_Price']:.5f}")
            
            with col2:
                pnl_color = "#00D084" if trade['PnL'] > 0 else "#FF4757"
                st.markdown(f"<span style='color: {pnl_color}; font-weight: bold;'>${trade['PnL']:,.2f}</span>", unsafe_allow_html=True)
                st.write(f"{trade['Pips']:.1f} pips")
            
            with col3:
                st.write(f"Strategy: {trade['Strategy']}")
                st.write(f"Grade: {trade['Confluence']}")
            
            with col4:
                if st.button("🗑️ Delete", key=f"delete_{trade['Trade_ID']}"):
                    delete_trade(trade['Trade_ID'])
            
            st.markdown("---")
    else:
        st.info("No trades to manage.")

# ==================== ANALYTICS PAGE ====================
elif page == "Analytics":
    st.title("📊 Performance Analytics")
    
    if len(st.session_state.trades) > 0:
        trades = st.session_state.trades
        
        st.subheader("Cumulative P&L")
        trades_sorted = trades.sort_values('Date')
        trades_sorted['Cumulative_PnL'] = trades_sorted['PnL'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_sorted['Date'],
            y=trades_sorted['Cumulative_PnL'],
            fill='tozeroy',
            line=dict(color='#00D084'),
            fillcolor='rgba(0, 208, 132, 0.2)'
        ))
        fig.update_layout(
            paper_bgcolor='#0F1419',
            plot_bgcolor='#0F1419',
            font_color='white',
            xaxis_gridcolor='#30363D',
            yaxis_gridcolor='#30363D'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance by Day")
            trades['DayOfWeek'] = pd.to_datetime(trades['Date']).dt.day_name()
            daily_perf = trades.groupby('DayOfWeek')['PnL'].sum().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
            ])
            
            fig = px.bar(daily_perf, 
                        color=daily_perf.values,
                        color_continuous_scale=[(0, '#FF4757'), (0.5, '#FFD700'), (1, '#00D084')])
            fig.update_layout(
                paper_bgcolor='#0F1419',
                plot_bgcolor='#0F1419',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Mood vs Performance")
            mood_perf = trades.groupby('Mood')['PnL'].mean()
            fig = px.bar(mood_perf, 
                        color=mood_perf.values,
                        color_continuous_scale=[(0, '#FF4757'), (0.5, '#FFD700'), (1, '#00D084')])
            fig.update_layout(
                paper_bgcolor='#0F1419',
                plot_bgcolor='#0F1419',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add trades to see analytics!")

# ==================== CALENDAR VIEW PAGE ====================
elif page == "Calendar View":
    st.title("📅 Calendar View")
    
    if len(st.session_state.trades) > 0:
        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("Year", [2025, 2026], index=1)
        with col2:
            month = st.selectbox("Month", range(1, 13), format_func=lambda x: calendar.month_name[x])
        
        trades = st.session_state.trades.copy()
        trades['Date'] = pd.to_datetime(trades['Date'])
        month_trades = trades[(trades['Date'].dt.year == year) & (trades['Date'].dt.month == month)]
        
        daily_summary = month_trades.groupby(month_trades['Date'].dt.day).agg({
            'PnL': 'sum',
            'Trade_ID': 'count'
        }).reset_index()
        daily_summary.columns = ['Day', 'Total_PnL', 'Num_Trades']
        
        cal = calendar.Calendar()
        month_days = cal.monthdayscalendar(year, month)
        
        st.markdown("### " + calendar.month_name[month] + " " + str(year))
        
        cols = st.columns(7)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, day in enumerate(days):
            cols[i].markdown(f"**{day}**")
        
        for week in month_days:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day == 0:
                    cols[i].empty()
                else:
                    day_data = daily_summary[daily_summary['Day'] == day]
                    
                    if len(day_data) > 0:
                        pnl = day_data.iloc[0]['Total_PnL']
                        num = int(day_data.iloc[0]['Num_Trades'])
                        color = "#00D084" if pnl > 0 else "#FF4757"
                        bg_color = '#0D2818' if pnl > 0 else '#280D0D'
                        
                        cols[i].markdown(f"""
                        <div style="background-color: {bg_color}; 
                                    padding: 10px; border-radius: 5px; border: 1px solid {color};">
                            <strong style="color: white;">{day}</strong><br>
                            <span style="color: {color}; font-size: 18px; font-weight: bold;">
                                {'+' if pnl > 0 else ''}${pnl:.0f}
                            </span><br>
                            <span style="color: #CCCCCC; font-size: 12px;">{num} trades</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # White text for "No trades"
                        cols[i].markdown(f"""
                        <div style="background-color: #1E2329; padding: 10px; border-radius: 5px; border: 1px solid #444;">
                            <strong style="color: #FFFFFF;">{day}</strong><br>
                            <span style="color: #AAAAAA; font-size: 12px;">No trades</span>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No trades to display on calendar.")

# Footer with YOUR branding
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Tradely**

Tradely is a professional trading journal and performance analytics platform. 

Designed for traders who want to track, analyse, and improve their trading.
""")
