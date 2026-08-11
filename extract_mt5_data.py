"""Extract all MT5 account data — trades, P&L, win rate, open positions"""
import MetaTrader5 as mt5
from datetime import datetime, timezone

if not mt5.initialize():
    print("ERROR: Cannot connect to MT5. Make sure MT5 is running.")
    exit(1)

# ── Account info ──
info = mt5.account_info()
if info:
    print("="*60)
    print("ACCOUNT INFO")
    print("="*60)
    print(f"  Login: {info.login}")
    print(f"  Server: {info.server}")
    print(f"  Balance: ${info.balance:,.2f}")
    print(f"  Equity: ${info.equity:,.2f}")
    print(f"  Margin: ${info.margin:,.2f}")
    print(f"  Free Margin: ${info.margin_free:,.2f}")

# ── Open positions ──
positions = mt5.positions_get(symbol="XAUUSD")
print(f"\n{'='*60}")
print(f"OPEN POSITIONS: {len(positions) if positions else 0}")
print(f"{'='*60}")
if positions:
    for p in positions:
        direction = "BUY" if p.type == 0 else "SELL"
        print(f"  #{p.ticket} | {direction} | Lot: {p.volume} | Entry: ${p.price_open:.2f}")
        print(f"    SL: ${p.sl:.2f} | TP: ${p.tp:.2f} | Profit: ${p.profit:+.2f}")

# ── Closed deals (trade history) ──
# Get deals for XAUUSD in the last 90 days
from datetime import datetime, timedelta
end = datetime.now()
start = end - timedelta(days=90)

deals = mt5.history_deals_get(start, end, group="XAUUSD")

print(f"\n{'='*60}")
print(f"CLOSED TRADES (XAUUSD, last 90 days)")
print(f"{'='*60}")

if deals is None or len(deals) == 0:
    print("  No trades found. Try a smaller date range or check symbol name.")
else:
    # Filter to entry/exit deals (not balance/correction)
    trades = []
    # MT5 logs two deals per trade: DEAL_ENTRY_IN + DEAL_ENTRY_OUT
    # We group by position ID to get full trades
    from collections import defaultdict
    position_groups = defaultdict(list)
    
    for d in deals:
        if d.symbol == "XAUUSD":
            position_groups[d.position_id].append(d)
    
    # Now process each group
    for pos_id, group in position_groups.items():
        entry_deal = None
        exit_deal = None
        for d in group:
            if d.entry == 1:  # DEAL_ENTRY_IN
                entry_deal = d
            elif d.entry == 0:  # DEAL_ENTRY_OUT
                exit_deal = d
        
        if entry_deal and exit_deal:
            direction = "BUY" if entry_deal.type == 0 else "SELL"
            pnl = exit_deal.profit + (entry_deal.profit if entry_deal.profit != 0 else 0)
            if pnl == 0:
                pnl = exit_deal.profit
            
            entry_time = datetime.fromtimestamp(entry_deal.time, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_deal.time, tz=timezone.utc)
            
            trades.append({
                "pos_id": pos_id,
                "dir": direction,
                "entry_price": entry_deal.price,
                "exit_price": exit_deal.price,
                "lot": entry_deal.volume,
                "pnl": pnl,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_commission": entry_deal.commission if hasattr(entry_deal, 'commission') else 0,
                "exit_commission": exit_deal.commission if hasattr(exit_deal, 'commission') else 0,
            })
    
    # Also try getting order history for additional data
    from_date = start.timestamp()
    to_date = end.timestamp()
    
    if not trades:
        # Fallback: try history_orders
        orders = mt5.history_orders_get(from_date, to_date, group="XAUUSD")
        if orders:
            print(f"  Found {len(orders)} orders (alternative method)")
    
    # Display trades
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    
    trades.sort(key=lambda x: x["entry_time"])
    
    for i, t in enumerate(trades):
        print(f"  {t['entry_time'].strftime('%Y-%m-%d %H:%M')} | {t['dir']} | Lot: {t['lot']:.2f}")
        print(f"    Entry: ${t['entry_price']:.2f} -> Exit: ${t['exit_price']:.2f} | PnL: ${t['pnl']:+.2f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"TRADE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)" if trades else "  Wins: 0")
    print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)" if trades else "  Losses: 0")
    
    total_pnl = sum(t["pnl"] for t in trades)
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    
    avg_win = sum(t["pnl"] for t in wins) / max(len(wins), 1)
    avg_loss = sum(t["pnl"] for t in losses) / max(len(losses), 1)
    
    total_commission = sum(t["entry_commission"] + t["exit_commission"] for t in trades)
    
    print(f"  Total PnL: ${total_pnl:+,.2f}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg Win: ${avg_win:+,.2f}")
    print(f"  Avg Loss: ${avg_loss:+,.2f}")
    print(f"  Commission: ${total_commission:+,.2f}")
    print(f"  Net After Commission: ${total_pnl + total_commission:+,.2f}")
    
    # Monthly breakdown
    monthly = {}
    for t in trades:
        m = t["entry_time"].strftime("%Y-%m")
        monthly[m] = monthly.get(m, 0) + t["pnl"]
    
    print(f"\n  Monthly P&L:")
    for m in sorted(monthly.keys()):
        print(f"    {m}: ${monthly[m]:+,.2f}")

mt5.shutdown()
print(f"\nData extraction complete.")