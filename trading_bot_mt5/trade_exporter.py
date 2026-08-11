"""
TRADE EXPORTER — Auto-pushes trade history to GitHub every hour
==============================================================
Exports trades_log to CSV, generates analysis, and commits/pushes to GitHub.

Trades are taken from bot_state_super.json (saved by the bot every cycle).
Every 60 min the bot calls export_and_push() which:
  1. Reads trades from saved state
  2. Appends new trades to trade_history/trades_all.csv
  3. Generates trade_history/trades_{timestamp}.csv (snapshot)
  4. Generates trade_history/analysis.json (hourly stats)
  5. Runs git add, git commit, git push

Usage from main bot:
  from trade_exporter import export_trades, last_export_cycle
  if cycle - last_export_cycle >= 60:
      export_trades(balance, open_positions, trades_log)
      last_export_cycle = cycle
"""

import os
import json
import subprocess
import csv
from datetime import datetime, timezone

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_history")

# Track which cycle we last exported at
last_export_cycle = 0

# Track which trades we've already exported (to avoid duplicates)
exported_trade_ids = set()


def _ensure_history_dir():
    """Create trade_history directory if it doesn't exist."""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
        print(f"[TradeExporter] Created directory: {HISTORY_DIR}")


def _read_state():
    """Read trades_log from bot_state_super.json."""
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_state_super.json")
    if not os.path.exists(state_file):
        return []
    try:
        with open(state_file) as f:
            state = json.load(f)
        return state.get("trades_log", [])
    except Exception as e:
        print(f"[TradeExporter] Error reading state: {e}")
        return []


def _trades_to_csv_rows(trades):
    """Convert trades list to CSV rows."""
    rows = []
    for t in trades:
        # Create unique ID for dedup
        tid = f"{t.get('open_time', '')}_{t.get('entry', 0)}_{t.get('dir', '')}_{t.get('close_time', '')}"
        rows.append({
            "trade_id": tid,
            "open_time": t.get("open_time", ""),
            "close_time": t.get("close_time", ""),
            "direction": t.get("dir", ""),
            "entry_price": t.get("entry", 0),
            "exit_price": t.get("close_price", 0),
            "stop_loss": t.get("sl", 0),
            "take_profit": t.get("tp", 0),
            "lot_size": t.get("lot", 0),
            "pnl": t.get("pnl", 0),
            "exit_reason": t.get("reason", "?"),
            "score": t.get("score", 0),
            "regime": t.get("regime", ""),
            "be_reached": t.get("be", False),
        })
    return rows


def _append_to_csv_all(rows):
    """Append new trades to the cumulative CSV file."""
    csv_path = os.path.join(HISTORY_DIR, "trades_all.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["trade_id", "open_time", "close_time", "direction", "entry_price",
                      "exit_price", "stop_loss", "take_profit", "lot_size", "pnl",
                      "exit_reason", "score", "regime", "be_reached"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            if row["trade_id"] not in exported_trade_ids:
                writer.writerow(row)
                exported_trade_ids.add(row["trade_id"])

    return csv_path


def _write_snapshot_csv(rows):
    """Write a timestamped snapshot CSV."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(HISTORY_DIR, f"trades_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["trade_id", "open_time", "close_time", "direction", "entry_price",
                      "exit_price", "stop_loss", "take_profit", "lot_size", "pnl",
                      "exit_reason", "score", "regime", "be_reached"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def _write_analysis_json(balance, open_positions, trades):
    """Write analysis.json with hourly statistics."""
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = total_trades - wins
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    
    win_pnls = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
    loss_pnls = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) <= 0]
    
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else 0
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.get("reason", "?")
        reasons[r] = reasons.get(r, 0) + 1

    # Direction breakdown
    buys = sum(1 for t in trades if t.get("dir") == "BUY")
    sells = sum(1 for t in trades if t.get("dir") == "SELL")

    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": {
            "balance": round(balance, 2),
            "open_positions": open_positions,
        },
        "summary": {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 1),
            "net_pnl": round(total_pnl, 2),
            "profit_factor": round(pf, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
        },
        "breakdown": {
            "exit_reasons": reasons,
            "buys": buys,
            "sells": sells,
        }
    }

    json_path = os.path.join(HISTORY_DIR, "analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    return json_path


def _git_push():
    """Run git add, commit, push in the project root."""
    try:
        # Project root is 1 level up from trading_bot_mt5/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        history_rel = "trading_bot_mt5/trade_history"

        # Git add
        subprocess.run(
            ["git", "add", f"{history_rel}/"],
            cwd=project_root,
            capture_output=True, text=True, timeout=30
        )

        # Git commit
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        commit_msg = f"[Auto] Trade history update - {timestamp}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg, "--no-status"],
            cwd=project_root,
            capture_output=True, text=True, timeout=30
        )

        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            return True  # No new trades, that's fine

        # Git push to final-version remote
        result = subprocess.run(
            ["git", "push", "final-version", "main"],
            cwd=project_root,
            capture_output=True, text=True, timeout=60
        )

        if "Everything up-to-date" in result.stdout or "Everything up-to-date" in result.stderr:
            return True

        if result.returncode == 0:
            print(f"[TradeExporter] Pushed to GitHub ✓")
            return True
        else:
            print(f"[TradeExporter] Push stderr: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[TradeExporter] Git operation timed out")
        return False
    except Exception as e:
        print(f"[TradeExporter] Git error: {e}")
        return False


def check_for_updates():
    """
    Check GitHub for new commits. If found, pull and return update info.
    
    Returns:
        dict with: {'updated': bool, 'commit': str, 'message': str, 'error': str}
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Fetch remote branches (lightweight, only downloads metadata)
        result = subprocess.run(
            ["git", "fetch", "final-version"],
            cwd=project_root,
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {"updated": False, "error": f"fetch failed: {result.stderr[:100]}"}
        
        # Check current vs remote
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..final-version/main"],
            cwd=project_root,
            capture_output=True, text=True, timeout=10
        )
        behind_count = result.stdout.strip()
        if not behind_count or behind_count == "0":
            return {"updated": False}  # Already up to date
        
        # Get the latest commit info from remote
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h|%s", "final-version/main"],
            cwd=project_root,
            capture_output=True, text=True, timeout=10
        )
        commit_info = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        # Stash any local changes (like trade history) to avoid conflicts
        subprocess.run(
            ["git", "stash"],
            cwd=project_root,
            capture_output=True, text=True, timeout=10
        )
        
        # Pull the update
        result = subprocess.run(
            ["git", "pull", "final-version", "main"],
            cwd=project_root,
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0:
            # Get the new HEAD commit info after pull
            result = subprocess.run(
                ["git", "log", "-1", "--format=%h|%s"],
                cwd=project_root,
                capture_output=True, text=True, timeout=5
            )
            new_head = result.stdout.strip() if result.returncode == 0 else commit_info
            
            print(f"[AutoUpdate] ✓ Updated! New HEAD: {new_head}")
            return {
                "updated": True,
                "commit": new_head,
                "message": f"Bot updated to {new_head}",
                "error": None
            }
        else:
            return {
                "updated": False,
                "error": f"pull failed: {result.stderr[:200]}"
            }
            
    except subprocess.TimeoutExpired:
        print(f"[AutoUpdate] Git operation timed out")
        return {"updated": False, "error": "timeout"}
    except Exception as e:
        print(f"[AutoUpdate] Error: {e}")
        return {"updated": False, "error": str(e)}


def export_trades(balance=0, open_positions=0, trades_log=None):
    """
    Main export function. Called by the bot every hour.
    
    Args:
        balance: Current account balance
        open_positions: Number of open positions  
        trades_log: List of trade dicts from the bot
    """
    global last_export_cycle

    _ensure_history_dir()

    # Get trades from the passed trades_log or from state file
    trades = trades_log if trades_log else _read_state()

    if not trades:
        print(f"[TradeExporter] No trades to export yet")
        _write_analysis_json(balance, open_positions, [])
        _git_push()
        return

    # Convert to CSV rows and dedup
    rows = _trades_to_csv_rows(trades)

    # Append to cumulative CSV
    csv_all = _append_to_csv_all(rows)
    print(f"[TradeExporter] Cumulative trades: {len(exported_trade_ids)}")

    # Write snapshot (only new since last export)
    snapshot_path = _write_snapshot_csv(rows)
    print(f"[TradeExporter] Snapshot: {snapshot_path}")

    # Write analysis
    analysis_path = _write_analysis_json(balance, open_positions, trades)
    print(f"[TradeExporter] Analysis: {analysis_path}")

    # Push to GitHub
    success = _git_push()
    if success:
        print(f"[TradeExporter] Trade history pushed to GitHub")
    else:
        print(f"[TradeExporter] Git push failed (will retry next cycle)")

    # Also push to notes repo
    _push_trades_to_notes(trades, balance, open_positions)


def _push_trades_to_notes(trades, balance, open_positions):
    """Push trade summary to performance notes repo."""
    try:
        import subprocess, json, os
        from datetime import datetime, timezone
        notes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_notes")
        if not os.path.exists(notes_dir):
            os.makedirs(notes_dir)

        # Write current trade summary
        summary = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "balance": balance,
            "open_positions": open_positions,
            "total_trades": len(trades),
            "trades": trades[-20:]  # last 20 trades
        }
        path = os.path.join(notes_dir, "trades_live.json")
        with open(path, "w") as f:
            json.dump(summary, f, default=str, indent=2)

        # Git push if notes repo exists
        git_dir = os.path.join(notes_dir, ".git")
        if os.path.exists(git_dir):
            subprocess.run(["git","-C",notes_dir,"add","trades_live.json"], capture_output=True, timeout=10)
            subprocess.run(["git","-C",notes_dir,"commit","-m","Update trades"], capture_output=True, timeout=10)
            subprocess.run(["git","-C",notes_dir,"push","origin","main"], capture_output=True, timeout=30)
    except Exception as e:
        print(f"[TradeExporter] Notes push: {e}")