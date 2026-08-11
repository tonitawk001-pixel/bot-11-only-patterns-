"""
PERFORMANCE TRACKER - DD Protection + Weekly AI Reviews + GitHub Notes
"""
import os, json, subprocess
from datetime import datetime, timedelta, timezone
from logger_mt5 import logger

PERF_STATE_FILE = "performance_state.json"
NOTES_REPO = "https://github.com/tonitawk001-pixel/mt5-bot-final-2.git"
NOTES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_notes")


class PerformanceTracker:
    def __init__(self, deepseek_client=None):
        self.peak_balance = 0.0
        self.peak_equity = 0.0
        self.last_weekly_report = None
        self.dd_halted = False
        self.dd_risk_reduced = False
        self.ai = deepseek_client
        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(PERF_STATE_FILE):
                with open(PERF_STATE_FILE) as f:
                    s = json.load(f)
                self.peak_balance = s.get("peak_balance", 0.0)
                self.peak_equity = s.get("peak_equity", 0.0)
                self.last_weekly_report = s.get("last_weekly_report")
        except:
            pass

    def _save_state(self):
        try:
            with open(PERF_STATE_FILE, "w") as f:
                json.dump({"peak_balance": self.peak_balance, "peak_equity": self.peak_equity,
                           "last_weekly_report": self.last_weekly_report}, f, default=str)
        except:
            pass

    def update(self, balance: float, equity: float):
        if balance > self.peak_balance: self.peak_balance = balance
        if equity > self.peak_equity: self.peak_equity = equity
        self._save_state()

    def check_dd_emergency(self, balance: float) -> bool:
        """DD Emergency Stop — DISABLED: bot runs 24/7 on demo account, even if balance drops to zero."""
        return False

    def check_dd_risk_reduce(self, equity: float) -> bool:
        if self.peak_equity <= 0: return False
        dd = (self.peak_equity - equity) / self.peak_equity
        if dd >= 0.10 and not self.dd_risk_reduced:
            self.dd_risk_reduced = True; return True
        if dd < 0.05 and self.dd_risk_reduced:
            self.dd_risk_reduced = False; return False
        return self.dd_risk_reduced

    def should_run_weekly(self, now: datetime) -> bool:
        if now.weekday() != 6 or now.hour != 20: return False
        if self.last_weekly_report and isinstance(self.last_weekly_report, str):
            try:
                last = datetime.fromisoformat(self.last_weekly_report)
                if (now - last).total_seconds() < 3600: return False
            except: pass
        return True


    def run_weekly_review(self, trades_log: list, balance: float) -> str:
        if self.ai is None: return "AI not available"
        self.last_weekly_report = datetime.now(timezone.utc).isoformat()
        self._save_state()
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        week_trades = []
        for t in trades_log:
            if not isinstance(t, dict): continue
            try:
                ct_str = t.get("close_time", "")
                if not ct_str: continue
                ct = datetime.fromisoformat(ct_str) if isinstance(ct_str, str) else ct_str
                if ct >= cutoff: week_trades.append(t)
            except: continue
        if not week_trades: return "No trades this week"

        wins = [t for t in week_trades if t.get("pnl", 0) > 0]
        total_pnl = sum(t.get("pnl", 0) for t in week_trades)
        wr = len(wins) / len(week_trades) * 100 if week_trades else 0
        buys = [t for t in week_trades if t.get("dir") == "BUY"]
        sells = [t for t in week_trades if t.get("dir") == "SELL"]

        prompt = f"""
ACT AS EXPERT TRADING COACH. Analyze this week:
Trades: {len(week_trades)} | WR: {wr:.1f}% | P&L: ${total_pnl:+.2f}
Balance: ${balance:.2f} | Peak: ${self.peak_balance:.2f}
BUY: {len(buys)} | SELL: {len(sells)}
OUTPUT JSON: {{"rating":"EXCELLENT/GOOD/AVERAGE/BELOW/POOR","strengths":"...","weaknesses":"...","recommendations":"...","notes":"..."}}
"""
        review = {"rating":"N/A","strengths":"","weaknesses":"","recommendations":"","notes":""}
        try:
            r = self.ai.client.chat.completions.create(
                model=self.ai.model,
                messages=[{"role":"system","content":"Trading coach. JSON only."},
                          {"role":"user","content":prompt}],
                response_format={"type":"json_object"}, max_tokens=250, timeout=10)
            review = json.loads(r.choices[0].message.content)
        except Exception as e:
            review["notes"] = f"AI failed: {str(e)[:80]}"

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report = (
            f"# Weekly Report {date_str}\n\n"
            f"## Rating: {review.get('rating','N/A')}\n\n"
            f"## Stats\n- Trades: {len(week_trades)} | WR: {wr:.1f}%\n"
            f"- P&L: ${total_pnl:+.2f} | Balance: ${balance:.2f}\n"
            f"- BUY: {len(buys)} | SELL: {len(sells)}\n\n"
            f"## AI Analysis\n"
            f"**Strengths:** {review.get('strengths','N/A')}\n\n"
            f"**Weaknesses:** {review.get('weaknesses','N/A')}\n\n"
            f"**Recommendations:** {review.get('recommendations','N/A')}\n\n"
            f"**Notes:** {review.get('notes','')}\n"
        )
        self._push_to_github(report, date_str)
        return report

    def _push_to_github(self, report: str, date_str: str):
        try:
            if not os.path.exists(NOTES_DIR): os.makedirs(NOTES_DIR)
            git_dir = os.path.join(NOTES_DIR, ".git")
            if not os.path.exists(git_dir):
                token = os.environ.get("GITHUB_TOKEN", "")
                url = NOTES_REPO
                if token: url = f"https://{token}@{NOTES_REPO.replace('https://', '')}"
                subprocess.run(["git","clone",url,NOTES_DIR], capture_output=True, timeout=60)
            path = os.path.join(NOTES_DIR, f"weekly_report_{date_str}.md")
            with open(path, "w", encoding="utf-8") as f: f.write(report)
            subprocess.run(["git","-C",NOTES_DIR,"fetch","origin"], capture_output=True, timeout=15)
            subprocess.run(["git","-C",NOTES_DIR,"reset","--hard","origin/main"], capture_output=True, timeout=15)
            subprocess.run(["git","-C",NOTES_DIR,"add","."], capture_output=True, timeout=10)
            subprocess.run(["git","-C",NOTES_DIR,"commit","-m",f"Weekly {date_str}"], capture_output=True, timeout=10)
            r = subprocess.run(["git","-C",NOTES_DIR,"push","origin","main"], capture_output=True, text=True, timeout=30)
            logger.info("[PERF] Pushed to GitHub" if r.returncode==0 else f"[PERF] Push: {r.stderr[:80]}")
        except Exception as e:
            logger.error(f"[PERF] GitHub: {e}")
