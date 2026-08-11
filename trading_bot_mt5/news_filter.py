"""
NEWS FILTER MODULE
==================
Fetches high-impact economic news events from public calendars.
Prevents trading during red-folder events.
Uses primary + fallback API sources.
"""
import requests
from datetime import datetime, timedelta, timezone
from logger_mt5 import logger

# Primary and fallback news API endpoints
NEWS_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_this_week.json",
    "https://fcsapi.com/api-v3/forex/economic_calendar?type=high&access_key=demo",
]

class NewsFilter:
    def __init__(self):
        self.red_folder_events = []
        self.last_update = None
        self.last_successful_update = None
        self.last_error = ""

    def update_news(self):
        """Fetch high impact news for the current week from primary or fallback."""
        for url in NEWS_URLS:
            try:
                logger.info(f"[NEWS] Fetching from: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    events_before = len(self.red_folder_events)
                    self.red_folder_events = []
                    for event in data:
                        if event.get('impact') == 'High' and event.get('country') in ['USD', 'ALL']:
                            try:
                                date_str = event.get('date', '')
                                for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                                    try:
                                        event_time = datetime.strptime(date_str, fmt)
                                        if event_time.tzinfo is None:
                                            event_time = event_time.replace(tzinfo=timezone.utc)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    continue
                                self.red_folder_events.append({
                                    'title': event.get('title', 'Unknown'),
                                    'time': event_time
                                })
                            except Exception:
                                continue
                    self.last_update = datetime.now(timezone.utc)
                    self.last_successful_update = datetime.now(timezone.utc)
                    self.last_error = ""
                    logger.info(f"[NEWS] SUCCESS from {url[:40]}... Found {len(self.red_folder_events)} red-folder events.")
                    return True
                else:
                    logger.warning(f"[NEWS] HTTP {response.status_code} from {url[:50]}... — trying next")
                    self.last_error = f"HTTP {response.status_code}"
            except requests.exceptions.Timeout:
                logger.warning(f"[NEWS] TIMEOUT from {url[:50]}... — trying next")
                self.last_error = "Timeout"
            except requests.exceptions.ConnectionError:
                logger.warning(f"[NEWS] CONNECTION ERROR from {url[:50]}... — trying next")
                self.last_error = "Connection refused"
            except Exception as e:
                logger.warning(f"[NEWS] Error from {url[:50]}...: {e} — trying next")
                self.last_error = str(e)[:80]
        
        self.last_update = datetime.now(timezone.utc)
        logger.warning(f"[NEWS] ALL SOURCES FAILED. Last error: {self.last_error}")
        return False

    def has_news(self) -> bool:
        """Check if we successfully loaded news recently."""
        if not self.last_successful_update:
            return False
        return (datetime.now(timezone.utc) - self.last_successful_update).total_seconds() < 86400

    def is_news_active(self, buffer_minutes=30):
        """Check if we are currently near a high-impact news event.
        Returns (is_active, event_title, event_time_str, pause_minutes)."""
        if not self.last_update or (datetime.now(timezone.utc) - self.last_update).total_seconds() > 43200:
            self.update_news()

        now = datetime.now(timezone.utc)
        for event in self.red_folder_events:
            diff = abs((now - event['time']).total_seconds() / 60.0)
            if diff <= buffer_minutes:
                time_str = event['time'].strftime("%H:%M UTC") if hasattr(event['time'], 'strftime') else str(event['time'])
                logger.warning(f"[NEWS] Pause active! Event: {event['title']} at {time_str}")
                return True, event['title'], time_str, buffer_minutes
        return False, None, None, buffer_minutes

    def get_news_impact_context(self) -> str:
        """
        Returns structured news analysis for AI decision-making.
        Includes upcoming events with timing and recently passed events
        that may still be influencing the market.
        """
        if not self.red_folder_events:
            return "No high-impact news events for today."

        now = datetime.now(timezone.utc)
        upcoming = [e for e in self.red_folder_events if e['time'] > now]
        recent = [e for e in self.red_folder_events if now - timedelta(hours=4) <= e['time'] <= now]

        lines = []
        if recent:
            lines.append("RECENT HIGH-IMPACT EVENTS (last 4 hours):")
            for e in recent:
                time_str = e['time'].strftime("%H:%M UTC") if hasattr(e['time'], 'strftime') else str(e['time'])
                mins_ago = int((now - e['time']).total_seconds() / 60)
                lines.append(f"  - {e['title']} at {time_str} ({mins_ago} min ago)")
                if "FOMC" in e['title'] or "Fed" in e['title'] or "Rate" in e['title']:
                    lines.append(f"    IMPACT: Sets market direction. Hawkish=SELL gold, Dovish=BUY gold")
                elif "NFP" in e['title'] or "Employment" in e['title'] or "Payroll" in e['title']:
                    lines.append(f"    IMPACT: Major volatility event.")
                elif "CPI" in e['title'] or "Inflation" in e['title']:
                    lines.append(f"    IMPACT: Inflation data. Higher=hawkish(SELL), Lower=dovish(BUY)")
                elif "GDP" in e['title']:
                    lines.append(f"    IMPACT: Growth data, affects dollar/gold.")
            lines.append("")

        if upcoming:
            lines.append("UPCOMING HIGH-IMPACT EVENTS:")
            for e in upcoming[:3]:
                time_str = e['time'].strftime("%H:%M UTC") if hasattr(e['time'], 'strftime') else str(e['time'])
                mins_until = int((e['time'] - now).total_seconds() / 60)
                lines.append(f"  - {e['title']} at {time_str} (in {mins_until} min)")
                if mins_until < 90:
                    lines.append(f"    WARNING: Within 90 min, market may range/drift.")
            lines.append("")

        if not upcoming and not recent:
            return "No high-impact news events today. Market is in technical flow."

        return "\n".join(lines)