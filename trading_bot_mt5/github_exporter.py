"""
GitHub Exporter — Auto-upload trade history, AI analysis, DeepSeek chats
=========================================================================
Pushes analysis files to GitHub as a second remote or commits to the repo.
"""

import os
import json
import subprocess
import time
from datetime import datetime, timezone
from logger_mt5 import logger

# Files to track and push
TRACKED_FILES = [
    "deepseek_chat_log.jsonl",     # DeepSeek chat conversations
    "deepseek_analysis_log.jsonl",  # DeepSeek market analysis history
    "performance_state.json",       # Performance tracker data
    "bot_state_super.json",         # Bot state
    "trade_history/",               # Trade history folder
]

# Push interval (seconds) — 30 minutes
PUSH_INTERVAL = 1800
_last_push = 0


def setup_git():
    """Ensure git is configured for this repo."""
    try:
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            logger.warning("[GitExport] Not a git repository — cannot auto-export")
            return False
        
        # Set user if not set
        subprocess.run(
            ["git", "config", "user.name", "tonitawk001-pixel"],
            capture_output=True, timeout=5
        )
        subprocess.run(
            ["git", "config", "user.email", "tonitawk001-pixel@users.noreply.github.com"],
            capture_output=True, timeout=5
        )
        return True
    except Exception as e:
        logger.warning(f"[GitExport] Setup failed: {e}")
        return False


def should_push() -> bool:
    """Check if enough time has passed since last push."""
    global _last_push
    now = time.time()
    if now - _last_push >= PUSH_INTERVAL:
        _last_push = now
        return True
    return False


def force_push_next():
    """Force the next push to happen immediately."""
    global _last_push
    _last_push = 0


def push_analysis():
    """Commit and push tracked analysis files to GitHub."""
    global _last_push
    if not should_push():
        return
    
    try:
        # Add tracked files (only if they exist)
        files_to_add = []
        for pattern in TRACKED_FILES:
            if pattern.endswith("/"):
                # Directory — check if it exists and has files
                if os.path.isdir(pattern):
                    files_to_add.append(pattern)
            else:
                if os.path.exists(pattern):
                    files_to_add.append(pattern)
        
        if not files_to_add:
            logger.debug("[GitExport] No analysis files to push")
            return
        
        # Add files
        for f in files_to_add:
            subprocess.run(["git", "add", f], capture_output=True, timeout=10)
        
        # Also add the config overrides if they exist
        if os.path.exists("config_overrides.json"):
            subprocess.run(["git", "add", "config_overrides.json"], capture_output=True, timeout=5)
        
        # Check if there's anything to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10
        )
        if not result.stdout.strip():
            logger.debug("[GitExport] Nothing new to commit")
            return
        
        # Commit
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        commit_msg = f"[Bot] Auto-update analysis & logs — {timestamp}"
        subprocess.run(
            ["git", "commit", "-m", commit_msg, "--no-verify"],
            capture_output=True, timeout=15
        )
        
        # Detect branch name (support both master and main)
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "main"
        
        # Push
        push_result = subprocess.run(
            ["git", "push", "origin", branch],
            capture_output=True, text=True, timeout=30
        )
        
        if push_result.returncode == 0:
            msg = f"[GitExport] ✅ Push successful to {branch} — {timestamp}"
            logger.info(msg)
            return {"success": True, "branch": branch, "message": msg}
        else:
            # If push fails (e.g., remote changes), try pull --rebase first
            logger.warning(f"[GitExport] Push failed, attempting rebase...")
            subprocess.run(
                ["git", "pull", "--rebase", "origin", branch],
                capture_output=True, timeout=30
            )
            retry = subprocess.run(
                ["git", "push", "origin", branch],
                capture_output=True, text=True, timeout=30
            )
            if retry.returncode == 0:
                msg = f"[GitExport] ✅ Push successful to {branch} after rebase"
                logger.info(msg)
                return {"success": True, "branch": branch, "message": msg}
            else:
                err = retry.stderr[:200] if retry.stderr else "Unknown error"
                logger.warning(f"[GitExport] Push failed: {err}")
                return {"success": False, "branch": branch, "error": err}
                
    except Exception as e:
        err = str(e)[:200]
        logger.warning(f"[GitExport] Failed: {err}")
        return {"success": False, "branch": "unknown", "error": err}
