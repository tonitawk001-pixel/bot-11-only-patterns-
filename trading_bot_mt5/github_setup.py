"""
GITHUB AUTO-SETUP — Called by the bot at startup.
Configures git remote with token for private repo access.
Fully automatic — no manual steps needed on the VPS.
"""

import os
import subprocess

TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO_URL = "https://github.com/tonitawk001-pixel/mt5-bot-edited-final-verison.git"
REMOTE_NAME = "final-version"


def setup_remote():
    """Configure git remote with token. Called at bot startup. Fully automatic."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        auth_url = f"https://{TOKEN}@{REPO_URL.replace('https://', '')}"
        result = subprocess.run(
            ["git", "remote", "set-url", REMOTE_NAME, auth_url],
            cwd=project_root,
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"[GitHubSetup] Remote configured with token ✓")
            return True
        else:
            print(f"[GitHubSetup] Failed: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"[GitHubSetup] Error: {e}")
        return False