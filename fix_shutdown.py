"""
fix_shutdown.py - Disable balance auto-shutdown for 24/7 demo trading.
Usage: python fix_shutdown.py [path-to-bot-folder]
"""
import os, sys, shutil

CANDIDATE_DIRS = [
    os.getcwd(),
    os.path.dirname(os.path.abspath(__file__)),
    r"C:\Users\Administrator\Downloads\final-bot-v8.9-main",
    r"C:\Users\Administrator\Downloads\final-bot-v8.9-main\final-bot-v8.9-main",
]
TARGET = "main_super.py"


def find_bot_dir():
    if len(sys.argv) > 1:
        p = sys.argv[1]
        for sub in [p, os.path.join(p, "trading_bot_mt5")]:
            if os.path.exists(os.path.join(sub, TARGET)):
                return sub
        print("[!] Not found in: " + p)
        sys.exit(1)
    for base in CANDIDATE_DIRS:
        for sub in [base, os.path.join(base, "trading_bot_mt5")]:
            if os.path.exists(os.path.join(sub, TARGET)):
                return sub
    print("[!] main_super.py not found. Pass path explicitly.")
    sys.exit(1)


def read(p):
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write(p, c):
    with open(p, "w", encoding="utf-8") as f:
        f.write(c)


def backup(p):
    bak = p + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
        print("  [backup] " + bak)


def patch_main_super(path):
    print("\n== Patching " + path + " ==")
    backup(path)
    src = read(path)
    orig = src
    patched = False

    # 0) BULLETPROOF FLOOR: set HARD_FLOOR to 0 so "balance < HARD_FLOOR"
    #    is never True regardless of how the floor block is written.
    import re as _re
    new_src = _re.sub(r"HARD_FLOOR\s*=\s*[0-9.]+", "HARD_FLOOR = 0.0", src)
    if new_src != src:
        src = new_src
        patched = True
        print("  [OK] HARD_FLOOR constant set to 0.0 (floor can never trigger)")

    # 1) Disable 25% DD Emergency Stop
    dd_markers = [
        "DD Emergency Stop (25% from peak)",
        "check_dd_emergency(balance)",
    ]
    lines = src.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_active = line.strip() and not line.strip().startswith("#")
        if is_active and "if len(trades_log) > 0 and perf.check_dd_emergency(balance) and not perf.dd_halted:" in line:
            new_lines.append("            # DD Emergency Stop (25% from peak) - DISABLED: 24/7 demo")
            new_lines.append("            # if len(trades_log) > 0 and perf.check_dd_emergency(balance) and not perf.dd_halted:")
            new_lines.append("            #     perf.dd_halted = True")
            new_lines.append("            #     _emergency_positions = mt5.positions_get(symbol=SYMBOL)")
            new_lines.append("            #     tg.send_message(\"EMERGENCY STOP: 25% drawdown from peak. Closing all positions.\")")
            new_lines.append("            #     if _emergency_positions:")
            new_lines.append("            #         for p in _emergency_positions: conn.close_position(p.ticket)")
            new_lines.append("            #     break")
            patched = True
            # skip the original block lines (8 lines total incl. the if line)
            i += 8
        elif is_active and "if balance < HARD_FLOOR:" in line and "notify_bot_crashed" in src[i:i+200]:
            new_lines.append("            # Balance floor - DISABLED: 24/7 demo")
            new_lines.append("            # if balance < HARD_FLOOR:")
            new_lines.append("            #     tg.notify_bot_crashed(f\"Balance ${balance:.2f} below floor ${HARD_FLOOR}\")")
            new_lines.append("            #     break")
            patched = True
            i += 3
        else:
            new_lines.append(line)
            i += 1

    if patched:
        write(path, "\n".join(new_lines))
        print("  [SAVED] " + path)
    else:
        print("  [NO CHANGE] Blocks not found (may already be disabled)")


def patch_perf(path):
    print("\n== Patching " + path + " ==")
    backup(path)
    src = read(path)
    old = "return (self.peak_balance - balance) / self.peak_balance >= 0.25"
    if old in src:
        src = src.replace(
            "        return (self.peak_balance - balance) / self.peak_balance >= 0.25",
            '        """DD Emergency Stop - DISABLED: 24/7 demo."""\n        return False'
        )
        write(path, src)
        print("  [OK] check_dd_emergency() now returns False")
        print("  [SAVED] " + path)
    else:
        print("  [SKIP] already disabled or not found")


def main():
    print("=" * 60)
    print(" fix_shutdown.py - Disable balance auto-shutdown (24/7 demo)")
    print("=" * 60)
    bot_dir = find_bot_dir()
    print("Bot folder: " + bot_dir)
    ms = os.path.join(bot_dir, "main_super.py")
    pt = os.path.join(bot_dir, "performance_tracker.py")
    if not os.path.exists(ms):
        print("[!] Missing: " + ms)
        sys.exit(1)
    patch_main_super(ms)
    if os.path.exists(pt):
        patch_perf(pt)
    print("\n" + "=" * 60)
    print(" DONE - bot will now run 24/7 even if balance drops to zero.")
    print(" Restart main_super.py for changes to take effect.")
    print("=" * 60)


if __name__ == "__main__":
    main()