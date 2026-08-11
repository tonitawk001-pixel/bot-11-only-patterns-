"""Runner: wait for backtest to complete, then print results"""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
result = subprocess.run([sys.executable, "backtest_3months_v7.py"], capture_output=True, text=True, timeout=300)
elapsed = time.time() - t0
output = result.stdout + result.stderr
print(f"Completed in {elapsed:.0f}s")
# Print last 40 lines
lines = output.strip().split('\n')
for line in lines[-40:]:
    print(line)
