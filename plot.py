"""
plot.py — Project-root launcher for the analysis visualization pipeline.

Usage (from the project root):
    python3 plot.py
"""
import subprocess
import os
import sys

HERE   = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(HERE, "analysis", "analyze.py")

result = subprocess.run([sys.executable, script], cwd=HERE)
sys.exit(result.returncode)
