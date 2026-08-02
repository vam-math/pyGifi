"""
Tests-side launcher for the full PyGifi versus R Gifi validation pipeline.
"""

import os
import subprocess
import sys


def check_rng_available(root_dir):
    rng_dir = os.path.join(root_dir, "pygifi", "rng")
    if rng_dir not in sys.path:
        sys.path.insert(0, rng_dir)
    try:
        import pygifi_rng  # noqa: F401
        print("  [RNG] pygifi_rng extension loaded — exact R parity mode")
        return True
    except ImportError:
        print("  [RNG] WARNING: pygifi_rng not found — using SVD fallback")
        print("  [RNG] Build it: cd pygifi/rng && python3 setup_rng.py build_ext --inplace")
        return False


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
report = os.path.join(ROOT, "validation", "report.py")

print("=" * 60)
print("  PyGifi vs R Gifi — RNG check")
print("=" * 60)
check_rng_available(ROOT)

result = subprocess.run([sys.executable, report], cwd=os.path.join(ROOT, "validation"))
sys.exit(result.returncode)
