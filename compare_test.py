"""
compare_test.py — Project-level launcher for the validation pipeline.

Checks that the pygifi_rng C extension is available for exact R parity
before running the full pipeline. Build it first if not available:
    cd pygifi/rng && python3 setup_rng.py build_ext --inplace

Usage (from the project root):
    python3 compare_test.py
"""
import subprocess
import os
import sys


def check_rng_available():
    """Check if the pygifi_rng C extension is compiled and ready."""
    rng_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pygifi", "rng")
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
report = os.path.join(HERE, "validation", "report.py")

print("=" * 60)
print("  PyGifi vs R Gifi — RNG check")
print("=" * 60)
check_rng_available()

result = subprocess.run([sys.executable, report], cwd=os.path.join(HERE, "validation"))
sys.exit(result.returncode)
