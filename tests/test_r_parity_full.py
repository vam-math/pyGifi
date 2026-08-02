import subprocess
import os
import pytest

def test_r_parity():
    import shutil
    import sys
    
    if not shutil.which("Rscript"):
        import pytest
        pytest.skip("Rscript not found in PATH")
        
    # Check if Gifi R package is installed
    res_pkg = subprocess.run(["Rscript", "-e", "library(Gifi)"], capture_output=True)
    if res_pkg.returncode != 0:
        import pytest
        pytest.skip("Gifi R package is not installed")
        
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validation_dir = os.path.join(root_dir, "validation")
    pipeline_script = os.path.join(root_dir, "tests", "parity", "run_validation_pipeline.py")

    if not os.path.exists(pipeline_script):
        pytest.skip("Parity validation scripts not found on disk")

    res_pipe = subprocess.run(
        [sys.executable, pipeline_script],
        cwd=root_dir,
    )
    assert res_pipe.returncode == 0, "Parity validation pipeline failed"
