import subprocess
import os

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
    
    r_script = os.path.join(validation_dir, "r_scripts", "run_gifi.R")
    py_script = os.path.join(validation_dir, "python_scripts", "run_pygifi.py")
    comp_script = os.path.join(validation_dir, "compare", "compare_results.py")
    
    # Skip if parity scripts aren't available locally
    if not all(os.path.exists(p) for p in [r_script, py_script, comp_script]):
        pytest.skip("Parity validation scripts not found on disk")
        
    # Run the preprocessor to clean datasets
    res_prep = subprocess.run(
        [sys.executable, "preprocess_datasets.py"],
        cwd=validation_dir,
    )
    assert res_prep.returncode == 0, "Dataset preprocessor failed"
    
    # Run R benchmark script
    res_r = subprocess.run(
        ["Rscript", "run_gifi.R"],
        cwd=os.path.join(validation_dir, "r_scripts"),
    )
    assert res_r.returncode == 0, "R benchmark runner failed"
    
    # Run Pygifi benchmark script
    res_py = subprocess.run(
        [sys.executable, "run_pygifi.py"],
        cwd=os.path.join(validation_dir, "python_scripts"),
    )
    assert res_py.returncode == 0, "Python benchmark runner failed"
    
    # Run result comparator script
    res_comp = subprocess.run(
        [sys.executable, "compare_results.py"],
        cwd=os.path.join(validation_dir, "compare"),
    )
    assert res_comp.returncode == 0, "Parity Result comparison failed bounds check"
