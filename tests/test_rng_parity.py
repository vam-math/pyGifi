import pygifi_rng
import numpy as np

print("=" * 60)
print("TEST 1: rnorm stream parity")
print("=" * 60)

# R reference: set.seed(1); rnorm(10)
r_reference = [
    -0.6264538,  0.1836433, -0.8356286,  1.5952808,
     0.3295078, -0.8204684,  0.4874291,  0.7383247,
     0.5757814, -0.3053884
]

pygifi_rng.r_set_seed(1)
py_vals = pygifi_rng.r_rnorm(10)

all_pass = True
for i, (py, r) in enumerate(zip(py_vals, r_reference)):
    diff = abs(py - r)
    status = "PASS" if diff < 1e-6 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  [{i+1:02d}]  R={r:.7f}   Py={py:.7f}   diff={diff:.2e}   {status}")

print(f"\n  Stream test: {'ALL PASS' if all_pass else 'FAILED'}")

print()
print("=" * 60)
print("TEST 2: r_init_x matrix parity")
print("=" * 60)

X = pygifi_rng.r_init_x(5, 2, 1)
X_np = np.array(X)
print(f"  Shape: {X_np.shape}")
print(f"  First column: {X_np[:, 0]}")
print(f"  Second column: {X_np[:, 1]}")
print(f"  Expected col1[0]: -0.6264538  Got: {X_np[0,0]:.7f}")
print(f"  Match: {'PASS' if abs(X_np[0,0] - (-0.6264538)) < 1e-6 else 'FAIL'}")

print()
print("=" * 60)
print("TEST 3: Tolerance check — must be < 1e-6")
print("=" * 60)
