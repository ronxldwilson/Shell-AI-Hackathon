# test_gpu_setup.py
import sys
print("Python version:", sys.version)

# Test basic imports
try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print("❌ Pandas import failed:", e)

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print("❌ NumPy import failed:", e)

# Test GPU libraries
try:
    import cupy as cp
    print("✅ CuPy imported successfully")
    print(f"   GPU available: {cp.cuda.is_available()}")
    print(f"   GPU count: {cp.cuda.runtime.getDeviceCount()}")
except ImportError as e:
    print("❌ CuPy import failed:", e)

try:
    import cudf
    print("✅ cuDF imported successfully")
    # Test basic operation
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print(f"   cuDF test: {df.shape}")
except ImportError as e:
    print("❌ cuDF import failed:", e)
except Exception as e:
    print("❌ cuDF test failed:", e)

try:
    import cuml
    print("✅ cuML imported successfully")
    from cuml.linear_model import Ridge
    print("   cuML Ridge imported successfully")
except ImportError as e:
    print("❌ cuML import failed:", e)

try:
    import xgboost as xgb
    print("✅ XGBoost imported successfully")
    print(f"   XGBoost version: {xgb.__version__}")
    
    # Test GPU support
    data = [[1, 2], [3, 4], [5, 6]]
    labels = [0, 1, 0]
    dtrain = xgb.DMatrix(data, label=labels)
    params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    model = xgb.train(params, dtrain, num_boost_round=1)
    print("   XGBoost GPU test passed")
except Exception as e:
    print("❌ XGBoost GPU test failed:", e)

try:
    import optuna
    print("✅ Optuna imported successfully")
except ImportError as e:
    print("❌ Optuna import failed:", e)