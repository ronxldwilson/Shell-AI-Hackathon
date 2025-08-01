import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
import torch
from datetime import datetime

# ------------------------------
# Configuration
# ------------------------------
SAVE_DIR = 'submissions-kfold'
LOG_FILE = 'kfold_mape_log.csv'
SLEEP_BETWEEN_RUNS = 30  # seconds (optional delay)
N_SPLITS = 5

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize run counter
run_counter = 1

# ------------------------------
# Start Infinite Loop
# ------------------------------
while True:
    print(f"\n🚀 Starting run #{run_counter} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load Data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    ID_test = test['ID']
    test = test.drop(columns=['ID'])

    X = train.iloc[:, :55]
    y = train.iloc[:, 55:]
    preds_val_all_targets = []
    preds_test_all_targets = []
    target_mapes = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kf = KFold(n_splits=N_SPLITS, shuffle=True)

    # Train per target using CV
    for target_col in y.columns:
        print(f'\n📌 Target: {target_col}')
        val_preds = np.zeros_like(y[target_col].values, dtype=float)
        test_preds_folds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f'  Fold {fold + 1}/{N_SPLITS}')
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[target_col].iloc[train_idx], y[target_col].iloc[val_idx]

            model = TabPFNRegressor(device=device)
            model.fit(X_train.values, y_train.values)

            val_pred = model.predict(X_val.values)
            val_preds[val_idx] = val_pred

            test_pred = model.predict(test.values)
            test_preds_folds.append(test_pred)

        # Compute MAPE for this target
        mape = mean_absolute_percentage_error(y[target_col].values, val_preds)
        target_mapes.append(mape)
        print(f'✅ MAPE for {target_col}: {mape:.4f}')

        preds_val_all_targets.append(val_preds)
        preds_test_all_targets.append(np.mean(test_preds_folds, axis=0))

    # Overall MAPE
    preds_val_np = np.stack(preds_val_all_targets, axis=1)
    overall_mape = mean_absolute_percentage_error(y.values, preds_val_np)
    print(f'\n📊 Overall MAPE across all targets: {overall_mape:.4f}')

    # Save Submission
    preds_test_np = np.stack(preds_test_all_targets, axis=1)
    submission = pd.DataFrame(preds_test_np, columns=y.columns)
    submission.insert(0, 'ID', ID_test)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_name = f'submission_{timestamp}_{run_counter:03d}.csv'
    submission_path = os.path.join(SAVE_DIR, submission_name)
    submission.to_csv(submission_path, index=False)
    print(f'✅ Saved: {submission_path}')

    # Log Results
    log_entry = {
        'timestamp': timestamp,
        'run': run_counter,
        'submission_file': submission_name,
        'overall_mape': overall_mape,
    }
    for i, col in enumerate(y.columns):
        log_entry[f'mape_{col}'] = target_mapes[i]

    log_df = pd.DataFrame([log_entry])
    if not os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, index=False)
    else:
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    print(f'📝 Logged results to {LOG_FILE}')

    run_counter += 1

    if SLEEP_BETWEEN_RUNS > 0:
        print(f'⏳ Sleeping for {SLEEP_BETWEEN_RUNS} seconds...')
        time.sleep(SLEEP_BETWEEN_RUNS)
