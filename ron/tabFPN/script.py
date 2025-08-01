import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch
from datetime import datetime

# ------------------------------
# Configuration
# ------------------------------
SAVE_DIR = 'submissions'
LOG_FILE = 'mape_log.csv'
SLEEP_BETWEEN_RUNS = 0  # seconds (optional delay)

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

    # Train/Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Model Training
    preds_val = []
    preds_test = []
    target_mapes = []

    for i, target_col in enumerate(y.columns):
        print(f'Training for target: {target_col}')
        model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
        model.fit(X_train.values, y_train[target_col].values)
        val_pred = model.predict(X_val.values)
        test_pred = model.predict(test.values)

        preds_val.append(val_pred)
        preds_test.append(test_pred)

        mape = mean_absolute_percentage_error(y_val[target_col].values, val_pred)
        target_mapes.append(mape)
        print(f'MAPE for {target_col}: {mape:.4f}')

    # Overall MAPE
    preds_val_np = np.stack(preds_val, axis=1)
    overall_mape = mean_absolute_percentage_error(y_val.values, preds_val_np)
    print(f'\n📊 Overall MAPE across all targets: {overall_mape:.4f}')

    # Save Submission
    preds_test_np = np.stack(preds_test, axis=1)
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
