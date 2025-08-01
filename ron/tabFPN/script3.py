import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Configuration
# ------------------------------
SAVE_DIR = 'REALMLP-SUBS'
LOG_FILE = 'real_mlp_kfold_mape_log.csv'
SLEEP_BETWEEN_RUNS = 120  # seconds (optional delay)
N_SPLITS = 5
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# Dataset
# ------------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# ------------------------------
# REAL MLP Model
# ------------------------------
class RealMLP(nn.Module):
    def __init__(self, input_dim):
        super(RealMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

# ------------------------------
# Start Infinite Loop
# ------------------------------
run_counter = 1

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

    for target_col in y.columns:
        print(f'\n📌 Target: {target_col}')
        val_preds = np.zeros_like(y[target_col].values, dtype=float)
        test_preds_folds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f'  Fold {fold + 1}/{N_SPLITS}')
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[target_col].iloc[train_idx], y[target_col].iloc[val_idx]

            train_ds = TabularDataset(X_train, y_train)
            val_ds = TabularDataset(X_val, y_val)
            test_ds = TabularDataset(test)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            model = RealMLP(input_dim=X.shape[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=LR)

            model.train()
            for epoch in range(EPOCHS):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()

            # Validation predictions
            model.eval()
            val_fold_preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    preds = model(xb).cpu().numpy()
                    val_fold_preds.extend(preds)
            val_preds[val_idx] = val_fold_preds

            # Test predictions
            test_fold_preds = []
            with torch.no_grad():
                for xb in test_loader:
                    xb = xb.to(device)
                    preds = model(xb).cpu().numpy()
                    test_fold_preds.extend(preds)
            test_preds_folds.append(test_fold_preds)

        mape = mean_absolute_percentage_error(y[target_col].values, val_preds)
        target_mapes.append(mape)
        print(f'✅ MAPE for {target_col}: {mape:.4f}')

        preds_val_all_targets.append(val_preds)
        preds_test_all_targets.append(np.mean(test_preds_folds, axis=0))

    preds_val_np = np.stack(preds_val_all_targets, axis=1)
    overall_mape = mean_absolute_percentage_error(y.values, preds_val_np)
    print(f'\n📊 Overall MAPE across all targets: {overall_mape:.4f}')

    preds_test_np = np.stack(preds_test_all_targets, axis=1)
    submission = pd.DataFrame(preds_test_np, columns=y.columns)
    submission.insert(0, 'ID', ID_test)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_name = f'submission_{timestamp}_{run_counter:03d}.csv'
    submission_path = os.path.join(SAVE_DIR, submission_name)
    submission.to_csv(submission_path, index=False)
    print(f'✅ Saved: {submission_path}')

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
