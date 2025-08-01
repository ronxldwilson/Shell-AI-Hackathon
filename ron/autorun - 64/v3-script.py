import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tabm import TabM

# ------------------------------ Config ------------------------------

class Config:
    TRAIN_PATH = 'train.csv'
    TEST_PATH = 'test.csv'
    TARGET_COLS = [f'BlendProperty{i}' for i in range(1, 11)]
    TEST_SIZE = 0.1
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 300
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    K_ENSEMBLE = 25
    RANDOM_STATE = 42
    N_SPLITS = 5

# ------------------------------ Dataset ------------------------------

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# ------------------------------ Helpers ------------------------------

def get_next_submission_filename(folder="submission"):
    os.makedirs(folder, exist_ok=True)
    existing_files = [f for f in os.listdir(folder) if f.startswith("submission") and f.endswith(".csv")]
    numbers = [int(f.split('_')[1]) for f in existing_files if f.split('_')[1].isdigit()] if existing_files else []
    next_num = max(numbers) + 1 if numbers else 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"submission_{next_num}_{timestamp}.csv")

def prepare_data(config):
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    for col in config.TARGET_COLS:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in test_df.columns:
            test_df[col] = 0

    id_cols = ['ID', 'BlendID']
    train_df.drop(columns=[col for col in id_cols if col in train_df.columns], inplace=True)
    test_df.drop(columns=[col for col in id_cols if col in test_df.columns], inplace=True)

    feature_cols = [col for col in train_df.columns if col not in config.TARGET_COLS]

    X = train_df[feature_cols].values
    y = train_df[config.TARGET_COLS].values
    X_test = test_df[feature_cols].values
    test_ids = pd.read_csv(config.TEST_PATH)['ID'].values if 'ID' in pd.read_csv(config.TEST_PATH).columns else np.arange(len(test_df))

    num_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X = num_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)
    X_test = num_scaler.transform(X_test)

    return X, y, X_test, test_ids, feature_cols, config.TARGET_COLS, num_scaler, y_scaler

# ------------------------------ Trainer ------------------------------

class Trainer:
    def __init__(self, config, X_train, y_train, X_val, y_val, X_test, test_ids, feature_cols, y_scaler):
        self.config = config
        self.device = config.DEVICE
        self.criterion_mse = nn.MSELoss()

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.test_ids = test_ids
        self.feature_cols = feature_cols
        self.target_cols = config.TARGET_COLS
        self.y_scaler = y_scaler

        self.train_loader = DataLoader(TabularDataset(self.X_train, self.y_train), batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(TabularDataset(self.X_val, self.y_val), batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(TabularDataset(self.X_test), batch_size=self.config.BATCH_SIZE, shuffle=False)

        self.create_model()

    def create_model(self):
        self.model = TabM.make(
            n_num_features=self.train_loader.dataset.X.shape[1],
            cat_cardinalities=None,
            d_out=len(self.target_cols),
            k=self.config.K_ENSEMBLE
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LR)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            preds = self.model(X)
            preds_median = torch.median(preds, dim=1)[0]
            loss = self.criterion_mse(preds_median, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                preds_median = torch.median(preds, dim=1)[0]
                loss = self.criterion_mse(preds_median, y)
                total_loss += loss.item()
                all_preds.append(preds_median.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds_all = np.vstack(all_preds)
        targets_all = np.vstack(all_targets)
        preds_orig = self.y_scaler.inverse_transform(preds_all)
        targets_orig = self.y_scaler.inverse_transform(targets_all)
        val_mae = mean_absolute_error(targets_orig, preds_orig)
        return total_loss / len(self.val_loader), val_mae

    def train(self):
        best_mae = float('inf')
        best_model_state = None
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss, val_mae = self.validate()
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE (orig): {val_mae:.6f}")
            if val_mae < best_mae:
                best_mae = val_mae
                best_model_state = self.model.state_dict()
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Loaded best model with Val MAE (orig): {best_mae:.6f}")

    def predict(self):
        self.model.eval()
        preds_all = []
        with torch.no_grad():
            for X in self.test_loader:
                X = X.to(self.device)
                preds = self.model(X)
                preds_median = torch.median(preds, dim=1)[0]
                preds_all.append(preds_median.cpu().numpy())
        preds_all = np.vstack(preds_all)
        return self.y_scaler.inverse_transform(preds_all)

# ------------------------------ Infinite K-Fold Loop ------------------------------

if __name__ == '__main__':
    run_count = 1
    try:
        while True:
            print(f"\n🔁 Starting K-Fold Run #{run_count}...\n")
            config = Config()
            X, y, X_test, test_ids, feature_cols, target_cols, num_scaler, y_scaler = prepare_data(config)
            kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

            all_preds = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                print(f"\n================ Fold {fold + 1} ================")
                trainer = Trainer(
                    config,
                    X_train=X[train_idx],
                    y_train=y[train_idx],
                    X_val=X[val_idx],
                    y_val=y[val_idx],
                    X_test=X_test,
                    test_ids=test_ids,
                    feature_cols=feature_cols,
                    y_scaler=y_scaler
                )
                trainer.train()
                preds = trainer.predict()
                all_preds.append(preds)

            final_preds = np.mean(all_preds, axis=0)
            submission = pd.DataFrame(final_preds, columns=config.TARGET_COLS)
            submission.insert(0, 'ID', test_ids)
            submission_path = get_next_submission_filename(folder="submission")
            submission.to_csv(submission_path, index=False)
            print(f"\n✅ Finished Run #{run_count} | Submission saved to {submission_path}\n")
            run_count += 1

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
