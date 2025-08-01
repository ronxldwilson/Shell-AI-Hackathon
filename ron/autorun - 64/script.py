import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tabm import TabM  # Ensure TabM is installed and imported

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
    K_ENSEMBLE = 64
    RANDOM_STATE = 42

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
    numbers = []
    for f in existing_files:
        parts = f.replace(".csv", "").split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            numbers.append(int(parts[1]))
    next_num = max(numbers) + 1 if numbers else 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"submission_{next_num}_{timestamp}.csv")

# ------------------------------ Trainer ------------------------------

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.criterion_mse = nn.MSELoss()
        self.load_and_prepare_data()
        self.create_model()

    def load_and_prepare_data(self):
        train_df = pd.read_csv(self.config.TRAIN_PATH)
        test_df = pd.read_csv(self.config.TEST_PATH)
        train_df = self.feature_engineering(train_df, fit=True)
        test_df = self.feature_engineering(test_df, fit=False)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            train_df.drop(columns=self.target_cols).values,
            train_df[self.target_cols].values,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

        self.X_test = test_df.drop(columns=self.target_cols, errors='ignore').values
        self.test_ids = test_df['ID'].values if 'ID' in test_df.columns else np.arange(len(test_df))

        self.num_scaler = RobustScaler()
        self.y_scaler = RobustScaler()

        self.X_train = self.num_scaler.fit_transform(self.X_train)
        self.X_val = self.num_scaler.transform(self.X_val)
        self.X_test = self.num_scaler.transform(self.X_test)

        self.y_train = self.y_scaler.fit_transform(self.y_train)
        self.y_val = self.y_scaler.transform(self.y_val)

        self.train_loader = DataLoader(TabularDataset(self.X_train, self.y_train), batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(TabularDataset(self.X_val, self.y_val), batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(TabularDataset(self.X_test), batch_size=self.config.BATCH_SIZE, shuffle=False)

    def feature_engineering(self, df, fit=False):
        df = df.copy()
        for col in Config.TARGET_COLS:
            if col not in df.columns:
                df[col] = 0

        id_cols = ['ID', 'BlendID']
        df.drop(columns=[col for col in id_cols if col in df.columns], inplace=True)

        if fit:
            self.feature_cols = [col for col in df.columns if col not in Config.TARGET_COLS]
            self.target_cols = Config.TARGET_COLS

        df = df[self.feature_cols + self.target_cols]
        return df

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
        for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}"):
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

        preds_all_orig = self.y_scaler.inverse_transform(preds_all)
        targets_all_orig = self.y_scaler.inverse_transform(targets_all)
        mape = mean_absolute_percentage_error(targets_all_orig, preds_all_orig) * 100

        return total_loss / len(self.val_loader), mape

    def train(self):
        best_mape = float('inf')
        best_model_state = None

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss, val_mape = self.validate()
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAPE: {val_mape:.2f}%")

            if val_mape < best_mape:
                best_mape = val_mape
                best_model_state = self.model.state_dict()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Loaded best model with Val MAPE: {best_mape:.2f}%")

        return best_mape

    def predict(self, submission_path=None, val_mape=None):
        self.model.eval()
        preds_all = []

        with torch.no_grad():
            for X in self.test_loader:
                X = X.to(self.device)
                preds = self.model(X)
                preds_median = torch.median(preds, dim=1)[0]
                preds_all.append(preds_median.cpu().numpy())

        preds_all = np.vstack(preds_all)
        preds_orig = self.y_scaler.inverse_transform(preds_all)

        submission = pd.DataFrame(preds_orig, columns=self.target_cols)
        submission.insert(0, 'ID', self.test_ids)

        if submission_path is None:
            submission_path = "submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"✅ Submission saved to {submission_path}")

        if val_mape is not None:
            with open("submission/val_log.csv", "a") as f:
                f.write(f"{submission_path},{val_mape:.4f}\n")

# ------------------------------ Run in Loop ------------------------------

if __name__ == '__main__':
    run_num = 1
    try:
        while True:
            print(f"🚀 Starting run #{run_num}")
            config = Config()
            trainer = Trainer(config)
            best_val_mape = trainer.train()
            submission_file = get_next_submission_filename(folder="submission")
            trainer.predict(submission_file, val_mape=best_val_mape)
            print(f"✅ Finished run #{run_num}\n")
            run_num += 1

    except KeyboardInterrupt:
        print("🛑 Training loop stopped by user.")
