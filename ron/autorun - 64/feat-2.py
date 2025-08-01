import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    TEST_SIZE = 0.2
    BATCH_SIZE = 128
    LR = 1e-3
    EPOCHS = 150
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    K_ENSEMBLE = 64  # Increased ensemble size
    RANDOM_STATE = 42
    
    # More conservative feature selection
    CORRELATION_THRESHOLD = 0.98  # Higher threshold
    MAX_FEATURES = 200  # Allow more features
    USE_TARGET_SCALING = False  # Try without target scaling first

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

def safe_mape(y_true, y_pred, epsilon=1e-3):
    """Calculate MAPE with protection against division by small values"""
    # Filter out values where |y_true| < epsilon to avoid division by tiny numbers
    mask = np.abs(y_true) >= epsilon
    if np.sum(mask) == 0:
        return 0.0  # All values are near zero
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    
    # If MAPE is still unreasonable, use symmetric MAPE instead
    if mape > 200:
        smape = np.mean(2 * np.abs(y_true_filtered - y_pred_filtered) / 
                       (np.abs(y_true_filtered) + np.abs(y_pred_filtered) + epsilon)) * 100
        return smape
    
    return mape

# ------------------------------ Trainer ------------------------------

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        self.load_and_prepare_data()
        self.create_model()

    def load_and_prepare_data(self):
        print("📊 Loading data...")
        train_df = pd.read_csv(self.config.TRAIN_PATH)
        test_df = pd.read_csv(self.config.TEST_PATH)
        
        print(f"📊 Raw data - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Basic data info
        print(f"📊 Train targets info:")
        for col in self.config.TARGET_COLS:
            if col in train_df.columns:
                vals = train_df[col].dropna()
                print(f"   {col}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")
        
        train_df = self.feature_engineering(train_df, fit=True)
        test_df = self.feature_engineering(test_df, fit=False)
        
        print(f"📊 After preprocessing - Train: {train_df.shape}, Test: {test_df.shape}")

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            train_df.drop(columns=self.target_cols).values,
            train_df[self.target_cols].values,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=None  # Remove stratification for regression
        )

        self.X_test = test_df.drop(columns=self.target_cols, errors='ignore').values
        self.test_ids = test_df['ID'].values if 'ID' in test_df.columns else np.arange(len(test_df))

        print(f"📊 Split shapes - Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")

        # Feature scaling
        self.feature_scaler = StandardScaler()  # Try StandardScaler instead of RobustScaler
        self.X_train = self.feature_scaler.fit_transform(self.X_train)
        self.X_val = self.feature_scaler.transform(self.X_val)
        self.X_test = self.feature_scaler.transform(self.X_test)

        # Target scaling (optional)
        self.y_train_orig = self.y_train.copy()
        self.y_val_orig = self.y_val.copy()
        
        if self.config.USE_TARGET_SCALING:
            print("📊 Scaling targets...")
            self.y_scaler = StandardScaler()
            self.y_train = self.y_scaler.fit_transform(self.y_train)
            self.y_val = self.y_scaler.transform(self.y_val)
        else:
            print("📊 Not scaling targets")
            self.y_scaler = None

        # Create data loaders
        self.train_loader = DataLoader(
            TabularDataset(self.X_train, self.y_train), 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            drop_last=True  # For stability with batch norm
        )
        self.val_loader = DataLoader(
            TabularDataset(self.X_val, self.y_val), 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False
        )
        self.test_loader = DataLoader(
            TabularDataset(self.X_test), 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False
        )

    def feature_engineering(self, df, fit=False):
        df = df.copy()

        # Ensure all target columns exist
        for col in self.config.TARGET_COLS:
            if col not in df.columns:
                df[col] = 0

        # Store target columns
        if fit:
            self.target_cols = self.config.TARGET_COLS

        # Drop ID columns
        id_cols = ['ID', 'BlendID']
        df.drop(columns=[col for col in id_cols if col in df.columns], inplace=True)

        if fit:
            self.original_features = [col for col in df.columns if col not in self.target_cols]
            print(f"📊 Original features: {len(self.original_features)}")

        # Keep only known features
        available_features = [col for col in self.original_features if col in df.columns]
        df = df[available_features + self.target_cols]

        # Handle missing values more carefully
        if fit:
            self.fill_strategies = {}
            for col in available_features:
                if df[col].dtype in ['object', 'category']:
                    self.fill_strategies[col] = df[col].mode().iloc[0] if not df[col].mode().empty else 'missing'
                else:
                    # Use median for robust filling
                    self.fill_strategies[col] = df[col].median()

        for col in available_features:
            if col in df.columns:
                df[col] = df[col].fillna(self.fill_strategies[col])

        # More conservative outlier handling
        if fit:
            self.outlier_bounds = {}
            for col in available_features:
                if df[col].dtype not in ['object', 'category']:
                    # Use 3 IQRs instead of percentiles
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    self.outlier_bounds[col] = (lower_bound, upper_bound)

        for col in available_features:
            if col in self.outlier_bounds and col in df.columns:
                lower, upper = self.outlier_bounds[col]
                df[col] = df[col].clip(lower, upper)

        # Feature selection with higher correlation threshold
        if fit:
            numeric_features = [col for col in available_features if col in df.columns and df[col].dtype not in ['object', 'category']]
            
            if len(numeric_features) > 1:
                corr_matrix = df[numeric_features].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # Find features to drop with higher threshold
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.config.CORRELATION_THRESHOLD)]
                self.corr_features_to_drop = to_drop
                print(f"📊 Dropping {len(to_drop)} highly correlated features (threshold: {self.config.CORRELATION_THRESHOLD})")
            else:
                self.corr_features_to_drop = []

        # Apply correlation-based dropping
        remaining_features = [col for col in available_features if col not in self.corr_features_to_drop and col in df.columns]

        # Feature selection by variance (if still too many features)
        if len(remaining_features) > self.config.MAX_FEATURES:
            if fit:
                numeric_remaining = [col for col in remaining_features if df[col].dtype not in ['object', 'category']]
                if len(numeric_remaining) > self.config.MAX_FEATURES:
                    feature_vars = df[numeric_remaining].var().sort_values(ascending=False)
                    self.selected_features = feature_vars.head(self.config.MAX_FEATURES).index.tolist()
                    # Add back non-numeric features if any
                    non_numeric = [col for col in remaining_features if col not in numeric_remaining]
                    self.selected_features.extend(non_numeric)
                    print(f"📊 Selected top {len(self.selected_features)} features by variance")
                else:
                    self.selected_features = remaining_features
            remaining_features = [col for col in self.selected_features if col in df.columns]

        if fit:
            self.final_features = remaining_features

        # Keep only final features plus targets
        final_cols = [col for col in self.final_features + self.target_cols if col in df.columns]
        df = df[final_cols]

        print(f"📊 Final features count: {len([col for col in self.final_features if col in df.columns])}")
        return df

    def create_model(self):
        input_size = self.train_loader.dataset.X.shape[1]
        output_size = len(self.target_cols)
        
        print(f"📊 Creating TabM model: input_size={input_size}, output_size={output_size}, k={self.config.K_ENSEMBLE}")
        
        self.model = TabM.make(
            n_num_features=input_size,
            cat_cardinalities=None,
            d_out=output_size,
            k=self.config.K_ENSEMBLE
        ).to(self.device)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.EPOCHS,
            eta_min=self.config.LR * 0.01
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_mae_loss = 0
        
        for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            preds = self.model(X)  # Shape: [batch_size, k, output_size]
            preds_mean = torch.mean(preds, dim=1)  # Use mean instead of median
            
            # Combined loss: MSE + MAE
            mse_loss = self.criterion_mse(preds_mean, y)
            mae_loss = self.criterion_mae(preds_mean, y)
            loss = mse_loss + 0.1 * mae_loss  # Small MAE component
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += mse_loss.item()
            total_mae_loss += mae_loss.item()

        return total_loss / len(self.train_loader), total_mae_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                preds_mean = torch.mean(preds, dim=1)  # Use mean instead of median
                loss = self.criterion_mse(preds_mean, y)
                total_loss += loss.item()

                all_preds.append(preds_mean.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds_all = np.vstack(all_preds)
        targets_all = np.vstack(all_targets)

        # Handle inverse transform correctly
        if self.y_scaler is not None:
            preds_all_orig = self.y_scaler.inverse_transform(preds_all)
            targets_all_orig = self.y_scaler.inverse_transform(targets_all)
        else:
            preds_all_orig = preds_all
            targets_all_orig = targets_all

        # Calculate metrics more carefully
        mae = np.mean(np.abs(preds_all_orig - targets_all_orig))
        rmse = np.sqrt(np.mean((preds_all_orig - targets_all_orig) ** 2))
        
        # Safe MAPE calculation with better filtering
        mape = safe_mape(targets_all_orig.flatten(), preds_all_orig.flatten())
        
        # Alternative metric: Normalized MAE (more stable than MAPE)
        normalized_mae = mae / (np.mean(np.abs(targets_all_orig)) + 1e-8) * 100
        
        # R² calculation
        ss_res = np.sum((targets_all_orig - preds_all_orig) ** 2)
        ss_tot = np.sum((targets_all_orig - np.mean(targets_all_orig)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        print(f"    📊 Val Metrics - MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%, NMAE: {normalized_mae:.2f}%, R²: {r2:.4f}")
        print(f"    📊 Pred range: [{preds_all_orig.min():.6f}, {preds_all_orig.max():.6f}]")
        print(f"    📊 True range: [{targets_all_orig.min():.6f}, {targets_all_orig.max():.6f}]")
        
        # Check for problematic small values
        small_values = np.sum(np.abs(targets_all_orig.flatten()) < 0.1)
        total_values = len(targets_all_orig.flatten())
        print(f"    📊 Small values (|y| < 0.1): {small_values}/{total_values} ({small_values/total_values*100:.1f}%)")

        # Use NMAE as primary metric if MAPE is unreliable
        primary_metric = normalized_mae if mape > 100 else mape
        return total_loss / len(self.val_loader), primary_metric, mae, r2, mape

    def train(self):
        print(f"🚀 Starting training on {self.device}")
        best_mape = float('inf')
        best_model_state = None
        best_raw_mape = float('inf')  # Keep track of actual MAPE
        patience_counter = 0
        patience = 20

        for epoch in range(self.config.EPOCHS):
            train_loss, train_mae = self.train_epoch(epoch)
            val_loss, val_metric, val_mae, val_r2, raw_mape = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1:3d}/{self.config.EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val Metric: {val_metric:7.2f}% | "
                  f"Val R²: {val_r2:6.4f} | "
                  f"LR: {current_lr:.6f}")

            # Save best model based on primary metric
            if val_metric < best_mape:
                best_mape = val_metric
                best_raw_mape = raw_mape  # Store the actual MAPE value
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"    ✅ New best metric: {best_mape:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"    🛑 Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Loaded best model with metric: {best_mape:.2f}%")

        return best_raw_mape  # Return actual MAPE for logging

    def predict(self, submission_path=None, val_mape=None):
        print("🔮 Generating predictions...")
        self.model.eval()
        preds_all = []

        with torch.no_grad():
            for X in self.test_loader:
                X = X.to(self.device)
                preds = self.model(X)
                preds_mean = torch.mean(preds, dim=1)  # Use mean for consistency
                preds_all.append(preds_mean.cpu().numpy())

        preds_all = np.vstack(preds_all)

        # Handle inverse transform
        if self.y_scaler is not None:
            preds_orig = self.y_scaler.inverse_transform(preds_all)
        else:
            preds_orig = preds_all

        print(f"📊 Final predictions range: [{preds_orig.min():.6f}, {preds_orig.max():.6f}]")

        # Create submission
        submission = pd.DataFrame(preds_orig, columns=self.target_cols)
        submission.insert(0, 'ID', self.test_ids)

        if submission_path is None:
            submission_path = "submission.csv"
        
        submission.to_csv(submission_path, index=False)
        print(f"✅ Submission saved to {submission_path}")

        # Log results - Always log the actual MAPE value
        if val_mape is not None:
            os.makedirs("submission", exist_ok=True)
            log_entry = f"{submission_path},{val_mape:.4f},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            with open("submission/val_log.csv", "a") as f:
                f.write(log_entry)
        
        return submission_path

# ------------------------------ Main Loop ------------------------------

if __name__ == '__main__':
    run_num = 1
    
    # Create log file header
    os.makedirs("submission", exist_ok=True)
    if not os.path.exists("submission/val_log.csv"):
        with open("submission/val_log.csv", "w") as f:
            f.write("submission_file,val_mape,timestamp\n")  # Always log MAPE
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"🚀 STARTING RUN #{run_num}")
            print(f"{'='*60}")
            
            config = Config()
            trainer = Trainer(config)
            best_val_mape = trainer.train()  # This now returns the actual MAPE
            
            submission_file = get_next_submission_filename()
            trainer.predict(submission_file, val_mape=best_val_mape)
            
            print(f"\n✅ COMPLETED RUN #{run_num} - Best MAPE: {best_val_mape:.2f}%")
            print(f"{'='*60}")
            
            run_num += 1

    except KeyboardInterrupt:
        print("\n🛑 Training loop stopped by user.")
    except Exception as e:
        print(f"\n❌ Error in run #{run_num}: {str(e)}")
        import traceback
        traceback.print_exc()