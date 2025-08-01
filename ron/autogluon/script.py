# import pandas as pd
# from autogluon.tabular import TabularDataset, TabularPredictor

# # ------------------------------ Config ------------------------------
# TRAIN_PATH = 'train.csv'
# TEST_PATH = 'test.csv'
# OUTPUT_PATH = 'submission.csv'

# TARGET_COLUMNS = [f'BlendProperty{i}' for i in range(1, 11)]
# ID_COLUMN = 'ID'

# # ------------------------------ Load Data ------------------------------
# train_df = pd.read_csv(TRAIN_PATH)
# test_df = pd.read_csv(TEST_PATH)

# # ------------------------------ Prepare Data ------------------------------
# # Input features = all columns except target columns
# feature_columns = [col for col in train_df.columns if col not in TARGET_COLUMNS]

# # Drop ID from training features if it exists
# if ID_COLUMN in feature_columns:
#     feature_columns.remove(ID_COLUMN)

# # ------------------------------ Train Predictor ------------------------------
# predictor = TabularPredictor(
#     label=TARGET_COLUMNS,
#     problem_type='regression',
#     eval_metric='mean_absolute_percentage_error'
# ).fit(
#     train_data=train_df[feature_columns + TARGET_COLUMNS],
#     presets='best_quality',
#     time_limit=3600  # 1 hour limit for tuning
# )

# # ------------------------------ Predict ------------------------------
# # Get only feature columns (test has ID too)
# test_features = test_df[feature_columns]

# # Predict multiple targets
# predictions = predictor.predict(test_features)

# # ------------------------------ Create Submission ------------------------------
# # Reattach ID column
# submission = pd.concat([test_df[ID_COLUMN], predictions], axis=1)

# # Ensure correct column order
# submission = submission[[ID_COLUMN] + TARGET_COLUMNS]

# # Save to CSV
# submission.to_csv(OUTPUT_PATH, index=False)

# print(f"✅ Submission saved to {OUTPUT_PATH}")


# import pandas as pd
# from autogluon.tabular import TabularPredictor

# # ------------------------------ Config ------------------------------
# TRAIN_PATH = 'train.csv'
# TEST_PATH = 'test.csv'
# OUTPUT_PATH = 'submission.csv'

# TARGET_COLUMNS = [f'BlendProperty{i}' for i in range(1, 11)]
# ID_COLUMN = 'ID'

# # ------------------------------ Load Data ------------------------------
# train_df = pd.read_csv(TRAIN_PATH)
# test_df = pd.read_csv(TEST_PATH)

# # ------------------------------ Prepare Data ------------------------------
# # Input features = all columns except target columns and ID
# feature_columns = [col for col in train_df.columns if col not in TARGET_COLUMNS + [ID_COLUMN]]

# # Storage for all predictions
# all_predictions = pd.DataFrame()

# # ------------------------------ Loop over each target ------------------------------
# for target in TARGET_COLUMNS:
#     print(f"🚀 Training for {target}...")
#     predictor = TabularPredictor(
#         label=target,
#         problem_type='regression',
#         eval_metric='mean_absolute_percentage_error'
#     ).fit(
#         train_data=train_df[feature_columns + [target]],
#         presets='best_quality',
#         time_limit=3600  # adjust time limit as needed per target
#     )
    
#     # Predict for this target
#     y_pred = predictor.predict(test_df[feature_columns])
#     all_predictions[target] = y_pred

# # ------------------------------ Attach ID and Save ------------------------------
# submission = pd.concat([test_df[ID_COLUMN], all_predictions], axis=1)
# submission = submission[[ID_COLUMN] + TARGET_COLUMNS]
# submission.to_csv(OUTPUT_PATH, index=False)

# print(f"✅ Submission saved to {OUTPUT_PATH}")
#########################################################


import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ------------------------------ Configuration ------------------------------
class Config:
    TRAIN_PATH = 'train.csv'
    TEST_PATH = 'test.csv'
    OUTPUT_PATH = 'submission.csv'
    TARGET_COLUMNS = [f'BlendProperty{i}' for i in range(1, 11)]
    ID_COLUMN = 'ID'
    
    # Training configuration
    TIME_LIMIT = 3600  # seconds per target
    PRESETS = 'best_quality'  # 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
    EVAL_METRIC = 'mean_absolute_percentage_error'
    
    # Cross-validation and model selection
    NUM_FOLDS = 5
    RANDOM_STATE = 42
    
    # Output configuration
    SAVE_MODELS = True
    MODEL_DIR = 'autogluon_models'
    LOG_LEVEL = logging.INFO

# ------------------------------ Setup Logging ------------------------------
def setup_logging():
    """Setup logging configuration with UTF-8 encoding"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'autogluon_training_{timestamp}.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ------------------------------ Data Validation ------------------------------
def validate_data(train_df, test_df, logger):
    """Validate input data and check for common issues"""
    logger.info("Validating data...")
    
    # Check if files exist and are readable
    issues = []
    
    # Check basic structure
    if train_df.empty or test_df.empty:
        issues.append("Empty dataframes detected")
    
    # Check for required columns
    missing_targets = [col for col in Config.TARGET_COLUMNS if col not in train_df.columns]
    if missing_targets:
        issues.append(f"Missing target columns: {missing_targets}")
    
    # Handle ID column logic - common case: ID exists in test but not train
    train_has_id = Config.ID_COLUMN in train_df.columns
    test_has_id = Config.ID_COLUMN in test_df.columns
    
    if not train_has_id and not test_has_id:
        # Neither has ID - create for both
        logger.info("No ID column found in either file, creating index-based ID columns")
        train_df.insert(0, 'ID', range(len(train_df)))
        test_df.insert(0, 'ID', range(len(train_df), len(train_df) + len(test_df)))
        Config.ID_COLUMN = 'ID'
    elif not train_has_id and test_has_id:
        # Test has ID, train doesn't (common competition setup)
        logger.info(f"ID column '{Config.ID_COLUMN}' found in test file but not train file")
        logger.info("This is typical for ML competitions - train file doesn't need ID column")
        # Don't create ID for train file - we only need it for final submission
    elif train_has_id and not test_has_id:
        # Train has ID, test doesn't - unusual but handle it
        logger.info(f"ID column '{Config.ID_COLUMN}' found in train file but not test file")
        logger.info("Creating ID column for test file")
        test_df.insert(0, Config.ID_COLUMN, range(len(test_df)))
    else:
        # Both have ID columns
        logger.info(f"ID column '{Config.ID_COLUMN}' found in both files")
    
    # If train doesn't have ID but test does, we need to handle this in feature selection
    if not train_has_id and test_has_id:
        # We'll exclude ID from features when training, but include it in final output
        pass
    
    # Check feature alignment
    train_features = set(train_df.columns) - set(Config.TARGET_COLUMNS + [Config.ID_COLUMN])
    test_features = set(test_df.columns) - set([Config.ID_COLUMN])
    
    if train_features != test_features:
        missing_in_test = train_features - test_features
        extra_in_test = test_features - train_features
        if missing_in_test:
            issues.append(f"Features in train but not test: {missing_in_test}")
        if extra_in_test:
            issues.append(f"Features in test but not train: {extra_in_test}")
    
    # Check for excessive missing values
    train_missing = train_df.isnull().sum()
    high_missing_cols = train_missing[train_missing > len(train_df) * 0.8].index.tolist()
    if high_missing_cols:
        logger.warning(f"Columns with >80% missing values: {high_missing_cols}")
    
    # Check target distribution
    for target in Config.TARGET_COLUMNS:
        if target in train_df.columns:
            target_data = train_df[target].dropna()
            if len(target_data) == 0:
                issues.append(f"Target {target} has no valid values")
            elif target_data.std() == 0:
                logger.warning(f"Target {target} has zero variance")
    
    if issues:
        raise ValueError(f"Data validation failed: {'; '.join(issues)}")
    
    logger.info("Data validation passed")
    return True, train_df, test_df

# ------------------------------ Feature Engineering ------------------------------
def engineer_features(train_df, test_df, logger):
    """Apply feature engineering to both training and test sets"""
    logger.info("Engineering features...")
    
    # Handle feature columns carefully - ID might only be in test file
    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)
    
    # Get feature columns from train (exclude targets and ID if present)
    train_feature_columns = [col for col in train_df.columns 
                            if col not in Config.TARGET_COLUMNS + [Config.ID_COLUMN]]
    
    # For test features, exclude ID if present
    test_feature_columns = [col for col in test_df.columns 
                           if col != Config.ID_COLUMN]
    
    # Use intersection of features that exist in both files
    common_features = list(set(train_feature_columns) & set(test_feature_columns))
    
    logger.info(f"Train features: {len(train_feature_columns)}, Test features: {len(test_feature_columns)}")
    logger.info(f"Common features for training: {len(common_features)}")
    
    # Create copies to avoid modifying original data
    train_engineered = train_df.copy()
    test_engineered = test_df.copy()
    
    # Example feature engineering (customize based on your data)
    numeric_features = train_engineered[common_features].select_dtypes(include=[np.number]).columns
    
    if len(numeric_features) > 1:
        # Add interaction features for top correlated features
        corr_matrix = train_engineered[numeric_features].corr().abs()
        
        # Find highly correlated feature pairs (but not perfectly correlated)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if 0.7 <= corr_matrix.iloc[i, j] <= 0.95:  # Avoid multicollinearity
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Create interaction features for top pairs (limit to avoid explosion)
        for feat1, feat2 in high_corr_pairs[:5]:  # Limit to top 5 pairs
            interaction_name = f'{feat1}_x_{feat2}'
            train_engineered[interaction_name] = train_engineered[feat1] * train_engineered[feat2]
            test_engineered[interaction_name] = test_engineered[feat1] * test_engineered[feat2]
            logger.info(f"Created interaction feature: {interaction_name}")
    
    logger.info("Feature engineering completed")
    return train_engineered, test_engineered, common_features

# ------------------------------ Model Training ------------------------------
def train_single_target(train_data, test_data, target, feature_columns, logger):
    """Train model for a single target with enhanced configuration"""
    logger.info(f"Training model for {target}...")
    
    # Prepare training data for this target
    target_train_data = train_data[feature_columns + [target]].dropna()
    
    if len(target_train_data) == 0:
        raise ValueError(f"No valid training data for target {target}")
    
    logger.info(f"Training samples for {target}: {len(target_train_data)}")
    
    # Create model directory
    model_path = None
    if Config.SAVE_MODELS:
        model_dir = Path(Config.MODEL_DIR) / target
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir)
    
    # Initialize predictor with enhanced configuration
    predictor = TabularPredictor(
        label=target,
        problem_type='regression',
        eval_metric=Config.EVAL_METRIC,
        path=model_path,
        verbosity=1
    )
    
    # Train with cross-validation and enhanced settings
    predictor = predictor.fit(
        train_data=target_train_data,
        presets=Config.PRESETS,
        time_limit=Config.TIME_LIMIT,
        num_cpus='auto',
        num_gpus='auto' if os.environ.get('CUDA_VISIBLE_DEVICES') else 0,
        # Enhanced hyperparameter search
        hyperparameters={
            'GBM': [
                {'num_boost_round': 10000, 'learning_rate': 0.05},
                {'num_boost_round': 5000, 'learning_rate': 0.1}
            ],
            'CAT': [
                {'iterations': 10000, 'learning_rate': 0.05},
                {'iterations': 5000, 'learning_rate': 0.1}
            ],
            'XGB': [
                {'n_estimators': 10000, 'learning_rate': 0.05},
                {'n_estimators': 5000, 'learning_rate': 0.1}
            ],
            'RF': [
                {'n_estimators': 300, 'max_features': 'sqrt'},
                {'n_estimators': 500, 'max_features': 'log2'}
            ]
        },
        num_bag_folds=Config.NUM_FOLDS,
        auto_stack=True,
    )
    
    # Get model performance summary
    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.index[0]
    best_score = leaderboard.loc[best_model, 'score_val']
    
    logger.info(f"Best model for {target}: {best_model} (score: {best_score:.4f})")
    
    # Make predictions
    try:
        predictions = predictor.predict(test_data[feature_columns])
        logger.info(f"Predictions generated for {target}")
        
        # Log prediction statistics
        logger.info(f"Prediction stats for {target}: "
                   f"mean={predictions.mean():.4f}, "
                   f"std={predictions.std():.4f}, "
                   f"min={predictions.min():.4f}, "
                   f"max={predictions.max():.4f}")
        
        return predictions, best_score, best_model
        
    except Exception as e:
        logger.error(f"Prediction failed for {target}: {str(e)}")
        raise

# ------------------------------ Main Execution ------------------------------
def main():
    """Main execution function with comprehensive error handling"""
    logger = setup_logging()
    logger.info("Starting AutoGluon multi-target regression pipeline")
    
    try:
        # Load data with error handling
        logger.info("Loading data...")
        if not os.path.exists(Config.TRAIN_PATH):
            raise FileNotFoundError(f"Training file not found: {Config.TRAIN_PATH}")
        if not os.path.exists(Config.TEST_PATH):
            raise FileNotFoundError(f"Test file not found: {Config.TEST_PATH}")
            
        train_df = pd.read_csv(Config.TRAIN_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)
        
        logger.info(f"Data loaded - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Validate data
        _, train_df, test_df = validate_data(train_df, test_df, logger)
        
        # Feature engineering
        train_df, test_df, feature_columns = engineer_features(train_df, test_df, logger)
        
        # Add any new engineered features to the feature list
        final_feature_columns = [col for col in train_df.columns 
                                if col not in Config.TARGET_COLUMNS + [Config.ID_COLUMN]]
        
        logger.info(f"Using {len(final_feature_columns)} features for training")
        
        # Storage for predictions and metadata
        all_predictions = pd.DataFrame()
        model_performance = {}
        
        # Train models for each target
        for i, target in enumerate(Config.TARGET_COLUMNS, 1):
            logger.info(f"Processing target {i}/{len(Config.TARGET_COLUMNS)}: {target}")
            
            try:
                predictions, score, best_model = train_single_target(
                    train_df, test_df, target, final_feature_columns, logger
                )
                
                all_predictions[target] = predictions
                model_performance[target] = {
                    'score': score,
                    'best_model': best_model
                }
                
            except Exception as e:
                logger.error(f"Failed to train model for {target}: {str(e)}")
                # Create dummy predictions to maintain structure
                all_predictions[target] = np.zeros(len(test_df))
                model_performance[target] = {'score': np.nan, 'best_model': 'FAILED'}
        
        # Create submission file
        logger.info("Creating submission file...")
        
        # Handle ID column for submission - it should be from test file
        if Config.ID_COLUMN in test_df.columns:
            submission = pd.concat([test_df[Config.ID_COLUMN], all_predictions], axis=1)
        else:
            # Fallback: create ID column if somehow missing
            logger.warning("ID column missing from test data, creating index-based IDs")
            id_series = pd.Series(range(len(test_df)), name=Config.ID_COLUMN)
            submission = pd.concat([id_series, all_predictions], axis=1)
        
        # Ensure column order: ID first, then targets
        submission = submission[[Config.ID_COLUMN] + Config.TARGET_COLUMNS]
        
        # Ensure output directory exists
        output_path = Path(Config.OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        submission.to_csv(Config.OUTPUT_PATH, index=False)
        
        # Save performance summary
        performance_df = pd.DataFrame(model_performance).T
        performance_path = output_path.parent / 'model_performance.csv'
        performance_df.to_csv(performance_path)
        
        # Log final summary
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Submission saved to: {Config.OUTPUT_PATH}")
        logger.info(f"Performance summary saved to: {performance_path}")
        
        # Print performance summary
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        for target, perf in model_performance.items():
            print(f"{target}: {perf['best_model']} (Score: {perf['score']:.4f})")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()