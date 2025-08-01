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


import pandas as pd
from autogluon.tabular import TabularPredictor

# ------------------------------ Config ------------------------------
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
OUTPUT_PATH = 'submission.csv'

TARGET_COLUMNS = [f'BlendProperty{i}' for i in range(1, 11)]
ID_COLUMN = 'ID'

# ------------------------------ Load Data ------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ------------------------------ Prepare Data ------------------------------
# Input features = all columns except target columns and ID
feature_columns = [col for col in train_df.columns if col not in TARGET_COLUMNS + [ID_COLUMN]]

# Storage for all predictions
all_predictions = pd.DataFrame()

# ------------------------------ Loop over each target ------------------------------
for target in TARGET_COLUMNS:
    print(f"🚀 Training for {target}...")
    predictor = TabularPredictor(
        label=target,
        problem_type='regression',
        eval_metric='mean_absolute_percentage_error'
    ).fit(
        train_data=train_df[feature_columns + [target]],
        presets='best_quality',
        time_limit=3600  # adjust time limit as needed per target
    )
    
    # Predict for this target
    y_pred = predictor.predict(test_df[feature_columns])
    all_predictions[target] = y_pred

# ------------------------------ Attach ID and Save ------------------------------
submission = pd.concat([test_df[ID_COLUMN], all_predictions], axis=1)
submission = submission[[ID_COLUMN] + TARGET_COLUMNS]
submission.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Submission saved to {OUTPUT_PATH}")
