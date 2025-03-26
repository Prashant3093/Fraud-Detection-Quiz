import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import time
import numpy as np
from scipy.sparse import save_npz, csr_matrix
import joblib

# Define output directory
output_dir = r"C:\Users\prash\OneDrive\Documents\Project\Backend\outputs"
os.makedirs(output_dir, exist_ok=True)

# Start timer
start_time = time.time()

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv(r"C:\Users\prash\OneDrive\Documents\Project\Backend\fraudTrain.csv")
test_df = pd.read_csv(r"C:\Users\prash\OneDrive\Documents\Project\Backend\fraudTest.csv")

# Drop unnecessary identifier columns
drop_columns = ["first", "last", "street", "zip", "trans_num"]
train_df.drop(columns=drop_columns, inplace=True)
test_df.drop(columns=drop_columns, inplace=True)

# Extract datetime features
for df in [train_df, test_df]:
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df.drop(columns=["trans_date_trans_time"], inplace=True)

# Separate features and target
X_train = train_df.drop(columns=["is_fraud"])
y_train = train_df["is_fraud"]
X_test = test_df.drop(columns=["is_fraud"])
y_test = test_df["is_fraud"]

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define preprocessing steps
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Fit and transform data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Convert to sparse format
X_train_preprocessed = csr_matrix(X_train_preprocessed)
X_test_preprocessed = csr_matrix(X_test_preprocessed)

# Save preprocessed data
save_npz(os.path.join(output_dir, "X_train.npz"), X_train_preprocessed)
save_npz(os.path.join(output_dir, "X_test.npz"), X_test_preprocessed)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

# Save the fitted preprocessor
joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))

# Print completion message
elapsed_time = time.time() - start_time
print(f"Preprocessing completed in {elapsed_time:.2f} seconds. Processed data saved to {output_dir}")
