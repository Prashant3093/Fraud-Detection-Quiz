import os
import time
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Define input directory
input_dir = r"C:\Users\prash\OneDrive\Documents\Project\Backend\outputs"

# Load preprocessed data
X_train = load_npz(os.path.join(input_dir, "X_train.npz"))
X_test = load_npz(os.path.join(input_dir, "X_test.npz"))
y_train = np.load(os.path.join(input_dir, "y_train.npy"))
y_test = np.load(os.path.join(input_dir, "y_test.npy"))

# Define model
model = RandomForestClassifier(
    n_estimators=100,  
    random_state=42,  
    class_weight="balanced",  
    n_jobs=-1,  
    max_depth=10  
)

# Train model
start_time = time.time()
model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC-AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Save trained model
joblib.dump(model, os.path.join(input_dir, "model.pkl"))
print(f"Trained model saved to {input_dir}")
