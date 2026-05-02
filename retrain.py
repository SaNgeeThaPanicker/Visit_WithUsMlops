import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
api = HfApi()

# Load data
train_df = pd.read_csv("hf://datasets/SANGU19/tourism-dataset/train.csv", index_col=0)
test_df  = pd.read_csv("hf://datasets/SANGU19/tourism-dataset/test.csv", index_col=0)

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]
X_test  = test_df.drop(columns=["ProdTaken"])
y_test  = test_df["ProdTaken"]

# Train RandomForest
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators":[100,200],"max_depth":[4,6],"min_samples_split":[2,5]},
    cv=5, scoring="f1", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_model = rf_grid.best_estimator_

y_pred = best_model.predict(X_test)
print(f"Best params: {rf_grid.best_params_}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Save with joblib
os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(best_model, "tourism_project/model_building/best_model.joblib")
print("Model saved")

# Upload to HF
api.upload_file(
    path_or_fileobj="tourism_project/model_building/best_model.joblib",
    path_in_repo="best_model.joblib",
    repo_id="SANGU19/tourism-model",
    repo_type="model"
)
print("Model uploaded successfully")