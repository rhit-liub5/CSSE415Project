import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv("../data/aggregated_rolling_features.csv")

# 2. Add diff features
stats = ["kills", "deaths", "assists", "adr", "kast", "kddiff"]
for stat in stats:
    df[f"{stat}_diff"] = (
        df[f"previous_10_games_team1_average_{stat}"]
        - df[f"previous_10_games_team2_average_{stat}"]
    )

df["map_score_diff"] = (
    df["team1_previous_10_average_map_score"]
    - df["team2_previous_10_average_map_score"]
)

# ---- Lasso Model Training (following Borui_lasso_and_random_forest.ipynb) ----
lasso_features = [
    "kills_diff",
    "deaths_diff",
    "assists_diff",
    "adr_diff",
    "kast_diff",
    "kddiff_diff",
    "map_score_diff"
]

# They sorted and reset index
df_lasso = df.copy()
df_lasso["datetime"] = pd.to_datetime(df_lasso["datetime"])
df_lasso = df_lasso.sort_values("datetime").reset_index(drop=True)
data_lasso = df_lasso[lasso_features + ["team1_win", "datetime"]].dropna()

split_index_lasso = int(len(data_lasso) * 2 / 3)
train_lasso = data_lasso.iloc[:split_index_lasso]

X_train_lasso = train_lasso[lasso_features]
y_train_lasso = train_lasso["team1_win"]

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=5000,
        random_state=42
    ))
])

param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}
print("Training Lasso GridSearch...")
grid_search_lasso = GridSearchCV(
    estimator=lasso_pipe,
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1
)
grid_search_lasso.fit(X_train_lasso, y_train_lasso)
best_lasso = grid_search_lasso.best_estimator_

os.makedirs("models", exist_ok=True)
joblib.dump(best_lasso, "models/best_lasso.pkl")
print("Saved best Lasso model to models/best_lasso.pkl")

# ---- KNN Model Training (following knn_analysis_team.ipynb) ----
knn_features = [f'{stat}_diff' for stat in stats] + ['map_score_diff', 'bestOf']
knn_features = [c for c in knn_features if c in df.columns]

df_knn = df.copy()
df_knn['datetime'] = pd.to_datetime(df_knn['datetime'])
df_knn = df_knn.sort_values('datetime') # They did NOT reset index

X_knn = df_knn[knn_features]
y_knn = df_knn['team1_win']

split_idx_knn = int(len(df_knn) * 0.8)
X_train_knn = X_knn.iloc[:split_idx_knn]
y_train_knn = y_knn.iloc[:split_idx_knn]

# They scaled manually, but for joblib it's easier to put inside a Pipeline
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

print("Training KNN loop...")
best_k = 1
best_acc = 0
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_knn)
X_test_scaled = scaler.transform(X_knn.iloc[split_idx_knn:])
y_test_knn = y_knn.iloc[split_idx_knn:]

for k in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_knn)
    acc = knn.score(X_test_scaled, y_test_knn)
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"Found best K = {best_k}")

best_knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])
best_knn_pipe.fit(X_train_knn, y_train_knn)

joblib.dump(best_knn_pipe, "models/best_knn.pkl")
print("Saved best KNN model to models/best_knn.pkl")
