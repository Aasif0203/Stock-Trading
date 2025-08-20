import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime

# -------------------------------
# 1. Download Stock Data (e.g., GOOG)
# -------------------------------
ticker = "ADBE"
start = "2020-01-01"
end = datetime.today().strftime('%Y-%m-%d')

df = yf.download(ticker, start=start, end=end)

# Flatten possible MultiIndex columns
df.columns = df.columns.get_level_values(0)

# Keep OHLC
df = df.loc[:, ["Open", "High", "Low", "Close"]].reset_index()

# Create target: Next day's Close
df["Next_Close"] = df["Close"].shift(-1)
df = df.dropna().reset_index(drop=True)

# -------------------------------
# 2. Features & Preprocessor
# -------------------------------
features = ["Open", "High", "Low", "Close"]
X = df[features]
y = df["Next_Close"]

preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), features)],
    remainder="drop"
)

# -------------------------------
# 3. Candidate ML Models
# -------------------------------
lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.001, max_iter=30000)
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

voting_reg = VotingRegressor(estimators=[("lr", lr), ("rf", rf), ("gb", gb)])
weighted_voting_reg = VotingRegressor(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    weights=[1, 2, 2]
)
stacking_reg = StackingRegressor(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    final_estimator=Ridge(alpha=1.0)
)

models = {
    "Linear Regression": lr,
    "Ridge Regression": ridge,
    "Lasso Regression": lasso,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Voting": voting_reg,
    "Weighted Voting": weighted_voting_reg,
    "Stacking": stacking_reg
}

# -------------------------------
# 4. Evaluate ML Models
# -------------------------------
tscv = TimeSeriesSplit(n_splits=5)
results = {}

for name, model in models.items():
    mae_scores, r2_scores = [], []
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    
    results[name] = {"MAE": np.mean(mae_scores), "R²": np.mean(r2_scores)}

# -------------------------------
# 5. Results
# -------------------------------
results_df = pd.DataFrame(results).T.sort_values(by="MAE", na_position="last")
print("\nModel Performance (Cross-Validation):")
print(results_df)

# -------------------------------
# 6. AutoML: Pick Best ML Model
# -------------------------------
best_model_name = results_df.dropna().index[0]
print(f"\n✅ Best ML Model Selected: {best_model_name}")

best_model = models[best_model_name]
final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
final_pipeline.fit(X, y)

# -------------------------------
# 7. Save Model as {ticker}_{today}.joblib
# -------------------------------
today_str = datetime.today().strftime("%Y%m%d")
model_filename = f"{ticker}_{today_str}.joblib"
joblib.dump(final_pipeline, model_filename)
print(f"💾 Model saved as {model_filename}")

# -------------------------------
# 8. Predict Tomorrow’s Price
# -------------------------------
latest_features = df[features].iloc[-1:]
next_day_prediction = final_pipeline.predict(latest_features)[0]

print(f"\n📈 Predicted next close (ML AutoML) for {ticker} (tomorrow): {next_day_prediction:.2f}")