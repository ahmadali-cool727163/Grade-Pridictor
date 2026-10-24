"""Train and save the student grade prediction model.

Run once:  python train_model.py
Creates:   student_grade_model.joblib
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("student-mat.csv", sep=";")
X = df.drop(columns=["G3"])
y = df["G3"]

cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

candidates = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_name, best_score, best_pipe = None, -1e9, None
for name, reg in candidates.items():
    pipe = Pipeline([("prep", prep), ("model", reg)])
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    print(f"{name:18s} CV R2 = {cv:.3f}")
    if cv > best_score:
        best_name, best_score, best_pipe = name, cv, pipe

best_pipe.fit(X_train, y_train)
pred = best_pipe.predict(X_test)
print(f"\nBest model: {best_name}")
print(f"Test R2  : {r2_score(y_test, pred):.3f}")
print(f"Test MAE : {mean_absolute_error(y_test, pred):.2f} grade points")

joblib.dump(best_pipe, "student_grade_model.joblib")
print("Saved student_grade_model.joblib")
