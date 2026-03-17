import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)

SEED = 42
np.random.seed(SEED)
n = 5000

df = pd.DataFrame({
    "engine_temp":  np.random.normal(90, 8, n),
    "vibration":    np.random.normal(5, 1.5, n),
    "oil_pressure": np.random.normal(40, 5, n),
    "rpm":          np.random.normal(2000, 300, n) 
    # rpm is intentionally excluded from risk score to test model's ability to identify actually relevant things
})


# each condition contributes risk
risk_score = (
    0.30 * np.clip((df["engine_temp"]  - 95) / 10, 0, 1) +
    0.25 * np.clip((df["vibration"]    - 6)  / 2,  0, 1) +
    0.25 * np.clip((35 - df["oil_pressure"]) / 5,  0, 1)
)
df["failure"] = (np.random.rand(n) < risk_score).astype(int)

print(f"Failure rate: {df['failure'].mean():.1%}\n")
print("Feature correlations with failure:")
print(df.corr(numeric_only=True)["failure"].sort_values(ascending=False).to_string(), "\n")

features = ["engine_temp", "vibration", "oil_pressure", "rpm"]
X = df[features]
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y 
    # preserve failure ratio
)

# Scale features — critical for logistic regression.
# Scaler is fit only on training data to prevent leakage into the test set.
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# balanced class weight compensates for the fact that failures are not as common as normal readings
model = LogisticRegression(random_state=SEED, max_iter=1000, class_weight="balanced")
model.fit(X_train_sc, y_train)

# cross-validation using a Pipeline so the scaler is re-fit independently on
# each CV training fold — prevents leakage from validation folds into scaling.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(random_state=SEED, max_iter=1000, class_weight="balanced"))
])
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
print(f"5-fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")

preds = model.predict(X_test_sc)
print("Test accuracy:", accuracy_score(y_test, preds))
print("\nClassification report:")
print(classification_report(y_test, preds, target_names=["Normal", "Failure"], zero_division=0))

# coefficients show which sensors matter most
coef_df = pd.DataFrame({
    "feature":     features,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)
print("\nModel coefficients (standardized):")
print(coef_df.to_string(index=False))

# compute scaled full dataset
X_sc = scaler.transform(X)

# plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# scatter; plot normal points first, failures on top so they aren't swamped
ax = axes[0]
for label, color, name in [(0, "steelblue", "Normal"), (1, "crimson", "Failure")]:
    mask = df["failure"] == label
    ax.scatter(df.loc[mask, "engine_temp"], df.loc[mask, "vibration"],
               c=color, alpha=0.4, s=10, linewidths=0, label=name)
ax.set_xlabel("Engine Temperature (°C)")
ax.set_ylabel("Vibration (mm/s)")
ax.set_title("Sensor Readings — Failure Events")
ax.legend()

# confusion matrix: actual normal vs predicted normal, actual failure vs predicted failure
ax2 = axes[1]
ConfusionMatrixDisplay(confusion_matrix(y_test, preds),
                       display_labels=["Normal", "Failure"]).plot(ax=ax2, colorbar=False)
ax2.set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("fault_detection.png", dpi=150, bbox_inches="tight")
plt.show()

# export for julia
df["predicted_failure"]   = model.predict(X_sc)
df["failure_probability"] = model.predict_proba(X_sc)[:, 1]

df.to_csv("sensor_data.csv", index=False)
coef_df.to_csv("model_coefficients.csv", index=False)

print("Exported sensor_data.csv and model_coefficients.csv")