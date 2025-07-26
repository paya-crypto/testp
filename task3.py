# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 2. Load the dataset
df = pd.read_csv("C:\\Users\jay30\OneDrive\Documents\myprojects\python\creditcard.csv")

# 3. Preprocessing
df['normalizedAmount'] = StandardScaler().fit_transform(df[['Amount']])
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 4. Feature matrix and label
X = df.drop('Class', axis=1)
y = df['Class']

# 5. Anomaly Detection (optional analysis)
print("\n🔍 Applying Isolation Forest...")
iso_forest = IsolationForest(contamination=0.01, random_state=42)
y_pred_if = iso_forest.fit_predict(X)
print(classification_report(y, y_pred_if == -1))

print("\n🔍 Applying Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
y_pred_lof = lof.fit_predict(X)
print(classification_report(y, y_pred_lof == -1))

# 6. Balance the dataset using SMOTE
print("\n⚖ Applying SMOTE...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 7. Stratified Split to prevent bias
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in sss.split(X_res, y_res):
    X_train, X_test = X_res.iloc[train_idx], X_res.iloc[test_idx]
    y_train, y_test = y_res.iloc[train_idx], y_res.iloc[test_idx]

# 8. Train XGBoost Classifier
print("\n🚀 Training XGBoost...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate the model
y_pred = model.predict(X_test)
print("\n📊 Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# 10. Check for overfitting
print("\n✅ Model Accuracy:")
print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy : {model.score(X_test, y_test):.4f}")

# 11. ROC Curve
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='XGBoost ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



import joblib

# Save the trained model
joblib.dump(model, "xgb_model_2.pkl")
print("✅ Model saved as xgb_model_2.pkl")