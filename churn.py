import pandas as pd

df = pd.read_excel('E_Commerce_Dataset.xlsx', sheet_name='E Comm')
print(df.shape)
print(df.head())

# Drop ID column (not predictive)
df = df.drop(columns=['CustomerID'])

# Impute missing numeric values with median
num_cols_with_na = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
                    'OrderAmountHikeFromlastYear', 'CouponUsed',
                    'OrderCount', 'DaySinceLastOrder']
for col in num_cols_with_na:
    df[col] = df[col].fillna(df[col].median())

# Remove exact duplicate rows. Without this, identical customers can land in both the
# train and test sets, and the Random Forest scores higher than it really should.
rows_before = len(df)
df = df.drop_duplicates()
print(f"Removed {rows_before - len(df)} duplicate rows ({rows_before} -> {len(df)})")

# Confirm no missing values remain
print("Missing values after cleaning:", df.isnull().sum().sum())
print("Shape after cleaning:", df.shape)
print("Churn distribution:")
print(df['Churn'].value_counts(normalize=True))

# Churn rate by categorical features
categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                    'PreferedOrderCat', 'MaritalStatus']

for col in categorical_cols:
    print(f"\nChurn rate by {col}:")
    print(df.groupby(col)['Churn'].mean().sort_values(ascending=False))

# Average values of key numeric features by churn status
numeric_cols = ['Tenure', 'SatisfactionScore', 'Complain', 'DaySinceLastOrder',
                'CashbackAmount', 'OrderCount']

print("\nAverage values by churn status:")
print(df.groupby('Churn')[numeric_cols].mean())

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nShape after encoding:", df_encoded.shape)
print(df_encoded.columns.tolist())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Separate features and target
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# Train/test split (80/20), stratify to preserve churn ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train churn rate:", y_train.mean(), "Test churn rate:", y_test.mean())

# Baseline model: Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Scale features (fixes convergence warning, helps logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain logistic regression on scaled data
log_reg_scaled = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = log_reg_scaled.predict(X_test_scaled)
y_proba_scaled = log_reg_scaled.predict_proba(X_test_scaled)[:, 1]

print("\n--- Logistic Regression (scaled) ---")
print(classification_report(y_test, y_pred_scaled))
print("ROC AUC:", roc_auc_score(y_test, y_proba_scaled))

# Random Forest (doesn't need scaling, but we'll use original features)
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))

import pandas as pd

# Feature importance from Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 most important features:")
print(importances.head(15))

import joblib

joblib.dump(rf, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # in case you want to use it later
print("\nModel saved as churn_model.pkl")

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create explainer for the Random Forest model
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

print("Type of shap_values:", type(shap_values))
if isinstance(shap_values, list):
    print("It's a list with", len(shap_values), "elements")
    shap_values_churn = shap_values[1]
else:
    print("Shape:", shap_values.shape)
    # New SHAP versions return shape (n_samples, n_features, n_classes)
    # We want class index 1 (churn) -> slice the last dimension
    shap_values_churn = shap_values[:, :, 1]

print("Final shap_values_churn shape:", shap_values_churn.shape)

# Summary plot
shap.summary_plot(shap_values_churn, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)
print("Saved shap_summary.png")

# Build a results table: actual churn, predicted probability, and top SHAP reasons
results = X_test.copy()
results['Actual_Churn'] = y_test.values
results['Churn_Probability'] = y_proba_rf

# Sort by churn probability, descending — most at-risk first
top_at_risk = results.sort_values('Churn_Probability', ascending=False).head(20)

print("\n--- Top 20 At-Risk Customers ---")
print(top_at_risk[['Actual_Churn', 'Churn_Probability']])

# For each of these customers, find their top 3 SHAP reasons (why the model flagged them)
top_at_risk_idx = top_at_risk.index
# Get positional indices for SHAP array lookup
positions = [X_test.index.get_loc(i) for i in top_at_risk_idx]

print("\n--- Top Reasons for Each At-Risk Customer ---")
for pos, idx in zip(positions, top_at_risk_idx):
    customer_shap = shap_values_churn[pos]
    feature_impact = pd.Series(customer_shap, index=X_test.columns).sort_values(ascending=False)
    top_reasons = feature_impact.head(3)
    print(f"\nCustomer (test row {idx}) — Churn Probability: {results.loc[idx, 'Churn_Probability']:.2f}, Actual: {results.loc[idx, 'Actual_Churn']}")
    for feat, val in top_reasons.items():
        actual_val = X_test.loc[idx, feat]
        print(f"   {feat} = {actual_val} (pushed churn risk up by {val:.3f})")

# Save the full at-risk list to a CSV for your portfolio/README
top_at_risk.to_csv('top_20_at_risk_customers.csv')
print("\nSaved top_20_at_risk_customers.csv")

# Build a long-format SHAP reasons table for all at-risk customers
# (used by the dashboard's "why is this customer at risk" chart and what-if sliders)
reason_rows = []
for pos, idx in zip(positions, top_at_risk_idx):
    customer_shap = shap_values_churn[pos]
    for feat, val in zip(X_test.columns, customer_shap):
        reason_rows.append({
            'customer_id': idx,
            'feature': feat,
            'shap_value': val,
            'feature_value': X_test.loc[idx, feat]
        })

shap_reasons_df = pd.DataFrame(reason_rows)
shap_reasons_df.to_csv('shap_reasons.csv', index=False)
print("Saved shap_reasons.csv")