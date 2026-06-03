#!/usr/bin/env python3
"""
Employee Attrition Prediction System - macOS Version
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Use non-interactive backend for macOS
import matplotlib
matplotlib.use('Agg')

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("=" * 60)
print("  EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("=" * 60)

# Load dataset
df = pd.read_csv('IBM_Employee_Attrition.csv')
print(f"\n[✓] Dataset loaded: {df.shape[0]} employees, {df.shape[1]} features")

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================
# Drop redundant columns
df_clean = df.drop(['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'], axis=1, errors='ignore')

# Convert target to binary
df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
cat_cols = df_clean.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df_clean[col] = le.fit_transform(df_clean[col])

# Split features and target
X = df_clean.drop(['Attrition'], axis=1)
y = df_clean['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================
print("\n[✓] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# =============================================================================
# 4. MODEL EVALUATION
# =============================================================================
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print("  MODEL RESULTS")
print("=" * 60)
print(f"\n  Accuracy: {accuracy*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(f"  [[TN={cm[0][0]:3d}  FP={cm[0][1]:3d}]")
print(f"   [FN={cm[1][0]:3d}  TP={cm[1][1]:3d}]]")

# =============================================================================
# 5. FEATURE IMPORTANCE (KEY INSIGHTS)
# =============================================================================
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(f"\n{'='*60}")
print("  TOP 10 FACTORS DRIVING ATTRITION")
print("=" * 60)

for i, row in feature_imp.head(10).iterrows():
    print(f"  {feature_imp.head(10).index.get_loc(i)+1}. {row['Feature']:<25} (Importance: {row['Importance']:.4f})")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Attrition Distribution
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.countplot(x='Attrition', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Employee Attrition Distribution')
plt.xlabel('Attrition Status')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('attrition_distribution.png', dpi=150)
print("\n[✓] Chart saved: attrition_distribution.png")

# Plot 2: Feature Importance
fig2, ax2 = plt.subplots(figsize=(10, 8))
top_feat = feature_imp.head(15)
sns.barplot(x='Importance', y='Feature', data=top_feat, palette='coolwarm')
plt.title('Top 15 Key Factors Affecting Attrition')
plt.xlabel('Importance Score')
plt.ylabel('Factor')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("[✓] Chart saved: feature_importance.png")

# Plot 3: Confusion Matrix
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("[✓] Chart saved: confusion_matrix.png")

print(f"\n{'='*60}")
print("  SYSTEM COMPLETE - Check generated PNG files")
print("=" * 60)