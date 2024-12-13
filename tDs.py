#IMPORTS
import pandas as pd
import numpy as np

# Data Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Evaluation Metrics
from sklearn.metrics import precision_score, f1_score, classification_report, confusion_matrix

# Visualization (Optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('customer_interactions.csv')

# Display the first few rows
print(data.head())

# Get basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())


# Separate features and target
X = data.drop(['customer_id', 'high_potential'], axis=1)
y = data['high_potential']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Impute numerical features with median
numerical_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])

# Impute categorical features with mode
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])


# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical features
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# If you prefer one-hot encoding:
# X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform numerical features
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Training samples: {X_train.shape[0]}')
print(f'Testing samples: {X_test.shape[0]}')


# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    random_state=42
)

# Initialize Logistic Regression
lr_classifier = LogisticRegression(
    solver='lbfgs', 
    max_iter=1000, 
    random_state=42
)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to evaluate models
def evaluate_model(model, X, y, cv):
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    return precision.mean(), f1.mean()

# Evaluate Random Forest
rf_precision, rf_f1 = evaluate_model(rf_classifier, X_train, y_train, cv)
print(f'Random Forest - Precision: {rf_precision:.2f}, F1-Score: {rf_f1:.2f}')

# Evaluate Logistic Regression
lr_precision, lr_f1 = evaluate_model(lr_classifier, X_train, y_train, cv)
print(f'Logistic Regression - Precision: {lr_precision:.2f}, F1-Score: {lr_f1:.2f}')

#Hyperparamater Tuning
from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best F1-Score: {grid_search.best_score_:.2f}')

# Retrieve the best model
best_rf = grid_search.best_estimator_

# Train on the full training set
best_rf.fit(X_train, y_train)

# Predict on test data
y_pred = best_rf.predict(X_test)

# Calculate Precision and F1-Score
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f'Test Precision: {test_precision:.2f}')
print(f'Test F1-Score: {test_f1:.2f}')

# Detailed Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Get feature importances
importances = best_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()





