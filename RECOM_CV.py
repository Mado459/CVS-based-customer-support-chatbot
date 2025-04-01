import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('Cardio1.csv', sep=';') # Assuming your CSV uses semicolon as delimiter
# Example of derived cardiovascular risk label based on thresholds
def compute_risk(row):
    risk_score = 0
    
    # Adding points for high blood pressure
    if row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
        risk_score += 2
    elif 130 <= row['ap_hi'] < 140 or 80 <= row['ap_lo'] < 90:
        risk_score += 1
    
    # Adding points for cholesterol levels
    if row['cholesterol'] == 2:
        risk_score += 1
    elif row['cholesterol'] == 3:
        risk_score += 2
    
    # Adding points for glucose levels
    if row['gluc'] == 2:
        risk_score += 1
    elif row['gluc'] == 3:
        risk_score += 2
    
    # Adding points for age (older = higher risk)
    if row['age'] > 50:
        risk_score += 1
    if row['age'] > 60:
        risk_score += 2
    
    return risk_score

# Apply the function to create a new 'risk_score' column
data['risk_score'] = data.apply(compute_risk, axis=1)
# Apply the function to create a new 'risk_score' column
data['risk_score'] = data.apply(compute_risk, axis=1)


# Categorizing risk score into binary high/low risk
def categorize_risk(risk_score):
    return 1 if risk_score >= 3 else 0  # High risk if score >= 3

data['target_column'] = data['risk_score'].apply(categorize_risk)

# (Optional) You can also use multiclass risk categorization
# (Optional) You can also use multiclass risk categorization
def categorize_risk_multiclass(risk_score):
    if risk_score <= 2:
        return 'low'
    elif 3 <= risk_score <= 4:
        return 'moderate'
    else:
        return 'high'

data['target_column'] = data['risk_score'].apply(categorize_risk_multiclass)




# Data Preprocessing
# Fill missing values if any
data.fillna(method='ffill', inplace=True)

# Separate features and target
X = data.drop(['target_column', 'risk_score'], axis=1)  # 'risk_score' is not needed for training
y = data['target_column']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print(classification_report(y_test, y_pred))

# Example personalized recommendation based on predicted risk
def recommend_lifestyle_changes(risk_score):
    if risk_score > 0.7:
        return "Recommend regular exercise, diet control, and medical check-ups."
    elif 0.4 < risk_score <= 0.7:
        return "Suggest moderate physical activity and balanced diet."
    else:
        return "Maintain current healthy lifestyle."

# Generate personalized recommendations
for i in range(5):  # Show first 5 recommendations
    risk_score = model.predict_proba([X_test[i]])[0][1]  # Probability of class 1 (high risk)
    print(f"Patient {i+1} - Risk Score: {risk_score:.2f}")
    print(recommend_lifestyle_changes(risk_score))
    print("---")

# Optimised model 1


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint
import shap

# Load the dataset
data = pd.read_csv('Cardio1.csv', sep=';')  # Assuming your CSV uses semicolon as delimiter

# Feature Engineering: Derived cardiovascular risk label based on thresholds
def compute_risk(row):
    risk_score = 0
    # Adding points for high blood pressure
    if row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
        risk_score += 2
    elif 130 <= row['ap_hi'] < 140 or 80 <= row['ap_lo'] < 90:
        risk_score += 1
    # Adding points for cholesterol levels
    if row['cholesterol'] == 2:
        risk_score += 1
    elif row['cholesterol'] == 3:
        risk_score += 2
    # Adding points for glucose levels
    if row['gluc'] == 2:
        risk_score += 1
    elif row['gluc'] == 3:
        risk_score += 2
    # Adding points for age (older = higher risk)
    if row['age'] > 50:
        risk_score += 1
    if row['age'] > 60:
        risk_score += 2
    return risk_score

# Apply the function to create a new 'risk_score' column
data['risk_score'] = data.apply(compute_risk, axis=1)

# Multi-class risk categorization for higher resolution predictions
def categorize_risk_multiclass(risk_score):
    if risk_score <= 2:
        return 'low'
    elif 3 <= risk_score <= 4:
        return 'moderate'
    else:
        return 'high'

data['target_column'] = data['risk_score'].apply(categorize_risk_multiclass)

# Data Preprocessing
# Fill missing values if any
data.fillna(method='ffill', inplace=True)

# Convert categorical variables using one-hot encoding
categorical_cols = ['cholesterol', 'gluc']  # Add more if necessary
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Separate features and target
X = data.drop(['target_column', 'risk_score'], axis=1)
y = data['target_column']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning for Random Forest using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

# Randomized search with cross-validation
rf_model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=20, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Model Training with the best hyperparameters
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)

# Model Evaluation
print(classification_report(y_test, y_pred))

# Model Explainability using SHAP (Optional but useful)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Plot summary for model explainability
shap.summary_plot(shap_values[1], X_test, feature_names=data.columns)

# Example personalized recommendation based on predicted risk
def recommend_lifestyle_changes(risk_score):
    if risk_score > 0.7:
        return "Recommend regular exercise, diet control, and medical check-ups."
    elif 0.4 < risk_score <= 0.7:
        return "Suggest moderate physical activity and balanced diet."
    else:
        return "Maintain current healthy lifestyle."

# Generate personalized recommendations for the first 5 patients
for i in range(5):
    probas = best_model.predict_proba([X_test[i]])[0]
    risk_score = probas[1]  # Probability of high risk
    print(f"Patient {i+1} - Predicted Risk Score: {risk_score:.2f}")
    print(recommend_lifestyle_changes(risk_score))
    print("---")
