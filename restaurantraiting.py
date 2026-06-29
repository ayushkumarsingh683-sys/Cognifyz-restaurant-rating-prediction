import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOAD YOUR ACTUAL DATASET
# ==========================================
try:
    # Changed to match your file name 'Dataset.csv'
    df = pd.read_csv('Dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'Dataset.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- FIXED COLUMNS TO MATCH YOUR CSV EXACTLY ---
feature_cols = ['Average Cost for two', 'Votes', 'Has Table booking', 'Cuisines']
target_col = 'Aggregate rating'

# Verify columns exist in the dataframe
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    print(f"Error: The following expected columns are missing from your CSV: {missing_cols}")
    print(f"Available columns in your CSV are: {df.columns.tolist()}")
    exit()

# Separate features (X) and target (y)
X = df[feature_cols]
y = df[target_col]


# ==========================================
# 2. PREPROCESSING & PIPELINE SETUP
# ==========================================
# Updated variable names to match your exact CSV columns
numerical_features = ['Average Cost for two', 'Votes']
categorical_features = ['Has Table booking', 'Cuisines']

# Numerical Pipeline: Fill missing values with median
numerical_transformer = SimpleImputer(strategy='median')

# Categorical Pipeline: Fill missing values with most frequent, then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Bundle preprocessing transformers together
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Final end-to-end Machine Learning Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# ==========================================
# 3. TRAIN-TEST SPLIT AND TRAINING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Random Forest model... (this may take a moment)")
model_pipeline.fit(X_train, y_train)
print("--- Model Training Complete ---\n")


# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== MODEL PERFORMANCE METRICS ===")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2) Score      : {r2:.4f}\n")


# ==========================================
# 5. FEATURE IMPORTANCE & INTERPRETATION
# ==========================================
encoded_cat_features = model_pipeline.named_steps['preprocessor'] \
                                     .transformers_[1][1] \
                                     .named_steps['onehot'] \
                                     .get_feature_names_out(categorical_features)

all_features = numerical_features + list(encoded_cat_features)
importances = model_pipeline.named_steps['regressor'].feature_importances_

feature_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

print("=== TOP 10 MOST INFLUENTIAL FEATURES ===")
print(feature_imp_df.to_string(index=False))

# Plotting the results visually
plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='teal')
plt.xlabel('Importance Score')
plt.title('Top Features Influencing Restaurant Ratings')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()