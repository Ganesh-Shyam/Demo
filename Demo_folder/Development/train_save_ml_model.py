import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Use the loaded dataframe from upstream
print("Training data:")
print(loaded_df)
print()

# Prepare features and target
# Use 'id' as feature to predict 'age'
X = loaded_df[['id']]
y = loaded_df['age']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple Linear Regression model
ml_model = LinearRegression()
ml_model.fit(X_train, y_train)

print("✓ Model trained successfully!")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print()

# Evaluate model
y_pred = ml_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"  - Mean Squared Error: {mse:.2f}")
print(f"  - R² Score: {r2:.3f}")
print()

# Create model directory
os.makedirs('model', exist_ok=True)

# Save model as pickle file
model_path = 'model/ml_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(ml_model, f)

print(f"✓ Model saved to: {model_path}")
