import pickle
import pandas as pd

# Load the saved model from pickle file
model_path = 'model/ml_model.pkl'
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

print(f"✓ Model loaded successfully from: {model_path}")
print(f"  - Model type: {type(loaded_model).__name__}")
print()

# Create test data for predictions
test_data = pd.DataFrame({
    'id': [1, 3, 5, 10]
})

print("Test data for predictions:")
print(test_data)
print()

# Generate predictions
predictions = loaded_model.predict(test_data)

print("✓ Predictions generated successfully!")
print()
print("Results:")
for i, (idx, pred) in enumerate(zip(test_data['id'], predictions)):
    print(f"  ID {idx} → Predicted age: {pred:.1f} years")
print()

# Create results dataframe
results_df = test_data.copy()
results_df['predicted_age'] = predictions

print("Complete predictions dataframe:")
print(results_df)
