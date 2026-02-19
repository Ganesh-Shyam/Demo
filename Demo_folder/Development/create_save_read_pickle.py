import pandas as pd
import pickle
import os

# Create dummy dataframe with sample data
dummy_df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 42],
    'city': ['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle']
})

print("Created dummy dataframe:")
print(dummy_df)
print()

# Create output folder if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save dataframe as pickle file
pickle_path = 'output/dummy_data.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(dummy_df, f)

print(f"Pickle file saved to: {pickle_path}")
print()

# Read the pickle file back
with open(pickle_path, 'rb') as f:
    loaded_df = pd.read_pickle(f)

print("Data read from pickle file:")
print(loaded_df)
print()

# Verify data matches
print("Data verification:")
print(f"Original shape: {dummy_df.shape}")
print(f"Loaded shape: {loaded_df.shape}")
print(f"Data matches: {dummy_df.equals(loaded_df)}")