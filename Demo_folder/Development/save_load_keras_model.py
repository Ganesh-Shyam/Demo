import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Create the 'output' directory if it doesn't exist
folder_name = "model"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 2. Define a simple Sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(10, activation='softmax')
])

# 3. Compile the model (required to save the training state)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Save the model as a .keras file
model_path = os.path.join(folder_name, "my_model.keras")
model.save(model_path)

print(f"Model saved successfully at: {model_path}")

# 5. Load the model back
loaded_model = keras.models.load_model(model_path)

# Verify the structure
loaded_model.summary()