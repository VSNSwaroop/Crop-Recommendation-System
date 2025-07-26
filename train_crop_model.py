import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
csv_path = 'crop_data.csv'
df = pd.read_csv(csv_path)

# Encode soil_type (categorical to one-hot)
soil_encoder = OneHotEncoder(sparse_output=False)
soil_type_1hot = soil_encoder.fit_transform(df[['soil_type']])

# Encode crop (label to integer)
crop_encoder = LabelEncoder()
y = crop_encoder.fit_transform(df['crop'])

# Features: soil_type (one-hot), temperature, rainfall
X = np.concatenate([
    soil_type_1hot,
    df[['temperature', 'rainfall']].values
], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
num_classes = len(np.unique(y))
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1)

# Save model and encoders
model.save('crop_model.h5')
import joblib
joblib.dump(soil_encoder, 'soil_encoder.pkl')
joblib.dump(crop_encoder, 'crop_encoder.pkl')

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {acc:.2f}') 