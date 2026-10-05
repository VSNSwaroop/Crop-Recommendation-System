import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import layers, models

csv_path = "crop_data.csv"
df = pd.read_csv(csv_path)

soil_encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
soil_type_1hot = soil_encoder.fit_transform(df[["soil_type"]])

crop_encoder = LabelEncoder()
y = crop_encoder.fit_transform(df["crop"])

X = np.concatenate(
    [
        soil_type_1hot,
        df[["temperature", "rainfall"]].values,
    ],
    axis=1,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

num_classes = len(np.unique(y))
model = models.Sequential(
    [
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    verbose=1,
)

model.save("crop_model.h5")
joblib.dump(soil_encoder, "soil_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.2f}")
