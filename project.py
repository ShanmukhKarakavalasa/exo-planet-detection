import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and preprocess the dataset
data_train = pd.read_csv(r'ExoPlanetTrain.csv')
data_test = pd.read_csv(r'ExoPlanetTest.csv')

X_train = data_train.iloc[:, 1:10].values  # Flux columns (FLUX1-FLUX9)
y_train = data_train.iloc[:, 0].values     # Labels

X_test = data_test.iloc[:, 1:10].values
y_test = data_test.iloc[:, 0].values

# Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.save')

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(9,)),  # 9 flux inputs
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=5, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Non-Exoplanet", "Exoplanet"]))

# Save the model
model.save('exoplanet_model.h5')
