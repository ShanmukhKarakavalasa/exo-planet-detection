from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the model and scaler
model = tf.keras.models.load_model('exoplanet_model.h5')
scaler = joblib.load('scaler.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    fluxes = [float(x) for x in request.form.values()]
    data = np.array(fluxes).reshape(1, -1)
    data = scaler.transform(data)  # Scale input data

    prediction = model.predict(data)
    output = (prediction > 1.0).astype("int32")[0][0]

    print(prediction)
    if output == 1:
        result = "Exoplanet Confirmed!"
    else:
        result = "No Exoplanet Detected."

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
