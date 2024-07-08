# from flask import Flask, redirect, url_for, request, render_template, jsonify

from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app=Flask(__name__)
CORS(app)

# Load scaler and label encoder
scaler = joblib.load('source/scaler.pkl')
label_encoder = joblib.load('source/label_encoder.pkl')

# Load the saved CNN model
model = tf.keras.models.load_model('source/cnn_air_quality_model.h5')

def predict_air_quality(user_input):
    user_input_df = pd.DataFrame(user_input, columns=['pm10', 'so2', 'co', 'o3', 'no2'])
    
    new_features = scaler.transform(user_input_df)
    new_features = np.expand_dims(new_features, axis=2)
    
    predictions = model.predict(new_features)
    predicted_classes = np.argmax(predictions, axis=1)
    
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    return predicted_labels[0]

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        pm10 = float(request.form["pm10"])
        so2 = float(request.form["so2"])
        co = float(request.form["co"])
        o3 = float(request.form["o3"])
        no2 = float(request.form["no2"])

        user_input = np.array([[pm10, so2, co, o3, no2]])
        predicted_category = predict_air_quality(user_input)

        return render_template("result.html", prediction=predicted_category)

@app.route("/result")
def result():
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/<not_found>")
def notfound(not_found):
    return render_template("notfound.html", page=not_found)

app.run(debug=True)
