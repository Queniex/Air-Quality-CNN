from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app=Flask(__name__)
CORS(app)

from source.CNN import (
    predict_air_quality
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process")
def process():
    return render_template("process.html")

@app.route("/insight")
def insight():
    return render_template("insight.html")

@app.route("/result", methods=["GET", "POST"])
def result():
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
        # return render_template("result.html") 

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/<not_found>")
def notfound(not_found):
    return render_template("notfound.html", page=not_found)

# app.run(debug=True)
