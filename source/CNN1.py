import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load the Keras model
keras_model = tf.keras.models.load_model('source/cnn_air_quality_model2.h5')

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('source/cnn_air_quality_model2.tflite', 'wb') as f:
    f.write(tflite_model)

# Load scaler and label encoder
scaler = joblib.load('source/scaler2.pkl')
label_encoder = joblib.load('source/label_encoder2.pkl')

# Load the saved CNN model
model = tf.keras.models.load_model('source/cnn_air_quality_model2.h5')

def get_user_input():
    pm10 = float(input("Masukkan nilai pm10: "))
    so2 = float(input("Masukkan nilai so2: "))
    co = float(input("Masukkan nilai co: "))
    o3 = float(input("Masukkan nilai o3: "))
    no2 = float(input("Masukkan nilai no2: "))
    return np.array([[pm10, so2, co, o3, no2]])

def predict_air_quality(user_input):
    user_input_df = pd.DataFrame(user_input, columns=['pm10', 'so2', 'co', 'o3', 'no2'])
    
    new_features = scaler.transform(user_input_df)
    new_features = np.expand_dims(new_features, axis=2)
    
    predictions = model.predict(new_features)
    predicted_classes = np.argmax(predictions, axis=1)
    
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    return predicted_labels[0]

# # Example usage
# user_input = get_user_input()
# predicted_category = predict_air_quality(user_input)
# print(f"Kategori yang diprediksi: {predicted_category}")