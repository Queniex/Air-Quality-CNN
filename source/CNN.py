import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load scaler and label encoder
scaler = joblib.load('source/scaler2.pkl')
label_encoder = joblib.load('source/label_encoder2.pkl')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='source/cnn_air_quality_model2.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

    # Prepare the input tensor
    interpreter.set_tensor(input_details[0]['index'], new_features.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    return predicted_labels[0]

# Example usage
# user_input = get_user_input()
# predicted_category = predict_air_quality(user_input)
# print(f"Kategori yang diprediksi: {predicted_category}")