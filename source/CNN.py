import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Muat scaler dan label encoder yang telah dilatih
scaler = joblib.load('.\source\scaler.pkl')
label_encoder = joblib.load('.\source\label_encoder.pkl')

# Fungsi untuk menerima input dari pengguna
def get_user_input():
    pm10 = float(input("Masukkan nilai pm10: "))
    so2 = float(input("Masukkan nilai so2: "))
    co = float(input("Masukkan nilai co: "))
    o3 = float(input("Masukkan nilai o3: "))
    no2 = float(input("Masukkan nilai no2: "))
    return np.array([[pm10, so2, co, o3, no2]])

# Ambil input dari pengguna
user_input = get_user_input()

# Normalisasi input pengguna
new_features = scaler.transform(user_input)

# Reshape data agar sesuai dengan input model CNN
new_features = np.expand_dims(new_features, axis=2)

# Muat model yang sudah disimpan
model = load_model('.\source\cnn_air_quality_model.keras')

# Buat prediksi
predictions = model.predict(new_features)
predicted_classes = np.argmax(predictions, axis=1)

# Decode kelas prediksi
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Tampilkan hasil prediksi
print(f"Kategori yang diprediksi: {predicted_labels[0]}")

# Jika ingin menyimpan hasil ke file CSV (opsional)
new_data = pd.DataFrame(user_input, columns=['pm10', 'so2', 'co', 'o3', 'no2'])
new_data['predicted_category'] = predicted_labels