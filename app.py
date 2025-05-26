from flask import Flask, render_template, request, redirect, url_for
import pickle, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import json
import os
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024   # batasi upload 2 MB

# — Load model & preprocessing sekali saat startup —
with open('model/student_lifestyle_svm.pkl', 'rb') as f:
    svm = pickle.load(f)
with open('model/student_lifestyle_nn.pkl', 'rb') as f:
    nn = pickle.load(f)
with open('model/scalerSVM.pkl', 'rb') as f:
    scalersvm = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scalernn = pickle.load(f)
with open('model/label_encoderSVM.pkl', 'rb') as f:
    lesvm = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    lenn = pickle.load(f)

# Urutan kolom seperti scaler.feature_names_in_
FEATURES_SVM = list(scalersvm.feature_names_in_)
FEATURES_NN = list(scalernn.feature_names_in_)
# Rentang batasan untuk setiap fitur
FEATURE_RANGES = {
    "Study_Hours_Per_Day": (0, 12),
    "Extracurricular_Hours_Per_Day": (0, 10),
    "Sleep_Hours_Per_Day": (0, 12),
    "Social_Hours_Per_Day": (0, 10),
    "Physical_Activity_Hours_Per_Day": (0, 10),
    "GPA": (0, 4)
}

STATS_FILE = prediction_stats = "prediction_stats.json"

# Fungsi untuk memuat statistik dari file
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {"total_predictions": 0, "successful_predictions": 0}

# Fungsi untuk menyimpan statistik ke file
def save_stats():
    with open(STATS_FILE, "w") as f:
        json.dump(prediction_stats, f)


# Statistik prediksi
prediction_stats = load_stats()

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/predict')
def predict():
    manual_pred = None
    upload_results = None
    manual_input = None

    if request.method == "POST":
        # 1) Jika datang dari manual form
        if 'submit_manual' in request.form:
            # Pilih model berdasarkan input pengguna
            model_choice = request.form['model_choice']
            if model_choice == 'svm':
                features = FEATURES_SVM
                scaler = scalersvm
                le = lesvm
                model = svm
            else:  # 'nn'
                features = FEATURES_NN
                scaler = scalernn
                le = lenn
                model = nn

            # Kumpulkan nilai input
            vals = [float(request.form[f]) for f in features]
            manual_input = dict(zip(features, vals))
            X = scaler.transform([vals])
            p = model.predict(X)
            manual_pred = le.inverse_transform(p)[0]

            # Tambahkan hasil prediksi ke data input
            manual_input['Predicted_Stress_Level'] = manual_pred

            # Simpan hasil prediksi ke gambar
            plt.figure(figsize=(8, 4))
            plt.title("Hasil Prediksi Tingkat Stres")

            #  Hanya gunakan fitur numerik untuk grafik
            numeric_features = {key: value for key, value in manual_input.items() if isinstance(value, (int, float))}
            bars = plt.bar(numeric_features.keys(), numeric_features.values(), color='skyblue')

            # Tambahkan teks hasil prediksi di atas grafik
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{bar.get_height():.2f}", ha='center', va='bottom')

            # Tambahkan hasil prediksi sebagai teks di grafik
            plt.text(len(numeric_features) - 1, max(numeric_features.values()) + 0.5,
                    f"Predicted Stress Level: {manual_pred}", fontsize=12, color='red', ha='center')
            
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("Fitur")
            plt.ylabel("Nilai")
            plt.tight_layout()
            plt.savefig("static/manual_prediction.jpg")
            plt.close()
            

        # 2) Jika datang dari upload CSV
        elif 'submit_upload' in request.form:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))

                # Pilih model berdasarkan input pengguna
                model_choice_upload = request.form['model_choice_upload']
                if model_choice_upload == 'svm':
                    features = FEATURES_SVM
                    scaler = scalersvm
                    le = lesvm
                    model = svm
                else:  # 'nn'
                    features = FEATURES_NN
                    scaler = scalernn
                    le = lenn
                    model = nn

                # Validasi kolom
                if not all(col in df.columns for col in features):
                    upload_results = f"CSV harus memiliki kolom: {features}"
                else:
                    X = scaler.transform(df[features])
                    preds = model.predict(X)
                    df['Prediction'] = le.inverse_transform(preds)

                    # Simpan hasil prediksi ke file Excel
                    output_file = "static/results_predictions.xlsx"
                    df.to_excel(output_file, index=False)

                    # Kirim DataFrame ke template
                    upload_results = df.to_dict(orient='records')
            else:
                upload_results = "File tidak valid. Unggah CSV."

    return render_template("index.html",
                           features=FEATURES_SVM,  # Default fitur untuk form manual
                           manual_pred=manual_pred,
                           manual_input=manual_input,
                           upload_results=upload_results)



@app.route("/download_template")
def download_template():
    # Buat template CSV
    template_data = """Study_Hours_Per_Day,Extracurricular_Hours_Per_Day,Sleep_Hours_Per_Day,Social_Hours_Per_Day,Physical_Activity_Hours_Per_Day,GPA
6.5,2.1,7.2,1.7,6.5,2.88
8.1,0.6,6.5,2.2,6.6,3.51
5.3,3.5,8,4.2,3,2.75
"""
    # Kirim file CSV sebagai respons
    return send_file(
        io.BytesIO(template_data.encode('utf-8')),
        mimetype="text/csv",
        as_attachment=True,
        download_name="template_student_lifestyle.csv"
    )

def calculate_success_rate():
    if prediction_stats["total_predictions"] == 0:
        return 0
    return (prediction_stats["successful_predictions"] / prediction_stats["total_predictions"]) * 100

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html",
                           total_predictions=prediction_stats["total_predictions"],
                           successful_predictions=prediction_stats["successful_predictions"],
                           success_rate=calculate_success_rate())

@app.route("/thanks")
def thanks():
    return render_template("thanks.html")

if __name__ == "__main__":
    app.run(debug=True)
