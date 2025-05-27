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
    default_stats = {
        "total_predictions": 0,
        "successful_predictions": 0,
        "low": {"Benar": 0, "Salah": 0},
        "moderate": {"Benar": 0, "Salah": 0},
        "high": {"Benar": 0, "Salah": 0}
    }
    
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'w') as f:
            json.dump(default_stats, f)
        return default_stats
        
    try:
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
            # Pastikan semua kategori ada
            for key in default_stats.keys():
                if key not in stats:
                    stats[key] = default_stats[key]
            return stats
    except:
        return default_stats

# Fungsi untuk menyimpan statistik ke file
def save_stats():
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(prediction_stats, f, indent=4)
    except Exception as e:
        print(f"Error saving stats: {e}")

# Statistik prediksi
prediction_stats = load_stats()

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    manual_pred = None
    upload_results = None
    manual_input = None

    if request.method == "POST":
        # 1) Jika datang dari manual form
        if 'submit_manual' in request.form:
            try:
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
                vals = []
                error_message = None
                for f in features:
                    value = float(request.form[f])
                    min_val, max_val = FEATURE_RANGES[f]
                    if not (min_val <= value <= max_val):
                        error_message = f"Nilai untuk {f.replace('_', ' ')} harus berada dalam rentang {min_val} hingga {max_val}."
                        break
                    vals.append(value)

                if error_message:
                    return render_template("index.html",
                                           features=FEATURES_SVM,
                                           manual_pred=None,
                                           manual_input=None,
                                           upload_results=None,
                                           error_message=error_message)

                # Jika validasi berhasil, lanjutkan ke prediksi
                manual_input = dict(zip(features, vals))
                X = scaler.transform([vals])
                p = model.predict(X)
                manual_pred = le.inverse_transform(p)[0]

                # Update statistik prediksi
                if model_choice == 'svm':
                    # Gunakan prediksi SVM sebagai referensi
                    X_ref = scalersvm.transform([vals])
                    p_ref = svm.predict(X_ref)
                    ref_pred = lesvm.inverse_transform(p_ref)[0]
                else:
                    # Gunakan prediksi NN sebagai referensi
                    X_ref = scalernn.transform([vals])
                    p_ref = nn.predict(X_ref)
                    ref_pred = lenn.inverse_transform(p_ref)[0]

                is_correct = manual_pred == ref_pred
                update_prediction_stats(manual_pred, is_correct)
                
                # Tambahkan status prediksi ke output
                manual_input['Prediction_Status'] = 'Benar' if is_correct else 'Salah'

                save_stats()  # Pastikan perubahan tersimpan

            except Exception as e:
                return render_template("index.html",
                                       features=FEATURES_SVM,
                                       manual_pred=None,
                                       manual_input=None,
                                       upload_results=None,
                                       error_message=f"Terjadi kesalahan: {str(e)}")

        # 2) Jika datang dari upload CSV
        elif 'submit_upload' in request.form:
            try:
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

                        # Update statistik untuk setiap baris
                        for idx, row in df.iterrows():
                            if model_choice_upload == 'svm':
                                X_ref = scalersvm.transform(row[features].values.reshape(1, -1))
                                p_ref = svm.predict(X_ref)
                                ref_pred = lesvm.inverse_transform(p_ref)[0]
                            else:
                                X_ref = scalernn.transform(row[features].values.reshape(1, -1))
                                p_ref = nn.predict(X_ref)
                                ref_pred = lenn.inverse_transform(p_ref)[0]
                                
                            is_correct = row['Prediction'] == ref_pred
                            update_prediction_stats(row['Prediction'], is_correct)
                            df.at[idx, 'Prediction_Status'] = 'Benar' if is_correct else 'Salah'

                        # Simpan hasil prediksi ke file Excel
                        output_file = "static/results_predictions.xlsx"
                        df.to_excel(output_file, index=False)

                        # Kirim DataFrame ke template
                        upload_results = df.to_dict(orient='records')
                else:
                    upload_results = "File tidak valid. Unggah CSV."

            except Exception as e:
                upload_results = f"Terjadi kesalahan saat memproses file: {str(e)}"

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



def calculate_target_statistics():
    default_stats = {
        "low": {"Benar": 0, "Salah": 0},
        "moderate": {"Benar": 0, "Salah": 0},
        "high": {"Benar": 0, "Salah": 0}
    }
    
    if not os.path.exists(STATS_FILE):
        return default_stats

    try:
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
            # Hanya ambil statistik kategori (low, moderate, high)
            category_stats = {
                k: v for k, v in stats.items() 
                if k in ['low', 'moderate', 'high']
            }
            
            # Pastikan setiap kategori memiliki struktur yang benar
            for category in ['low', 'moderate', 'high']:
                if category not in category_stats:
                    category_stats[category] = {"Benar": 0, "Salah": 0}
                elif not isinstance(category_stats[category], dict):
                    category_stats[category] = {"Benar": 0, "Salah": 0}
                    
            return category_stats
    except:
        return default_stats

def update_target_statistics(target, status):
    stats = calculate_target_statistics()

    if target in stats:
        stats[target][status] += 1

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

def update_prediction_stats(prediction, is_correct):
    global prediction_stats
    
    # Update kategori (low, moderate, high)
    prediction = prediction.lower()  # konversi ke lowercase untuk konsistensi
    if prediction in ['low', 'moderate', 'high']:
        status = "Benar" if is_correct else "Salah"
        prediction_stats[prediction][status] += 1
    
    # Update total statistik
    prediction_stats["total_predictions"] += 1
    if is_correct:
        prediction_stats["successful_predictions"] += 1
            
    save_stats()

@app.route("/dashboard")
def dashboard():
    target_stats = calculate_target_statistics()
    return render_template("dashboard.html",
                            total_predictions=prediction_stats.get("total_predictions", 0),
                            successful_predictions=prediction_stats.get("successful_predictions", 0),
                            success_rate=calculate_success_rate(),
                            target_stats=target_stats)


@app.route("/thanks")
def thanks():
    return render_template("thanks.html")

if __name__ == "__main__":
    app.run(debug=True)
