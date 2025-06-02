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
from PIL import Image, ImageDraw, ImageFont
import threading

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

HISTORY_FILE = "history.json"
history_lock = threading.Lock()

def append_history(entry):
    with history_lock:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            history = []
        history.append(entry)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

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
                manual_input['Prediction_Label'] = manual_pred

                # Tambahkan status prediksi ke output
                manual_input['Prediction_Status'] = 'Benar'  # Asumsikan benar untuk input manual
                manual_input['Metode'] = model_choice.upper()
                append_history({
                    "fitur": {k: v for k, v in manual_input.items() if k not in ["Prediction_Label", "Prediction_Status", "Metode"]},
                    "hasil_prediksi": manual_input.get("Prediction_Label"),
                    "metode": manual_input.get("Metode"),
                    "tipe": "Manual"
                })

                # Simpan input manual ke file sementara untuk diunduh sebagai JPG
                with open("manual_input_temp.json", "w") as f:
                    json.dump(manual_input, f)

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

                        # Simpan hasil prediksi ke file Excel
                        output_file = "static/results_predictions.xlsx"
                        df.to_excel(output_file, index=False)

                        # Kirim DataFrame ke template
                        upload_results = df.to_dict(orient='records')
                        for _, row in df.iterrows():
                            fitur = {col: row[col] for col in features}
                            append_history({
                                "fitur": fitur,
                                "hasil_prediksi": row["Prediction"],
                                "metode": model_choice_upload.upper(),
                                "tipe": "Upload"
                            })
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


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/thanks")
def thanks():
    return render_template("thanks.html")

@app.route("/download_manual_prediction")
def download_manual_prediction():
    try:
        if os.path.exists("manual_input_temp.json"):
            with open("manual_input_temp.json", "r") as f:
                manual_input = json.load(f)
        else:
            return "Tidak ada data prediksi manual terbaru.", 404
    except Exception as e:
        return f"Error: {e}", 500

    # Siapkan teks tanpa Prediction_Status
    lines = []
    for k, v in manual_input.items():
        if k != "Prediction_Status":
            lines.append(f"{k.replace('_',' ')}: {v}")
    # Tambahkan hasil label prediksi (ambil dari manual_input)
    pred_label = manual_input.get("Prediction_Label") or manual_input.get("Prediction") or None
    if pred_label:
        lines.append(f"Hasil Prediksi: {pred_label}")

    # Prediction_Status tidak perlu ditambahkan lagi

    # Buat gambar dari teks
    font = ImageFont.load_default()
    padding = 10
    width = 400
    line_height = 20
    height = line_height * len(lines) + 2 * padding

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_height), line, fill="black", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="manual_prediction.jpg")

@app.route("/history")
def history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    # Statistik jumlah prediksi per model
    model_counts = {"SVM": 0, "NN": 0}
    hasil_per_model = {"SVM": {}, "NN": {}}
    for item in history:
        model = item.get("metode", "UNKNOWN")
        hasil = item.get("hasil_prediksi", "UNKNOWN")
        if model in model_counts:
            model_counts[model] += 1
            hasil_per_model[model][hasil] = hasil_per_model[model].get(hasil, 0) + 1

    # Insight teks
    most_used = max(model_counts, key=model_counts.get) if history else "-"
    most_pred_svm = max(hasil_per_model["SVM"], key=hasil_per_model["SVM"].get) if hasil_per_model["SVM"] else "-"
    most_pred_nn = max(hasil_per_model["NN"], key=hasil_per_model["NN"].get) if hasil_per_model["NN"] else "-"

    return render_template(
        "history.html",
        history=history,
        model_counts=model_counts,
        hasil_per_model=hasil_per_model,
        most_used=most_used,
        most_pred_svm=most_pred_svm,
        most_pred_nn=most_pred_nn
    )

if __name__ == "__main__":
    app.run(debug=True)
