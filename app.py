from flask import Flask, render_template, request, redirect, url_for
import pickle, io
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024   # batasi upload 2 MB

# — Load model & preprocessing sekali saat startup —
with open('model/student_lifestyle_svm.pkl','rb') as f:
    svm = pickle.load(f)
with open('model/student_lifestyle_nn.pkl','rb') as f:
    nn = pickle.load(f)
with open('model/scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open('model/label_encoder.pkl','rb') as f:
    le = pickle.load(f)
with open('model/scalerSVM.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/label_encoderSVM.pkl', 'rb') as f:
    le = pickle.load(f)

# urutan kolom seperti scaler.feature_names_in_
FEATURES = list(scaler.feature_names_in_)

@app.route("/", methods=["GET","POST"])
def index():
    manual_pred = None
    upload_results = None

    if request.method == "POST":
        # 1) Jika datang dari manual form
        if 'submit_manual' in request.form:
            # kumpulkan 6 nilai
            vals = [float(request.form[f]) for f in FEATURES]
            X = scaler.transform([vals])
            model = svm if request.form['model_choice']=='svm' else nn
            p = model.predict(X)
            manual_pred = le.inverse_transform(p)[0]

        # 2) Jika datang dari upload CSV
        elif 'submit_upload' in request.form:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
                # validasi kolom
                if not all(col in df.columns for col in FEATURES):
                    upload_results = f"CSV harus memiliki kolom: {FEATURES}"
                else:
                    X = scaler.transform(df[FEATURES])
                    model = svm if request.form['model_choice_upload']=='svm' else nn
                    preds = model.predict(X)
                    df['Prediction'] = le.inverse_transform(preds)
                    # kirim DataFrame ke template
                    upload_results = df.to_dict(orient='records')
            else:
                upload_results = "File tidak valid. Unggah CSV."

    return render_template("index.html",
                            features=FEATURES,
                            manual_pred=manual_pred,
                            upload_results=upload_results)

if __name__ == "__main__":
    app.run(debug=True)
