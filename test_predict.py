import pickle
import numpy as np
import pandas as pd

# --- load dengan pickle ---
with open('model/student_lifestyle_svm.pkl', 'rb') as f:
    svm = pickle.load(f)
with open('model/student_lifestyle_nn.pkl', 'rb') as f:
    nn = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('model/scalerSVM.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/label_encoderSVM.pkl', 'rb') as f:
    le = pickle.load(f)

# cek urutan kolom
feat_names = list(scaler.feature_names_in_)

# 1 baris data baru sebagai dict atau list sesuai urutan
data_dict = {
    'Study_Hours_Per_Day'           : 4.0,
    'Extracurricular_Hours_Per_Day' : 3.0,
    'Sleep_Hours_Per_Day'           : 4.5,
    'Social_Hours_Per_Day'          : 2.0,
    'Physical_Activity_Hours_Per_Day':1.5,
    'GPA'                           : 3.2
}

# jadikan DataFrame agar nama kolomnya cocok
X_new = pd.DataFrame([data_dict], columns=feat_names)

# transform
X_scaled = scaler.transform(X_new)

# prediksi
svm_pred = svm.predict(X_scaled)
nn_pred = nn.predict(X_scaled)
# inverse transform
print("SVM predicts:", le.inverse_transform(svm_pred))
print("NN predicts:", le.inverse_transform(nn_pred))
