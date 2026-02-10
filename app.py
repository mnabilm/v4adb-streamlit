import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model & fitur
# =========================
model = joblib.load("model/model_rf_adb.pkl")
fitur = joblib.load("model/fitur_model.pkl")

# =========================
# Judul Aplikasi
# =========================
st.title("Prediksi Anemia Defisiensi Besi (ADB)")
st.write(
    """
    Aplikasi ini digunakan untuk **memprediksi kemungkinan Anemia Defisiensi Besi (ADB)**
    berdasarkan parameter pemeriksaan darah.
    
    ⚠️ *Hasil prediksi ini bukan diagnosis medis.*
    """
)

st.divider()

# =========================
# Input Pengguna
# =========================
st.subheader("Input Data Pemeriksaan Darah")

gender = st.radio(
    "Jenis Kelamin",
    options=[0, 1],
    format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan"
)

hgb = st.number_input("Hemoglobin (HGB)", min_value=0.0, step=0.1)
mch = st.number_input("Mean Corpuscular Hemoglobin (MCH)", min_value=0.0, step=0.1)
mcv = st.number_input("Mean Corpuscular Volume (MCV)", min_value=0.0, step=0.1)
mchc = st.number_input("Mean Corpuscular Hemoglobin Concentration (MCHC)", min_value=0.0, step=0.1)

st.divider()

# =========================
# Tombol Prediksi
# =========================
if st.button("Prediksi"):
    if hgb == 0 or mch == 0 or mcv == 0 or mchc == 0:
        st.warning("Mohon lengkapi seluruh data dengan nilai yang valid.")
    else:
        # Susun input sesuai fitur model
        data_input = pd.DataFrame([{
            "GENDER": gender,
            "HGB": hgb,
            "MCH": mch,
            "MCV": mcv,
            "MCHC": mchc
        }])

        data_input = data_input[fitur]

        # Prediksi
        prediksi = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0][1]

        # =========================
        # Output
        # =========================
        st.subheader("Hasil Prediksi")

        if prediksi == 1:
            st.error("Status: **Anemia Defisiensi Besi (ADB)**")
        else:
            st.success("Status: **Tidak Anemia Defisiensi Besi**")

        st.write(
            f"Tingkat keyakinan model terhadap ADB: **{probabilitas * 100:.2f}%**"
        )

        st.caption(
            "Catatan: Sistem ini bersifat prediktif dan digunakan untuk keperluan akademik."
        )
