import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer


st.set_page_config(
    page_title="Breast Cancer Risk Classifier",
    layout="centered"
)


if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("model_mlp.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model_and_scaler()


data = load_breast_cancer()
feature_names = data.feature_names

df_reference = pd.DataFrame(
    data.data,
    columns=feature_names
)


try:
    with open("accuracy.txt", "r") as f:
        accuracy = float(f.read())
except:
    accuracy = None


st.title("Breast Cancer Risk Classifier")

st.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan risiko kanker payudara "
    "berdasarkan data pemeriksaan menggunakan model Artificial Neural Network."
)


st.subheader("Performa Model")

if accuracy is not None:
    st.metric("Accuracy Model", f"{accuracy * 100:.2f}%")
else:
    st.info("File accuracy.txt belum ditemukan.")


st.subheader("Data Pasien")

patient_name = st.text_input("Nama Pasien")

patient_age = st.number_input(
    "Umur Pasien",
    min_value=1,
    max_value=120,
    value=30
)


st.subheader("Input Data Pemeriksaan")

st.write(
    "Masukkan nilai fitur pemeriksaan. Nilai default diambil dari rata-rata dataset."
)


input_values = []

with st.form("prediction_form"):
    for feature in feature_names:
        mean_value = float(df_reference[feature].mean())
        min_value = float(df_reference[feature].min())
        max_value = float(df_reference[feature].max())

        value = st.number_input(
            label=feature,
            min_value=min_value,
            max_value=max_value,
            value=mean_value,
            step=0.01,
            format="%.5f"
        )

        input_values.append(value)

    submitted = st.form_submit_button("Predict Risk")


if submitted:
    if patient_name.strip() == "":
        st.warning("Mohon isi nama pasien terlebih dahulu.")
    else:
        input_array = np.array(input_values).reshape(1, -1)

        input_scaled = scaler.transform(input_array)

        probabilities = model.predict_proba(input_scaled)[0]

        prob_malignant = float(probabilities[0])
        prob_benign = float(probabilities[1])

        prob_benign_display = np.clip(prob_benign, 0.0001, 0.9999)
        prob_malignant_display = np.clip(prob_malignant, 0.0001, 0.9999)

        if prob_benign >= prob_malignant:
            result = "Low Risk / Benign"
            confidence = prob_benign
            st.success("Hasil Prediksi: Low Risk / Benign")
        else:
            result = "High Risk / Malignant"
            confidence = prob_malignant
            st.error("Hasil Prediksi: High Risk / Malignant")

        confidence_display = np.clip(confidence, 0.0001, 0.9999)

        st.subheader("Hasil Analisis")

        st.metric("Prediksi", result)
        st.metric("Confidence Prediksi", f"{confidence_display * 100:.2f}%")

        st.write("### Detail Probabilitas")
        st.write(f"Probabilitas Benign: **{prob_benign_display * 100:.2f}%**")
        st.write(f"Probabilitas Malignant: **{prob_malignant_display * 100:.2f}%**")

        st.progress(int(confidence_display * 100))

        st.session_state.history.append({
            "nama": patient_name,
            "umur": patient_age,
            "hasil": result,
            "confidence": confidence_display * 100,
            "prob_benign": prob_benign_display * 100,
            "prob_malignant": prob_malignant_display * 100
        })


st.subheader("Riwayat Prediksi Pasien")

if len(st.session_state.history) == 0:
    st.info("Belum ada riwayat prediksi pasien.")
else:
    for i, item in enumerate(reversed(st.session_state.history), start=1):
        with st.expander(f"{i}. {item['nama']} - {item['hasil']}"):
            st.write(f"**Nama Pasien:** {item['nama']}")
            st.write(f"**Umur:** {item['umur']} tahun")
            st.write(f"**Hasil Prediksi:** {item['hasil']}")
            st.write(f"**Confidence:** {item['confidence']:.2f}%")
            st.write(f"**Probabilitas Benign:** {item['prob_benign']:.2f}%")
            st.write(f"**Probabilitas Malignant:** {item['prob_malignant']:.2f}%")
