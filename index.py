import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import json

# Load model dan vectorizer
svm_model = joblib.load("svm_sentiment_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

data_file = "sentiment_data.json"
dataset_file = "processed_dataset.json"

# Fungsi untuk memuat data statistik sentimen
def load_sentiment_data():
    try:
        with open(data_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            return {key: int(value) for key, value in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"positif": 0, "netral": 0, "negatif": 0}

# Fungsi untuk menyimpan data statistik sentimen
def save_sentiment_data(data):
    with open(data_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Fungsi untuk memuat dataset yang sudah diproses
def load_processed_dataset():
    try:
        with open(dataset_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Fungsi untuk menyimpan dataset yang sudah diproses
def save_processed_dataset(data):
    with open(dataset_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Fungsi untuk prediksi sentimen
def predict_sentiment(text):
    text_vectorized = tfidf_vectorizer.transform([text])  # Transformasi TF-IDF
    prediction = svm_model.predict(text_vectorized)[0]  # Prediksi dengan SVM
    return prediction

# Inisialisasi data sentimen dan dataset
sentiment_data = load_sentiment_data()
processed_dataset = load_processed_dataset()

# UI Streamlit
st.set_page_config(page_title="Analisis Sentimen", page_icon="💬", layout="centered")

st.title("💬 Analisis Sentimen Terhadap Influencer Timothy Ronald")
st.write("Masukkan teks atau unggah file CSV untuk mengetahui sentimennya.")

# Input pengguna
user_input = st.text_area("📝 Masukkan teks:", height=100)

# Tombol analisis teks tunggal
if st.button("🔍 Analisis Sentimen"):
    if user_input:
        result = predict_sentiment(user_input)

        # Update statistik hanya untuk input teks tunggal
        sentiment_data[result] += 1
        save_sentiment_data(sentiment_data)

        # Menampilkan hasil dengan warna berbeda
        if result == "positif":
            st.success("✅ **Sentimen Positif**")
        elif result == "netral":
            st.info("ℹ️ **Sentimen Netral**")
        else:
            st.error("❌ **Sentimen Negatif**")
    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu.")

# 📌 **Syarat Dataset**
st.markdown("---")
st.markdown("### 📌 Syarat Dataset yang Dapat Dimasukkan")
st.markdown("""
1. Format file harus **CSV (.csv)**.
2. Harus memiliki **kolom 'teks'** berisi teks yang akan dianalisis.
3. Pastikan **tidak ada baris kosong** atau **format yang salah**.
4. Jika teks sudah pernah dianalisis, tetap bisa melihat hasilnya tanpa analisis ulang.
""")

# 📂 **Unggah Dataset**
st.subheader("📂 Unggah File CSV untuk Analisis Massal")
file = st.file_uploader("Unggah file CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    
    if "teks" in df.columns:
        st.write("✅ File berhasil diunggah! Menampilkan hasil analisis...")

        # Buat dictionary dari dataset yang sudah diproses untuk pencarian cepat
        processed_texts = {item["teks"]: item["sentimen"] for item in processed_dataset}
        
        # Cek apakah teks sudah dianalisis sebelumnya
        df["sentimen"] = df["teks"].map(processed_texts)

        # Analisis hanya untuk teks yang belum dianalisis
        new_data = df[df["sentimen"].isna()].copy()

        if not new_data.empty:
            new_data["sentimen"] = new_data["teks"].apply(predict_sentiment)

            # Konversi tipe data kolom sentimen ke string untuk menghindari warning
            df["sentimen"] = df["sentimen"].astype(str)
            df.loc[new_data.index, "sentimen"] = new_data["sentimen"]

            # Tambahkan hasil baru ke dataset yang sudah diproses
            for _, row in new_data.iterrows():
                processed_dataset.append({"teks": row["teks"], "sentimen": row["sentimen"]})
                sentiment_data[row["sentimen"]] += 1  # Update statistik

            # Simpan dataset yang sudah diperbarui
            save_processed_dataset(processed_dataset)
            save_sentiment_data(sentiment_data)

        # Tampilkan seluruh dataset (termasuk yang sudah ada sebelumnya)
        st.write("### Hasil Analisis Sentimen:")
        st.dataframe(df)

    else:
        st.error("⚠️ Pastikan file memiliki kolom 'teks' untuk diproses.")

# 📊 **Menampilkan Grafik Sentimen**
st.markdown("---")
st.subheader("📊 Statistik Sentimen")

total_sentimen = sum(sentiment_data.values())

if total_sentimen == 0:
    st.warning("📊 Belum ada data sentimen yang bisa ditampilkan.")
else:
    # **Pie Chart dengan Label Jumlah Sentimen**
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [f"{key} ({value})" for key, value in sentiment_data.items()]
    colors = ['green', 'blue', 'red']
    ax.pie(
        sentiment_data.values(),
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "black"},
        textprops={"fontsize": 10}
    )
    ax.set_title("Distribusi Sentimen", fontsize=14, fontweight="bold")
    st.pyplot(fig)

    # **Tampilkan Total Sentimen Per Kategori**
    st.markdown("### Total Sentimen Per Kategori:")
    st.write(f"- ✅ **Positif**: {sentiment_data['positif']}")
    st.write(f"- ℹ️ **Netral**: {sentiment_data['netral']}")
    st.write(f"- ❌ **Negatif**: {sentiment_data['negatif']}")
    st.info(f"📊 **Total Semua Sentimen: {total_sentimen}**")
