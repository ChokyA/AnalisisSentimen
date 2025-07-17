import streamlit as st
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Data Latih untuk Naive Bayes (contoh, Anda perlu mengganti ini) ---
# Idealnya, Anda akan memuat ini dari file CSV atau database yang lebih besar
training_texts = [
    "Saya sangat senang hari ini.", "positif",
    "Ini adalah pengalaman yang luar biasa.", "positif",
    "Produk ini sangat bagus dan berfungsi dengan baik.", "positif",
    "Saya suka sekali dengan layanan ini.", "positif",
    "Cuaca hari ini cerah sekali.", "positif",
    "Saya benci dengan situasi ini.", "negatif",
    "Ini adalah hal yang buruk.", "negatif",
    "Sangat mengecewakan.", "negatif",
    "Saya tidak suka makanan ini.", "negatif",
    "Filmnya membosankan.", "negatif",
    "Ini netral.", "netral",
    "Tidak ada yang spesial.", "netral",
    "Oke saja.", "netral",
    "Informasi standar.", "netral",
    "Cukup baik.", "netral" # Ini bisa jadi positif atau netral tergantung konteks, untuk Naive Bayes perlu label yang jelas
]

# Pisahkan teks dan label
texts = [training_texts[i] for i in range(0, len(training_texts), 2)]
labels = [training_texts[i] for i in range(1, len(training_texts), 2)]

# Bagi data menjadi training dan testing set (opsional, tapi disarankan untuk evaluasi model)
# Dalam aplikasi Streamlit sederhana ini, kita bisa latih langsung di semua data
# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# --- Definisikan Classifier Naive Bayes ---
# Kita akan menggunakan pipeline yang menggabungkan CountVectorizer (untuk mengubah teks menjadi fitur numerik)
# dan MultinomialNB (classifier Naive Bayes)
model_naive_bayes = make_pipeline(CountVectorizer(), MultinomialNB())

# Latih model Naive Bayes
model_naive_bayes.fit(texts, labels) # Melatih model dengan semua data

# Fungsi deteksi sentimen dengan TextBlob (tetap ada)
def detect_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        return "Positif"
    elif sentiment < 0:
        return "Negatif"
    else:
        return "Netral"

# Fungsi deteksi sentimen kustom (sekarang menggunakan Naive Bayes)
def detect_sentiment_naive_bayes(text):
    # Prediksi sentimen menggunakan model yang sudah dilatih
    prediction = model_naive_bayes.predict([text])
    return prediction[0]

# Streamlit UI
st.title("Analisis Sentimen")

text_input = st.text_area("Masukkan teks untuk analisis:")

if st.button('Analisis Sentimen'):
    if text_input:
        # Analisis dengan TextBlob
        sentiment_textblob = detect_sentiment_textblob(text_input)
        st.write(f"Hasil Analisis Sentimen (TextBlob): {sentiment_textblob}")

        # Analisis dengan Naive Bayes
        sentiment_naive_bayes = detect_sentiment_naive_bayes(text_input)
        st.write(f"Hasil Analisis Sentimen (Naive Bayes): {sentiment_naive_bayes}")
    else:
        st.write("Mohon masukkan teks untuk analisis.")

# Bagian untuk evaluasi model (opsional, bisa dihapus dari aplikasi akhir)
# if st.checkbox('Tampilkan Evaluasi Model Naive Bayes (Hanya untuk debugging/pengembangan)'):
#     if len(X_test) > 0: # Pastikan ada data test jika split dilakukan
#         y_pred = model_naive_bayes.predict(X_test)
#         st.subheader("Evaluasi Model Naive Bayes:")
#         st.text(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
#         st.text("Laporan Klasifikasi:")
#         st.text(classification_report(y_test, y_pred))
#     else:
#         st.write("Tidak ada data pengujian (test data) untuk evaluasi.")
