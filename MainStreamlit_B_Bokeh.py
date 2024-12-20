import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

st.set_page_config(
    page_title="Klasifikasi Jenis Anggur",
    page_icon="🍇",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #4c4c6d;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1f1f2e;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #1f1f2e;
        margin: 1rem 0;
        color: white;
    }
    .confidence-bar {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #7272a8;
    }
    .info-box {
        background-color: #1f1f2e;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #4c4c6d;
    }
    .upload-box {
        border: 2px dashed #4c4c6d;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .stAlert {
        background-color: #1f1f2e;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: white !important;
    }
    .css-1dp5vir {
        background-color: #1f1f2e;
        border: 1px solid #4c4c6d;
    }
    ul, ol {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_classification_model():
    return load_model(r'BaseModel_MobileNet_Bokeh.h5')

model = load_classification_model()
class_names = ['Merah', 'Thompson', 'Concord']

def classify_image(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

st.title("🍇 Sistem Klasifikasi Jenis Anggur")
st.markdown("""
    <div class='info-box'>
        <h4>Tentang Aplikasi</h4>
        <p>Aplikasi ini menggunakan model yang dilatih untuk mengklasifikasikan 3 jenis anggur:</p>
        <ul>
            <li>Anggur Merah</li>
            <li>Anggur Thompson</li>
            <li>Anggur Concord</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📸 Upload Gambar")
    st.markdown("""
        <div class='upload-box'>
            <p>Format yang didukung: JPG, PNG, JPEG</p>
            <p>Anda dapat memilih beberapa gambar sekaligus</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Unggah Gambar Anggur", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown("### 🖼️ Preview Gambar")
        for idx, uploaded_file in enumerate(uploaded_files):
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)


with col2:
    st.markdown("### 🎯 Kontrol Prediksi")
    
    predict_button = st.button("🔍 Mulai Prediksi", use_container_width=True)
    
    if predict_button:
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.markdown(f"#### 📊 Analisis: {uploaded_file.name}")
                
                with st.spinner('Menganalisis gambar...'):
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    label, confidence = classify_image(uploaded_file.name)
                    time.sleep(0.5)

                if label != "Error":
                    st.markdown("""
                        <div style='padding: 1rem; border-radius: 10px; background-color: #2e2e4d; margin: 1rem 0;'>
                            ✨ Prediksi Berhasil!
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='prediction-box'>
                            <h4>Hasil Prediksi:</h4>
                            <h2 style='color: #7272a8;'>{label}</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### Tingkat Kepercayaan:")
                    for idx, (class_name, conf) in enumerate(zip(class_names, confidence)):
                        confidence_percentage = float(conf * 100)
                        st.markdown(f"**{class_name}**")
                        st.progress(confidence_percentage / 100)
                        st.markdown(f"**{confidence_percentage:.1f}%**")
                        
                    st.markdown("<hr style='border-color: #4c4c6d;'>", unsafe_allow_html=True)
                else:
                    st.error(f"Terjadi kesalahan: {confidence}")
        else:
            st.warning("⚠️ Silakan unggah gambar terlebih dahulu!")

    st.markdown("""
        <div class='info-box' style='margin-top: 2rem;'>
            <h4>📝 Petunjuk Penggunaan:</h4>
            <ol>
                <li>Upload satu atau beberapa gambar anggur</li>
                <li>Klik tombol "Mulai Prediksi"</li>
                <li>Tunggu hasil analisis</li>
                <li>Lihat hasil prediksi dan tingkat kepercayaan</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
