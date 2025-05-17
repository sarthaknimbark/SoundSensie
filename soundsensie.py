import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
from tensorflow.image import resize
from streamlit_option_menu import option_menu
import plotly.express as px

# ========== CONFIG ==========
st.set_page_config(page_title="SoundSensie | Genre Classifier", page_icon="🎧", layout="wide")

# ========== STYLES ==========
st.markdown("""
<style>
:root {
    --primary-color: #4CAF50;
    --secondary-color: #2e4053;
    --accent-color: #FF4B4B;
    --font-family: 'Segoe UI', sans-serif;
}
body {
    font-family: var(--font-family);
    background-color: #f4f6fa;
}
h1, h2, h3, h4 {
    color: var(--secondary-color);
}
.stButton>button {
    background-color: var(--primary-color);
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: scale(1.05);
    transition: all 0.2s ease-in-out;
}
@media (max-width: 768px) {
    .stButton>button {
        width: 100%;
        padding: 1rem;
    }
    h1, h2, h3 {
        font-size: 1.5rem;
    }
}
.footer {
    text-align: center;
    padding: 10px;
    color: #999;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource()
def load_model():
    try:
        model = tf.keras.models.load_model("Trained_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ========== PREPROCESSING ==========
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        chunk_duration = 4
        overlap_duration = 2
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
        data = []
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1).astype(np.float32)
            mel_spectrogram = tf.image.resize(mel_spectrogram, target_shape)
            data.append(mel_spectrogram)
        return np.array(data)
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {str(e)}")
        return None

# ========== PREDICTION ==========
def model_prediction(X_test):
    model = load_model()
    if model is None:
        return None
    try:
        y_pred = model.predict(X_test)
        predicted_categories = np.argmax(y_pred, axis=1)
        unique_elements, counts = np.unique(predicted_categories, return_counts=True)
        max_elements = unique_elements[counts == np.max(counts)]
        return int(max_elements[0])
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# ========== SIDEBAR NAVIGATION ==========
with st.sidebar:
    selected = option_menu(
        menu_title="SoundSensie",
        options=["Home", "Prediction", "About Project"],
        icons=["house", "headphones", "info-circle"],
        menu_icon="headphones",
        default_index=0
    )

# ========== PAGE: HOME ==========
import os
import streamlit as st

# ... (rest of your code)

if selected == "Home":
    st.markdown("<h1 style='text-align: center;'>🎵 Welcome to SoundSensie</h1>", unsafe_allow_html=True)
    
    # Path to your image file (adjust if needed)
    img_path = os.path.join(os.path.dirname(__file__), "music_genre_home.png")
    
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning("⚠️ Image file 'music_genre_home.png' not found. Please check your file path or upload the image.")
    
    st.markdown("""
    <div style='text-align: center;'>
        <p><strong>SoundSensie</strong> is a powerful music genre classifier.</p>
        <p>🎧 Upload your audio. 🧠 Let AI do the thinking. 🎶 Know your genre.</p>
    </div>
    <hr/>
    ### 🚀 Get Started
    Navigate to the **Prediction** tab and upload your track.

    ### ✅ Why SoundSensie?
    - Deep Learning powered classification
    - Supports `.mp3` and `.wav` formats
    - Fast, accurate, and user-friendly
    """, unsafe_allow_html=True)

# ========== PAGE: PREDICTION ==========
elif selected == "Prediction":
    st.markdown("<h2 style='text-align: center;'>🔍 Genre Prediction</h2>", unsafe_allow_html=True)
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3", "wav"], label_visibility="collapsed")

    if test_mp3:
        filepath = os.path.join("Test_Music", test_mp3.name)
        os.makedirs("Test_Music", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())
        st.success("✅ File uploaded successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.audio(test_mp3, format=test_mp3.type)
        with col2:
            if st.button("🎼 Predict Genre"):
                with st.spinner("⏳ Predicting..."):
                    X_test = load_and_preprocess_data(filepath)
                    if X_test is not None:
                        result_index = model_prediction(X_test)
                        if result_index is not None:
                            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
                                     'metal', 'pop', 'reggae', 'rock']
                            st.balloons()
                            st.markdown(f"<h3 style='color:#FF4B4B;'>🎵 Predicted Genre: <em>{label[result_index]}</em></h3>", unsafe_allow_html=True)

# ========== PAGE: ABOUT ==========
elif selected == "About Project":
    st.markdown("<h2 style='text-align: center;'>📁 About Project</h2>", unsafe_allow_html=True)
    st.markdown("""
    This system classifies music into genres using deep learning.

    ### 📂 Dataset Info
    - **GTZAN Dataset**: 10 genres, 100 audio files each (30 seconds)
    - Genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

    ### 🧠 Model Info
    - Trained on Mel-spectrograms using CNNs
    - Input shape: 150x150 spectrogram images
    - Prediction is based on chunk-wise voting
    """)

    st.markdown("### 📊 Sample Genre Distribution")
    df = {
        "Genre": ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'],
        "Files": [100] * 10
    }
    fig = px.bar(df, x="Genre", y="Files", color="Genre", title="GTZAN Genre Distribution")
    st.plotly_chart(fig)

# ========== FOOTER ==========
st.markdown("""
<div class='footer'>
    Developed with ❤️ by Sarthak Nimbark | © 2025 SoundSensie
</div>
""", unsafe_allow_html=True)
