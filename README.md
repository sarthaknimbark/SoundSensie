# 🎵 SoundSensie - Music Genre Classifier

SoundSensie is a deep learning-based web application that classifies music tracks into genres with high accuracy. Built with **Streamlit**, this app uses **mel-spectrograms** and a **convolutional neural network (CNN)** model trained on the **GTZAN dataset** to predict music genres such as:

> 🎸 Blues, 🎼 Classical, 🤠 Country, 💃 Disco, 🎤 Hip-Hop, 🎷 Jazz, 🤘 Metal, 🎧 Pop, 🌴 Reggae, 🎸 Rock

---
## 🔍 Features

- 🎧 Upload `.mp3` or `.wav` music files
- 🧠 Converts audio into mel-spectrograms using Librosa
- 🧩 Splits audio into overlapping chunks for better prediction
- 🤖 Predicts genre using a TensorFlow CNN model
- 📊 Real-time genre prediction and visual feedback
- 💡 Clean, responsive Streamlit UI with audio playback support

## 🛠 Tech Stack

### 🎯 Frontend
- **Streamlit** – For building the interactive web interface
- **HTML/CSS** – Underlying web rendering through Streamlit components

### 🧠 Machine Learning
- **TensorFlow & Keras** – Building and training the Convolutional Neural Network (CNN) model

### 🎵 Audio Processing
- **Librosa** – For extracting audio features and generating mel-spectrograms
- **NumPy** – Efficient numerical operations on audio data

### 📊 Data Visualization
- **Matplotlib** – For spectrogram visualization

### 📁 Dataset
- **GTZAN Genre Dataset** – Benchmark dataset containing 10 genres of 100 audio tracks each

### 🧪 Development Tools
- **Jupyter Notebook / Colab** – For model training and prototyping
- **Python 3.10+** – Primary programming language
- **Anaconda** – Environment management


## 🧠 Model Details

- **Model Type:** CNN (Convolutional Neural Network)
- **Input:** 150x150 mel-spectrogram images
- **Training Data:** 10 genres, 100 tracks per genre (GTZAN)
- **Preprocessing:** 4-second chunks with 2-second overlaps
- **Prediction:** Voting-based approach from chunk-wise predictions
