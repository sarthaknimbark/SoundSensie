# 🎵 SoundSensie — AI-Powered Music Genre Classifier

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
