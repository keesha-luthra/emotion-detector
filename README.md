# 🧠 Emotion Detection System

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/keesha-luthra/emotion-detector?style=for-the-badge)](https://github.com/keesha-luthra/emotion-detector/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/keesha-luthra/emotion-detector?style=for-the-badge)](https://github.com/keesha-luthra/emotion-detector/network)
[![GitHub issues](https://img.shields.io/github/issues/keesha-luthra/emotion-detector?style=for-the-badge)](https://github.com/keesha-luthra/emotion-detector/issues)
[![GitHub license](https://img.shields.io/badge/license-Unlicensed-blue.svg?style=for-the-badge)](LICENSE)

**A multimodal emotion detection system that analyzes Text, Speech, and Facial Expressions.**



</div>

## 📖 Overview

The Emotion Detection System is a comprehensive, interactive application designed to analyze and predict human emotions using a multimodal approach. By combining Text, Speech, and Facial Expression data through a robust weighted fusion logic, the system calculates an accurate final emotion prediction alongside a detailed confidence breakdown. It provides a seamless interface for users to test individual modalities or experience the combined power of all three in real-time.

## ✨ Features

- ✍️ **Text Emotion Detection**: Transformer-based NLP model for accurate emotion analysis.
- 🎤 **Speech Emotion Detection**: Integrates Speech-to-Text with NLP to detect sentiment from voice.
- 📷 **Facial Emotion Detection**: Uses FER and OpenCV to read emotions from facial expressions.
- 🔀 **Multimodal Fusion**: A robust weighted scoring system combining inputs for higher accuracy `(Text + Speech) * 0.7 + (Face) * 0.3`.
- 📊 **Detailed Insights**: Provides confidence scores and a breakdown of predictions.
- 🌐 **Modern UI**: Clean, interactive, and responsive frontend built with vanilla HTML/JS.

## 🛠️ Tech Stack

**Application Framework:**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

## 🚀 Quick Start

Follow these steps to get the Emotion Detection System up and running on your local machine.

### Prerequisites
- **Python 3.8+**: Ensure you have a compatible Python version installed. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/keesha-luthra/emotion-detector.git
   cd emotion-detector
   ```

2. **Install dependencies**
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   pip install -r backend/requirements.txt
   ```

3. **Start the backend API server**
   ```bash
   cd backend
   python app.py
   ```
   The backend API will run on `http://127.0.0.1:5000`.

4. **Start the frontend application**
   Open a new terminal window and navigate to the frontend directory:
   ```bash
   cd frontend
   python -m http.server 8000
   ```

5. **Open your browser**
   Navigate to `http://localhost:8000` to access the main interface.

## 📁 Project Structure

```
emotion-detector/
├── backend/                 # Flask backend API
│   ├── app.py               # Main API routes
│   └── requirements.txt     # Backend-specific dependencies
├── frontend/                # Web interface
│   ├── index.html           # Main application view
│   ├── script.js            # Frontend logic
│   └── assets/              # Styles and static files
├── training/                # Machine learning models and training scripts
│   ├── train_model.py       # Script for training the NLP model
│   ├── evaluation.py        # Script for evaluating model performance
│   └── my_emotion_model_20/ # Saved Hugging Face model weights
├── requirements.txt         # Core environment dependencies
└── README.md                # Project documentation
```

## 🔧 API Reference

### Text Prediction
```http
POST /predict
```
```json
{
  "text": "I feel amazing today!"
}
```

### Face Prediction
```http
POST /predict-face
```
```json
{
  "image": "base64-image"
}
```

### Multimodal Fusion
```http
POST /fuse
```
```json
{
  "text": "I am happy",
  "speech": "this is great",
  "image": "base64-image"
}
```

## 🤝 Contributing

We welcome contributions to enhance the Emotion Detection System! If you have suggestions for new features, bug fixes, or improvements, please feel free to:

1. **Fork the repository.**
2. **Create a new branch** for your feature or bug fix (`git checkout -b feature/your-feature-name` or `bugfix/your-bug-name`).
3. **Implement your changes.**
4. **Commit your changes** with a clear and concise message.
5. **Push to your fork.**
6. **Open a Pull Request** to the `main` branch of this repository.

## 📄 License

This project is currently **Unlicensed**. See the repository for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformer models and datasets.
- Contributors to the `FER` (Facial Expression Recognition) open-source libraries.

## 📞 Support & Contact

- 🐛 Issues: If you encounter any problems or have suggestions, please open an issue on the [GitHub Issues page](https://github.com/keesha-luthra/emotion-detector/issues).

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by [keesha-luthra](https://github.com/keesha-luthra)

</div>
