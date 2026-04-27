from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fer import FER
import torch
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# ================= MODEL PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "training", "my_emotion_model_20")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ================= FACE MODEL =================
face_detector = FER(mtcnn=True)

# ================= FUSION WEIGHTS =================
TEXT_WEIGHT = 0.7
FACE_WEIGHT = 0.3

# ================= TEMPERATURE =================
TEMPERATURE = 0.7  # 🔥 stabilizes predictions

# ================= TEXT ROUTE =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"emotion": "empty", "confidence": 0.0})

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask")
            )

        # 🔥 temperature scaling
        logits = outputs.logits / TEMPERATURE
        probs = torch.nn.functional.softmax(logits, dim=1)

        predicted_class = torch.argmax(probs, dim=1).item()
        label = model.config.id2label[predicted_class]
        confidence = probs[0][predicted_class].item()

        return jsonify({
            "emotion": label,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= FACE ROUTE =================
@app.route("/predict-face", methods=["POST"])
def predict_face():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"emotion": "no image", "confidence": 0})

        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = face_detector.detect_emotions(img)

        if not results:
            return jsonify({"emotion": "no face", "confidence": 0})

        emotions = results[0]["emotions"]
        face_emotion = max(emotions, key=emotions.get)
        confidence = emotions[face_emotion]

        return jsonify({
            "emotion": face_emotion,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= FUSION ROUTE =================
@app.route("/fuse", methods=["POST"])
def fuse():
    try:
        data = request.json

        text = data.get("text", "").strip()
        speech = data.get("speech", "").strip()
        image_data = data.get("image")

        combined_text = (text + " " + speech).strip()

        scores = {}

        # ---------- TEXT + SPEECH ----------
        if combined_text:
            inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask")
                )

            logits = outputs.logits / TEMPERATURE
            probs = torch.nn.functional.softmax(logits, dim=1)

            pred_class = torch.argmax(probs, dim=1).item()
            label = model.config.id2label[pred_class]
            confidence = probs[0][pred_class].item()

            scores[label] = confidence * TEXT_WEIGHT

        # ---------- FACE ----------
        if image_data:
            image_bytes = base64.b64decode(image_data.split(",")[1])
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results = face_detector.detect_emotions(img)

            if results:
                emotions = results[0]["emotions"]
                face_emotion = max(emotions, key=emotions.get)
                face_conf = emotions[face_emotion]

                if face_emotion in scores:
                    scores[face_emotion] += face_conf * FACE_WEIGHT
                else:
                    scores[face_emotion] = face_conf * FACE_WEIGHT

        # ---------- FINAL ----------
        if not scores:
            return jsonify({"emotion": "no input", "confidence": 0})

        final_emotion = max(scores, key=scores.get)
        final_conf = scores[final_emotion]

        return jsonify({
            "emotion": final_emotion,
            "confidence": round(final_conf, 3),
            "details": scores
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)