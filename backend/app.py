import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# ── Paths ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'xray_model.hdf5')

# ── App ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load model ───────────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

TARGET_SIZE = (180, 180)

# ── Serve frontend ───────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ── Health ───────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# ── Predict ──────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        img  = Image.open(file).convert('RGB')
        img  = img.resize(TARGET_SIZE)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction      = model.predict(img_array, verbose=0)
        normal_score    = float(prediction[0][0])
        pneumonia_score = float(prediction[0][1])

        if pneumonia_score > normal_score:
            label      = 'PNEUMONIA'
            confidence = round(pneumonia_score * 100, 2)
        else:
            label      = 'NORMAL'
            confidence = round(normal_score * 100, 2)

        return jsonify({
            "result":     label,
            "confidence": confidence,
            "raw": {
                "normal":    round(normal_score,    4),
                "pneumonia": round(pneumonia_score, 4),
            }
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ── Entry point ──────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀  http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)