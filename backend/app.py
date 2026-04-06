import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort

# ── Paths ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'xray_model.onnx')

# ── App ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load ONNX model ──────────────────────────────────────────
print(f"Loading ONNX model from: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("✅ Model loaded successfully")

TARGET_SIZE = (180, 180)


# ── Preprocessing ────────────────────────────────────────────
def preprocess(file):
    img = Image.open(file).convert('RGB')
    img = img.resize(TARGET_SIZE)
    arr = np.array(img, dtype=np.float32)  # shape: (180, 180, 3)
    arr = np.expand_dims(arr, axis=0)       # shape: (1, 180, 180, 3)
    return arr


# ══ ROUTES ════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model": "xray_model.onnx"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        arr  = preprocess(file)

        # Run inference
        outputs = session.run([output_name], {input_name: arr})
        prediction = outputs[0]  # shape: (1, 2)
        
         # Swapped: output[0]=PNEUMONIA, output[1]=NORMAL
        pneumonia_score = float(prediction[0][1])
        normal_score    = float(prediction[0][0])
 
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
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
 


# ── Entry point ──────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀  http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
