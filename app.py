# app.py

import os
import uuid
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
import torch
from feedback.feedback_generator import FeedbackGenerator, run_full_analysis

app = Flask(__name__, static_folder="ui", template_folder="ui")
CORS(app)  # Enable CORS for cross-origin requests

UPLOAD_FOLDER = os.path.join("ui", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STATIC_IMAGE_FOLDER = "static_images"  # Folder where Untitled.png is placed
os.makedirs(STATIC_IMAGE_FOLDER, exist_ok=True)  # Create if missing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/static-image", methods=["GET"])
def static_image():
    # Serve the local static image Untitled.png
    return send_from_directory(STATIC_IMAGE_FOLDER, "Untitled.png")

@app.route("/analyze", methods=["POST"])
def analyze():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image provided"}), 400

    filename = f"{uuid.uuid4().hex}_{image_file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)

    try:
        _, layout_report, contrast_results, saliency_map = run_full_analysis(filepath, device)
        generator = FeedbackGenerator(model_name="llama3")
        feedback = generator.generate_feedback(layout_report, contrast_results, saliency_map)
        return jsonify({
            "feedback": feedback,
            "image_url": f"/uploads/{filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
