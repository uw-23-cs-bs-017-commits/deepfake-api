from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

print("Model load ho raha hai...")
pipe = pipeline(
    "image-classification",
    model="dima806/ai_vs_human_generated_image_detection"
)

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    result = pipe(image)
    label = result[0]["label"]
    score = round(result[0]["score"] * 100, 2)
    return jsonify({
        "label": label,
        "confidence": score,
        "is_ai": label != "REAL",
        "status": "success"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Deepfake Detection API Ready!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
