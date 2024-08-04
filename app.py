# app.py

from flask import Flask, request, jsonify, render_template
from app.models import build_image_model
from app.utils import load_text_model, generate_description
import numpy as np
from PIL import Image

app = Flask(__name__)

image_model = build_image_model()
tokenizer, text_model = load_text_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image = Image.open(file).resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    features = image_model.predict(image_array)
    description = generate_description(tokenizer, text_model, features)
    return jsonify({"description": description})

if __name__ == '__main__':
    app.run(debug=True)
