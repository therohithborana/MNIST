from flask import Flask, render_template, request, jsonify
from inference import load_model, predict_image
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model, device = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream).convert('L').resize((28, 28))
    prediction = predict_image(image, model, device)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True) 