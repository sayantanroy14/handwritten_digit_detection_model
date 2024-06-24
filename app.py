from flask import Flask, request, jsonify, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('mnist_model.h5')

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    data = request.get_json()
    img_data = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(img_data)).convert('L')
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28)

    # Predict the digit
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)
