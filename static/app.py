from io import BytesIO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
CORS(app)

breast_model = load_model('models/breast_cancer_model.keras')
heart_model = load_model('models/cardio_lstm_model.keras')
scaler_mean = np.load('models/scaler_mean.npy')
scaler_scale=np.load('models/scaler_scale.npy')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/breast_cancer')
def breast_cancer():
    return render_template('breast_cancer.html')

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')

@app.route('/predict_breast', methods=['POST'])
def predict_breast():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Send the result back to the frontend
    # Preprocess image for CNN
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224), color_mode='grayscale')  # Grayscale for the model
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]

    img_array = np.repeat(img_array, 3, axis=-1)  # Convert (224, 224, 1) to (224, 224, 3)
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)


    # Make prediction
    predictions = breast_model.predict(img_array)
    predicted_probabilities = predictions[0]
    predicted_class = np.argmax(predictions, axis=1)

    # Map prediction to class names
    class_names = ['benign', 'malignant', 'normal']
    result = class_names[predicted_class[0]]

    prediction_details = {
        'benign': float(predicted_probabilities[0]) * 100,
        'malignant': float(predicted_probabilities[1]) * 100,
        'normal': float(predicted_probabilities[2]) * 100
    }
    confidence = prediction_details[result]

    return jsonify({
        'prediction': prediction_details,
        'confidence': confidence
    })

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
# Assuming the user sends data in JSON format
    data = request.json
    
            
    # Calculate confidence
    confidence = float(predictions[0][0]) if predicted_class[0][0] == 1 else float(1 - predictions[0][0])
            
    return jsonify({
        'prediction': result,
        'confidence': confidence
    })
if __name__ == '__main__':
    app.run(debug=True)
