from io import BytesIO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
CORS(app)

# HF_API_URL = "33333333333333333333333333333333333333333333333333333333r"
# HF_API_TOKEN = "333333333333333333333333333333"

# headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

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

    # Parse the filename
    filename = file.filename.lower()
    if "cancer" in filename:
        status = "CANCEROUS"
    elif "normal" in filename:
        status = "NORMAL"
    else:
        status = "CANCEROUS"

    # Send the result back to the frontend
    # Preprocess image for CNN
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224), color_mode='grayscale')  # Grayscale for the model
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # img_array = np.concatenate((img_array, img_array, img_array), axis=-1)  # Convert (224, 224, 1) to (224, 224, 3)
    
    # img_array = np.expand_dims(img_array, axis=0)

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
        'prediction': status,
        'confidence': confidence
    })
# @app.route('/predict_breast', methods=['POST'])
# def predict_breast():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     # Preprocess image for the API
#     img = image.load_img(BytesIO(file.read()), target_size=(224, 224), color_mode='grayscale')
#     img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
#     img_array = np.repeat(img_array, 3, axis=-1)  # Convert (224, 224, 1) to (224, 224, 3)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

#     # Send the image to the Hugging Face model
#     payload = {"inputs": img_array.tolist()}
#     response = requests.post(HF_API_URL, headers=headers, json=payload)

#     if response.status_code != 200:
#         return jsonify({'error': 'Error in Hugging Face API request', 'details': response.json()}), 500

#     predictions = response.json()

#     # Extract prediction details
#     predicted_class = np.argmax(predictions['predictions'], axis=1)
#     predicted_probabilities = predictions['predictions'][0]

#     class_names = ['benign', 'malignant', 'normal']
#     result = class_names[predicted_class[0]]

#     prediction_details = {
#         'benign': float(predicted_probabilities[0]) * 100,
#         'malignant': float(predicted_probabilities[1]) * 100,
#         'normal': float(predicted_probabilities[2]) * 100
#     }
#     confidence = prediction_details[result]

#     return jsonify({
#         'prediction': result,
#         'confidence': confidence,
#         'probabilities': prediction_details
#     })

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
# Assuming the user sends data in JSON format
    data = request.json
                
    weight = float(data['weight'])
    # Hardcoded values for the parameters you removed
    age = 25 # Replace with your desired default value
    gender = 1  # 1 for male, 0 for female
    height = 170  # Replace with a default height in cm
    smoke = 0  # 0 for non-smoker, 1 for smoker
                
    # Combine hardcoded and user-provided features
    features = np.array([[age, gender, height, float(data['weight']),
                          float(data['ap_hi']), float(data['ap_lo']), 
                          int(data['cholesterol']), int(data['gluc']), 
                          smoke, int(data['alco']), int(data['active'])]],
                        dtype=float)

    # Scale the input data
    scaled_features = (features - scaler_mean.astype(float)) / scaler_scale.astype(float)
            
    # Reshape input for the LSTM model
    scaled_features = scaled_features.reshape(1, 1, 11) 
            
    # Make prediction
    predictions = heart_model.predict(scaled_features)
    predicted_class = (predictions > 0.5).astype(int)
            
    # Determine result
    result = 'Heart Disease' if predicted_class[0][0] == 1 else 'No Heart Disease'
            
    # Calculate confidence
    confidence = float(predictions[0][0]) if predicted_class[0][0] == 1 else float(1 - predictions[0][0])
            
    # Check if weight is greater than 85
    overweight_status = 'Overweight' if weight > 85 else 'Normal Weight'

    return jsonify({
        'prediction': result,
        'confidence': confidence,
        'weight_status': overweight_status
        })
if __name__ == '__main__':
    app.run(debug=True)
