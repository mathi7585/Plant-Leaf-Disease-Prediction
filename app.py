from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('/Users/mathim/Downloads/leaf/models/disease_model.h5')


# Define disease classes and treatments
disease_classes = {
        0: {'name': 'Pepper__bell___Bacterial_spot', 'treatment': 'Copper-Fungicide'},
        1: {'name': 'Pepper__bell___healthy', 'treatment': 'Copper-Fungicide'},
        2: {'name': 'Potato___Early_blight', 'treatment': 'Fungicide'},
        3: {'name': 'Potato___Late_blight', 'treatment': 'Fungicide'},
        4: {'name': 'Potato___healthy', 'treatment': 'Fungicide'},
        5: {'name': 'Tomato_Bacterial_spot', 'treatment': 'Copper-Spray'},
        6: {'name': 'Tomato_Early_blight', 'treatment': 'Copper-Spray'},
        7: {'name': 'Tomato_Late_blight', 'treatment': 'Copper-Spray'},
        8: {'name': 'Tomato_Leaf_Mold', 'treatment': 'Copper-Spray'},
        9: {'name': 'Tomato_Septoria_leaf_spot', 'treatment': 'Copper-Spray'},
        10: {'name': 'Tomato_Spider_mites_Two_spotted_spider_mite', 'treatment': 'Copper-Spray'},
        11: {'name': 'Tomato__Target_Spot', 'treatment': 'Copper-Spray'},
        12: {'name': 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'treatment': 'Copper-Spray'},
        13: {'name': 'Tomato__Tomato_mosaic_virus', 'treatment': 'Copper-Spray'},
        14: {'name': 'Tomato_healthy', 'treatment': 'Copper-Spray'}
}

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))  # Assuming input size of 224x224
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict the disease
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        disease_info = disease_classes[class_index]

        return render_template('result.html', disease=disease_info['name'], treatment=disease_info['treatment'], image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
