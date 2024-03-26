# Description: This is the main file for the Flask web app. It defines the routes and the prediction function.
import os
import uuid
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Create a Flask web app
covid_ct = Flask(__name__, static_url_path='/static')

# Get the absolute path to the Models directory
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Models'))

# Load the model
model_path = os.path.join(models_dir, 'model.keras')
model1_path = os.path.join(models_dir, 'model1.keras')
model = load_model(model_path)
model1 = load_model(model1_path)

try:
    model = load_model(model_path)
except Exception as e:
    print("Error loading model from:", model_path)
    print(e)
    # Handle the error gracefully, e.g., return an error page or log the error

try:
    model1 = load_model(model1_path)
except Exception as e:
    print("Error loading model from:", model1_path)
    print(e)
    # Handle the error gracefully, e.g., return an error page or log the error

# Define the path to the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

covid_ct.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the home page
@covid_ct.route('/')
def index():
    return render_template('index.html')

@covid_ct.route('/predict', methods=['POST'])

# Define the prediction function
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(covid_ct.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load and preprocess the image for prediction
        img = image.load_img(filename, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make the prediction using your model
        prediction = model.predict(img_array)

        # Convert the prediction to a human-readable label (adjust as per your model's output)
        result_label = "COVID-19 Positive" if prediction > 0.5 else "COVID-19 Negative"

        # Save the result image with a unique filename
        result_image_filename = f'result_{uuid.uuid4()}.jpg'
        result_image_path = os.path.join(covid_ct.config['UPLOAD_FOLDER'], result_image_filename)
        result_img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8))
        result_img.save(result_image_path)

        return render_template('result.html', prediction=result_label, result_image=result_image_filename)

    return redirect(request.url)

# Add this route to serve uploaded images
@covid_ct.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(covid_ct.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == '__main__':
    covid_ct.run(debug=True, use_reloader=False)
