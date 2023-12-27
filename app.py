from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')

# Load your trained model
model = load_model('df_model.h5')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

UPLOAD_FOLDER = 'D:/c backup/Pictures'  # Change this to your desired upload folder
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif' , 'Jfif'}  # Add allowed extensions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file selected.")

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            if prediction>0.9:
                result="It is a Deepfake photo"
            else: result ="It is not a Deepfake photo"
            os.remove(file_path)  # Remove the uploaded file after processing

            return render_template('index.html', prediction=result)
        else:
            return render_template('index.html', prediction="Invalid file type.")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
