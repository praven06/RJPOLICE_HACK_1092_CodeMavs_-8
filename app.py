from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Initial model loading
model = load_model('model.h5')
target_size = (224, 224)

# List to store feedback data
feedback_data = []

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_path = 'uploads/uploaded_image.jpg'
        file.save(image_path)
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)
        result = "Deepfake" if prediction[0, 0] > 0.5 else "Real"
        feedback_data.append((img_array, result))  # Store feedback data
        return jsonify({"prediction": result, "feedback_available": True})

@app.route('/feedback', methods=['POST'])
def feedback():
    if request.method == 'POST':
        feedback = request.form['feedback']
        if feedback.lower() in ['deepfake', 'real']:
            feedback_label = 1 if feedback.lower() == 'deepfake' else 0
            for img_array, _ in feedback_data:
                model.train_on_batch(img_array, np.array([feedback_label]))  # Update model based on feedback
            return jsonify({"message": "Feedback received and model updated successfully"})
        else:
            return jsonify({"message": "Invalid feedback label"})

if __name__ == '__main__':
    app.run(debug=True)
