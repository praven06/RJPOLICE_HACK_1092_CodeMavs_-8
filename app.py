import os
import cv2
import tensorflow as tf
import numpy as np
import requests
import subprocess
import smtplib
from email.message import EmailMessage
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import librosa
import joblib
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load your trained model for image detection
image_model = load_model('df_model.h5')

# Load your trained model for audio detection
audio_model = joblib.load('audio_model.joblib')

# Load the pre-trained deep fake detection model for video frames
video_model = load_model('df_model.h5')

UPLOAD_FOLDER = 'backups'  # Change this to your desired upload folder
ALLOWED_EXTENSIONS_IMAGE = {'jpg', 'jpeg', 'png', 'gif', 'jfif'}  # Add allowed image extensions
ALLOWED_EXTENSIONS_AUDIO = {'wav', 'mp3'}  # Add allowed audio extensions
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mkv'}  # Add allowed video extensions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_user_ip():
    # Get the user's IP address
    return request.headers.get('X-Real-IP') or request.remote_addr

def get_user_location(ip_address):
    try:
        api_key = "51d73bc53d3033b8c08d858cf31716b4"  # Replace with your actual API key
        response = requests.get(f"http://ipinfo.io/{ip_address}?token={api_key}")
        data = response.json()
        location = f"{data.get('city', 'Unknown')}, {data.get('region', 'Unknown')}, {data.get('country', 'Unknown')}"
        return location
    except Exception as e:
        print(f"Error fetching location: {str(e)}")
        return "Unknown"

# Function to preprocess the input image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# Function to preprocess video frames
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = tf.keras.preprocessing.image.img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)
        frames.append(frame)

    cap.release()

    return np.vstack(frames)

# Function to extract features from audio file
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms_value = np.sqrt(np.mean(librosa.feature.rms(y=y)))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    roll_off = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_cross_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    feature_mapping = {
        'a': chroma_stft, 'b': rms_value, 'c': spec_centroid,
        'd': spec_bandwidth, 'e': roll_off, 'f': zero_cross_rate,
        'g': mfccs[0], 'h': mfccs[1], 'i': mfccs[2], 'j': mfccs[3],
        'k': mfccs[4], 'l': mfccs[5], 'm': mfccs[6], 'n': mfccs[7],
        'o': mfccs[8], 'p': mfccs[9], 'q': mfccs[10], 'r': mfccs[11],
        's': mfccs[12], 't': mfccs[13], 'u': mfccs[14], 'v': mfccs[15],
        'w': mfccs[16], 'x': mfccs[17], 'y': mfccs[18], 'z': mfccs[19]
    }
    return feature_mapping

# Function to check allowed file extensions for image
def allowed_file_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGE

# Function to check allowed file extensions for audio
def allowed_file_audio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_AUDIO

# Function to check allowed file extensions for video
def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VIDEO

# Send email function
def send_email(user_ip, user_location, prediction_result, image_path=None, audio_path=None, video_path=None):
    email_message = f"User IP: {user_ip}\nUser Location: {user_location}\nPrediction Result: {prediction_result}\n"
    email_message += " "  # Add your email content here
    
    email = EmailMessage()
    email["from"] = "hshri5111@gmail.com"
    email["to"] = "bpraven05@gmail.com"
    email["subject"] = "Deepfake Report"
    email.set_content(email_message)

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        email.add_attachment(image_data, maintype='image', subtype='png', filename='uploaded_image.png')

    if audio_path and os.path.exists(audio_path):
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        email.add_attachment(audio_data, maintype='audio', subtype='mp3', filename='uploaded_audio.mp3')

    if video_path and os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
        email.add_attachment(video_data, maintype='video', subtype='mp4', filename='uploaded_video.mp4')

    with smtplib.SMTP(host="smtp.gmail.com", port=587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login("hshri5111@gmail.com", "rpapqglrwvevjpxh")
        smtp.send_message(email)

def send_email_background(user_ip, user_location, prediction_result, image_path=None, audio_path=None, video_path=None):
    email_thread = threading.Thread(target=send_email, args=(user_ip, user_location, prediction_result, image_path, audio_path, video_path))
    email_thread.start()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/detect_image', methods=['POST'])
def detect_image():
    user_ip = get_user_ip()
    user_location = get_user_location(user_ip)
    
    if 'file' not in request.files:
        return jsonify({"prediction_result": "No file selected.", "user_ip": user_ip, "user_location": user_location})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"prediction_result": "No file selected.", "user_ip": user_ip, "user_location": user_location})

    try:
        if file and allowed_file_image(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            img_array = preprocess_image(image_path)
            image_prediction = image_model.predict(img_array)
            result = "It is a Deepfake photo" if image_prediction > 0.9 else "It is not a Deepfake photo"

            # Perform email sending in the background
            send_email_background(user_ip, user_location, result, image_path=image_path)

            return render_template('result.html', prediction_result=result, user_ip=user_ip, user_location=user_location)
        else:
            return jsonify({"prediction_result": "Invalid file type.", "user_ip": user_ip, "user_location": user_location})
    except Exception as e:
        return jsonify({"prediction_result": f"Error: {str(e)}", "user_ip": user_ip, "user_location": user_location})

# Route to handle audio file upload and prediction
@app.route('/detect_audio', methods=['POST'])
def detect_audio():
    user_ip = get_user_ip()
    user_location = get_user_location(user_ip)
    
    if 'file' not in request.files:
        return jsonify({"prediction_result": "No file selected.", "user_ip": user_ip, "user_location": user_location})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"prediction_result": "No file selected.", "user_ip": user_ip, "user_location": user_location})

    try:
        if file and allowed_file_audio(file.filename):
            filename = secure_filename(file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_path)

            audio_features = extract_audio_features(audio_path)
            audio_prediction = audio_model.predict([list(audio_features.values())])

            # Perform email sending in the background
            send_email_background(user_ip, user_location, f"Audio prediction: {audio_prediction[0]}", audio_path=audio_path)

            return render_template('result.html', audio_prediction_result=audio_prediction[0], user_ip=user_ip, user_location=user_location)
        else:
            return jsonify({"prediction_result": "Invalid file type.", "user_ip": user_ip, "user_location": user_location})
    except Exception as e:
        return jsonify({"prediction_result": f"Error: {str(e)}", "user_ip": user_ip, "user_location": user_location})

# Route to handle video file upload and prediction
@app.route('/detect_video', methods=['POST'])
def detect_video():
    user_ip = get_user_ip()
    user_location = get_user_location(user_ip)
    
    if 'file' not in request.files:
        return jsonify({"prediction_result": "No file selected.", "user_ip": user_ip, "user_location": user_location})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"prediction_result": "No file selected.", "user_ip": user_ip, "user_location": user_location})

    try:
        if file and allowed_file_video(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            video_frames = preprocess_video(video_path)
            video_prediction = video_model.predict(video_frames)
            result = "It is a Deepfake video" if np.any(video_prediction > 0.9) else "It is not a Deepfake video"
            
            # Perform email sending in the background
            send_email_background(user_ip, user_location, prediction_result=result, video_path=video_path)

            return render_template('result.html', video_prediction_result=result, user_ip=user_ip, user_location=user_location)
        else:
            return jsonify({"prediction_result": "Invalid file type.", "user_ip": user_ip, "user_location": user_location})
    except Exception as e:
        return jsonify({"prediction_result": f"Error: {str(e)}", "user_ip": user_ip, "user_location": user_location})

if __name__ == '__main__':
    app.run(debug=True)
