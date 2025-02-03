import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from PIL import Image

# Constants
img_size = 128  
frames_per_video = 25  

# Load Model
model = tf.keras.models.load_model('deepfake_detection_model.h5')

def extract_frames(video_path, max_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, total_frames // max_frames)

    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def predict_video(video_path):
    frames = extract_frames(video_path, frames_per_video)
    if len(frames) == 0:
        return None, None
    frames = np.array(frames) / 255.0 
    predictions = model.predict(frames)
    
    avg_prediction = np.mean(predictions, axis=0)
    final_prediction = np.argmax(avg_prediction)
    return final_prediction, avg_prediction

def predict_image(image):
    image = image.resize((img_size, img_size))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    final_prediction = np.argmax(prediction)
    return final_prediction, prediction

def prediction_to_label(prediction):
    return 'Fake' if prediction == 0 else 'Real'

# Streamlit UI
st.title("Deepfake Detection (Videos & Images)")
st.write("Upload a video or image to check if it's real or fake.")

uploaded_file = st.file_uploader("Choose a video or image file", type=["mp4", "avi", "mov", "jpg", "png"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "video":
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
        
        st.video(temp_video_path)
        st.write("Processing video...")
        
        final_prediction, avg_prediction = predict_video(temp_video_path)
        
    elif file_type == "image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        final_prediction, avg_prediction = predict_image(image)
    
    if final_prediction is not None:
        label = prediction_to_label(final_prediction)
        st.write(f'**Prediction:** {label}')
        st.write(f'**Confidence scores (Fake, Real):** {avg_prediction}')
    else:
        st.write("Error processing file. Please try again.")
