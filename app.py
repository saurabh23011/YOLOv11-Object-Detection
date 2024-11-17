import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time

# Set page configuration
st.set_page_config(
    page_title="Safety-Detection Using Yolo V11",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.5

def load_model():
    """Load YOLOv11 model"""
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model = YOLO('yolo11n.pt')
    return st.session_state.model

def process_image(image, conf_threshold):
    """Process image with YOLOv11 model"""
    model = load_model()
    results = model(image, conf=conf_threshold)
    return results

def draw_boxes(image, results):
    """Draw bounding boxes and labels on the image"""
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get confidence score
            conf = float(box.conf[0])
            # Get class name
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label = f'{cls_name} {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1-20), (x1+w, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 1)
    return image

def main():
    st.title("Industry-Safety-Detection-Using-Computer-Vision")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Main options
    options = ["Upload Image", "Webcam", "Video Upload"]
    choice = st.radio("Select Input Source", options)
    
    if choice == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # Convert uploaded file to image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image_np, channels="RGB", use_column_width=True)
            
            # Process image
            results = process_image(image_np, confidence)
            
            # Draw boxes on image
            output_image = image_np.copy()
            output_image = draw_boxes(output_image, results)
            
            with col2:
                st.subheader("Detected Objects")
                st.image(output_image, channels="RGB", use_column_width=True)
                
    elif choice == "Webcam":
        st.write("Webcam Object Detection")
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access webcam")
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_image(frame, confidence)
            frame = draw_boxes(frame, results)
            FRAME_WINDOW.image(frame)
            
        camera.release()
            
    elif choice == "Video Upload":
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            FRAME_WINDOW = st.image([])
            
            stop_button = st.button("Stop Processing")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = process_image(frame, confidence)
                frame = draw_boxes(frame, results)
                FRAME_WINDOW.image(frame)
                time.sleep(0.1)  # Add small delay to control video speed
                
            cap.release()

if __name__ == '__main__':
    main()