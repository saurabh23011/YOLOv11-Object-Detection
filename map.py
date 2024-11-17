# app.py
from flask import Flask, render_template, request, Response, jsonify
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = '192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'  # Change this
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
camera = None

def load_model():
    """Load YOLOv11 model"""
    global model
    if model is None:
        model = YOLO('yolo11n.pt')
    return model

def process_image(image, conf_threshold=0.5):
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

def generate_frames():
    """Generate frames for webcam feed"""
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_image(frame)
            frame = draw_boxes(frame, results)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_uploaded_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    confidence = float(request.form.get('confidence', 0.5))
    
    # Read and process image
    image_stream = BytesIO(file.read())
    image = Image.open(image_stream)
    image_np = np.array(image)
    
    # Process image with YOLO
    results = process_image(image_np, confidence)
    output_image = image_np.copy()
    output_image = draw_boxes(output_image, results)
    
    # Convert processed image to base64 for display
    output_buffer = BytesIO()
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save(output_buffer, format='JPEG')
    output_base64 = base64.b64encode(output_buffer.getvalue()).decode()
    
    return jsonify({
        'processed_image': f'data:image/jpeg;base64,{output_base64}'
    })

@app.route('/video_feed')
def video_feed():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam')
def start_webcam():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({'status': 'success'})

@app.route('/stop_webcam')
def stop_webcam():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'success'})

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save video temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
    file.save(temp_path)
    
    return jsonify({'video_path': temp_path})

if __name__ == '__main__':
    app.run(debug=True)