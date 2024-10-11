from flask import Flask, request, jsonify, render_template, send_file,Response,after_this_request
import numpy as np
import cv2
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import time
count = 0
app = Flask(__name__,static_folder='static')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('./models/HAS.weights.h5')
print("file loaded successfully")

facecasc = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 5: "Sad", 6: "Surprised", 4: "Neutral"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/usecase')
def usecase():
    return render_template('usecase.html')

@app.route('/try')
def try_it():
    return render_template('try.html')

@app.route('/patient_therapist/home')
def patient_therapist_home():
    return render_template('patient_home.html')

@app.route('/security/home')
def security_home():
    return render_template('security_home.html')

@app.route('/customer/home')
def customer_home():
    return render_template('customer_home.html')

@app.route('/result')
def result():
    # Retrieve the parameters passed via the URL
    emotion = request.args.get('emotion')
    image_data = request.args.get('image')

    return render_template('result.html', emotion=emotion, image=image_data)

#image processing
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is part of the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in the request."}), 400

    # Read the uploaded image file
    file = request.files['image']
    
    # Convert the image file to a numpy array
    npimg = np.frombuffer(file.read(), np.uint8)
    img_color = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)
    print("No of faces detected:", len(faces))

    # If no faces are detected, return a message
    if len(faces) == 0:
        return jsonify({"message": "No faces detected"}), 200

    # Prepare the response
    emotions = []

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop and resize the face region for emotion prediction
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict emotion
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        print(maxindex)
        
        # Get the emotion label and emoji
        emotion_text = emotion_dict[maxindex]
        # Annotate the image with the emotion
        cv2.putText(img_color, f'{emotion_text}', (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add emotion to response list
        emotions.append({'emotion': emotion_text})

    # Encode the processed image back to base64
    _, img_encoded = cv2.imencode('.png', img_color)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Return the predictions and image as JSON
    return jsonify({'emotions': emotions, 'image': img_base64}),200


# video processing
def preprocess_frame(face_roi):
    face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (if required by model)
    face = cv2.resize(face, (48, 48))  # Resize to match the model input size
    face = face / 255.0  # Normalize (if required by model)
    face = np.reshape(face, (1, 48, 48, 1))  # Reshape to match the model input
    return face

@app.route('/predict-video', methods=['POST'])
def predict_video():
    # Get the video file from the request
    global count 
    count = (count + 1)%100
    file = request.files['video']
    video_path = './'+str(count)+'temp_video.mp4'
    file.save(video_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return jsonify({'error': 'Error opening video file.'}), 400

    # Prepare the output video writer with the same dimensions and FPS
    output_path = './'+str(count)+'output_video.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original video FPS
    frame_width = int(cap.get(3))  # Frame width
    frame_height = int(cap.get(4))  # Frame height
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        faces = facecasc.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=10)

        for (x, y, w, h) in faces:
            # Extract the region of interest (face) from the frame
            face_roi = frame[y:y + h, x:x + w]
            cropped_face = preprocess_frame(face_roi)

            # Predict the emotion using the model
            prediction = model.predict(cropped_face)
            predicted_emotion = np.argmax(prediction)
            emotion_text = emotion_dict.get(predicted_emotion, "Unknown")

            # Draw a rectangle around the face and put the emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

    # Release video resources
    cap.release()
    out.release()
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted file: {video_path}")
        except Exception as e:
            print(f"Error deleting files: {e}")
        return response

    # Send the output video file
    return send_file(output_path, as_attachment=True, mimetype='video/mp4')

@app.before_request
def before_specific_request():
    if request.endpoint == 'predict_video':
        # Delete all output video files
        for file in os.listdir('.'):
            if file.endswith('output_video.mp4'):
                try:
                    os.remove(file)
                    print(f"Deleted file: {file}")
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")

if __name__ == '__main__':
    app.run(debug=True)