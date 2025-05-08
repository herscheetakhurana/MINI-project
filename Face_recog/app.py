from flask import Flask, render_template, request
import face_recognition
import numpy as np
import os
import cv2
import base64
from datetime import datetime
import pandas as pd

app = Flask(__name__)
KNOWN_FOLDER = 'images'
ATTENDANCE_FOLDER = 'Attendance'

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

# Load known faces
known_images = []
known_names = []

print("\n=== Loading Known Faces ===")

for filename in os.listdir(KNOWN_FOLDER):
    filepath = os.path.join(KNOWN_FOLDER, filename)
    try:
        # Load using OpenCV
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            print(f"[Warning] Cannot read image {filename}. Skipping.")
            continue

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Diagnostics
        print(f"\n[DEBUG] {filename}")
        print(f" - dtype: {img_rgb.dtype}")
        print(f" - shape: {img_rgb.shape}")
        print(f" - min/max pixel: {img_rgb.min()} / {img_rgb.max()}")

        # Check for 8bit 3-channel image
        if img_rgb.dtype != np.uint8 or len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
            print(f"[Error] Image {filename} is not 8-bit 3-channel RGB. Skipping.")
            continue

        # Optional: Re-save to enforce format
        fixed_dir = 'fixed_images'
        os.makedirs(fixed_dir, exist_ok=True)
        cv2.imwrite(os.path.join(fixed_dir, filename), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # Get face encoding
        encodings = face_recognition.face_encodings(img_rgb)
        if len(encodings) == 0:
            print(f"[Warning] No face found in {filename}. Skipping.")
            continue

        known_images.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])
        print(f"[Info] Successfully loaded: {filename} as {known_names[-1]}")

    except Exception as e:
        print(f"[Error] Failed to process {filename}: {e}")

def mark_attendance(name):
    today = datetime.now().strftime('%d-%m-%Y')
    filename = os.path.join(ATTENDANCE_FOLDER, f'Attendance_{today}.csv')
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['Name', 'Time'])

    if name not in df['Name'].values:
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        new_entry = {'Name': name, 'Time': time_string}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(filename, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    status = ""
    if request.method == 'POST':
        image_data = request.form.get('captured_image')
        if not image_data:
            message = "No image captured!"
            status = "error"
            return render_template('index.html', message=message, status=status)

        try:
            # Decode base64
            encoded_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(encoded_data)

            # Convert to OpenCV image
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Validate input image
            if rgb_img.dtype != np.uint8 or len(rgb_img.shape) != 3 or rgb_img.shape[2] != 3:
                message = "Uploaded image is not a valid 8-bit RGB image."
                status = "error"
                return render_template('index.html', message=message, status=status)

            unknown_encodings = face_recognition.face_encodings(rgb_img)

            if len(unknown_encodings) == 0:
                message = "No face detected! Try again."
                status = "error"
            else:
                unknown_encoding = unknown_encodings[0]
                matches = face_recognition.compare_faces(known_images, unknown_encoding)
                face_distances = face_recognition.face_distance(known_images, unknown_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name)
                    message = f"Welcome {name}! Your attendance has been marked."
                    status = "success"
                else:
                    message = "Face not recognized! Please try again."
                    status = "error"

        except Exception as e:
            message = f"Error processing image: {str(e)}"
            status = "error"

    return render_template('index.html', message=message, status=status)

if __name__ == "__main__":
    app.run(debug=True)
