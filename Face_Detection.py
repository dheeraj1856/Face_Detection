import cv2
import os
import numpy as np

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    images = []
    labels = []
    names = {}
    label = 0
    for image_path in image_paths:
        for image_name in os.listdir(image_path):
            if image_name.startswith('.'):
                continue
            image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(label)
        names[label] = os.path.basename(image_path)
        label += 1
    return images, np.array(labels), names

# Path to the dataset of known faces
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'person_data')

# Load images and labels
images, labels, names = get_images_and_labels(dataset_path)

# Initialize the face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()


# Train the face recognizer
recognizer.train(images, labels)

# Start the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)
        if confidence < 100:  # Confidence threshold can be adjusted
            print(f"Recognized {names[label]} with confidence {confidence}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, names[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
