import cv2
import numpy as np
from PIL import Image
import os
import pyttsx3  # Import text-to-speech library

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

path = 'data'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def imgsandlables(path):
    imagePaths = [os.path.join(path, i) for i in os.listdir(path)]
    indfaces = []
    ids = []
    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')  # grayscale
        imgnp = np.array(img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        
        faces = detector.detectMultiScale(imgnp)
        for (x, y, w, h) in faces:
            indfaces.append(imgnp[y:y + h, x:x + w])
            ids.append(id)
    return indfaces, ids

faces, ids = imgsandlables(path)
recognizer.train(faces, np.array(ids))

names = ['None', 'Bharat', 'Mamatha','Prasad','Aravind']
roles = {
    'None': 'Unknown person',
    'Bharat': 'Software Engineer - Expert in AI and Computer Vision',
    'Mamatha': 'Data Scientist - Specializes in Machine Learning',
    'Prasad' : 'Full Stack Developer',
    'Aravind': 'Kdl Intern'

}

cam = cv2.VideoCapture(0)
current_id = None  # Track current person being recognized

while True:
    _, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detected_id = None
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        
        if confidence < 100:
            name = names[id]
            confidence_text = "  {0}%".format(round(100 - confidence))
            detected_id = name  # Store detected person's name
        else:
            name = "Unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))
        
        role_text = roles.get(name, "")
        
        cv2.putText(img, name, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img, role_text, (x - 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if detected_id:
        if detected_id != current_id:
            speak(f"Hello {detected_id}, {roles.get(detected_id, '')} Welcome To F9 17th Anniversary")
        current_id = detected_id  # Update last detected ID
    else:
        current_id = None  # Reset when no faces are detected

    cv2.imshow('camera', img)
    
    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break

cam.release()
cv2.destroyAllWindows()

