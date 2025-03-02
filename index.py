import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        # Initialize camera and face recognition
        self.cam = cv2.VideoCapture(0)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Load trained model if exists
        if os.path.exists("trainer.yml"):
            self.recognizer.read("trainer.yml")

        # Names and roles
        self.names = ['None', 'Bharat', 'Mamatha']
        self.roles = {
            'None': 'Unknown person',
            'Bharat': 'Software Engineer - Expert in AI and Computer Vision',
            'Mamatha': 'Data Scientist - Specializes in Machine Learning'
        }

        self.running = False
        self.capture_mode = False

        # Create GUI elements
        self.create_gui()

    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Style configuration
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)
        style.configure("TLabel", font=("Helvetica", 14), background="#2c3e50", foreground="white")

        # Video display frame
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed", padding="5")
        video_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill="both", expand=True)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.pack(fill="x", padx=5, pady=5)

        # Buttons frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=5)

        self.start_btn = ttk.Button(button_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop Detection", command=self.stop_detection, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.capture_btn = ttk.Button(button_frame, text="Capture Face", command=self.start_capture)
        self.capture_btn.pack(side="left", padx=5)

        # User info frame
        info_frame = ttk.LabelFrame(main_frame, text="User Information", padding="5")
        info_frame.pack(fill="x", padx=5, pady=5)

        self.name_label = ttk.Label(info_frame, text="Name: Unknown")
        self.name_label.pack(fill="x", pady=2)

        self.role_label = ttk.Label(info_frame, text="Role: Not identified")
        self.role_label.pack(fill="x", pady=2)

        self.confidence_label = ttk.Label(info_frame, text="Confidence: 0%")
        self.confidence_label.pack(fill="x", pady=2)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Idle")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(fill="x", side="bottom", padx=5, pady=2)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set("Status: Detecting faces...")
            threading.Thread(target=self.detect_faces, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Status: Idle")

    def start_capture(self):
        if not self.capture_mode:
            self.capture_mode = True
            self.capture_btn.config(text="Stop Capture")
            self.status_var.set("Status: Capturing face data...")
            threading.Thread(target=self.capture_faces, daemon=True).start()
        else:
            self.capture_mode = False
            self.capture_btn.config(text="Capture Face")
            self.status_var.set("Status: Training model...")
            self.train_model()

    def detect_faces(self):
        while self.running:
            ret, img = self.cam.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 100:
                    name = self.names[id]
                    confidence_str = f"{round(100 - confidence)}%"
                else:
                    name = "Unknown"
                    confidence_str = f"{round(100 - confidence)}%"

                # Update info labels
                self.name_label.config(text=f"Name: {name}")
                self.role_label.config(text=f"Role: {self.roles.get(name, 'Not identified')}")
                self.confidence_label.config(text=f"Confidence: {confidence_str}")

                # Draw text on image
                cv2.putText(img, name, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(img, confidence_str, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if name in self.roles:
                    cv2.putText(img, self.roles[name], (x-5, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Convert image for Tkinter
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk  # Keep reference

            time.sleep(0.03)  # Control frame rate

    def capture_faces(self):
        id = tk.simpledialog.askstring("Input", "Enter user ID:", parent=self.root)
        if not id:
            self.capture_mode = False
            self.capture_btn.config(text="Capture Face")
            return

        count = 0
        while self.capture_mode and count < 30:
            ret, img = self.cam.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                count += 1
                cv2.imwrite(f"data/{id}.{count}.jpg", gray[y:y+h, x:x+w])

                # Update display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk

            self.status_var.set(f"Status: Capturing face data... {count}/30")
            time.sleep(0.1)

        self.capture_mode = False
        self.capture_btn.config(text="Capture Face")

    def train_model(self):
        path = 'data'
        imagePaths = [os.path.join(path, i) for i in os.listdir(path)]
        faces = []
        ids = []

        for imagePath in imagePaths:
            img = Image.open(imagePath).convert('L')
            imgnp = np.array(img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[0])
            
            detected_faces = self.detector.detectMultiScale(imgnp)
            for (x, y, w, h) in detected_faces:
                faces.append(imgnp[y:y+h, x:x+w])
                ids.append(id)

        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write("trainer.yml")
        self.status_var.set("Status: Model trained successfully!")
        messagebox.showinfo("Success", "Face data trained successfully!")

    def on_closing(self):
        self.running = False
        self.capture_mode = False
        self.cam.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()