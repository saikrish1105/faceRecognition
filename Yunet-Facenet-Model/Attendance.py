import cv2
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import Normalizer
import tensorflow.lite as tflite
import time

# Configuration
FACE_DETECTION_MODEL = r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Models\yunet.onnx"
EMBEDDING_MODEL = r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Models\mobile_facenet.tflite"
INPUT_SIZE = (320, 240)
MIN_CONFIDENCE = 0.7
RECOGNITION_THRESHOLD = 0.5
ATTENDANCE_COOLDOWN = 300  # 5 minutes

class FaceRecognizer:
    def __init__(self):
        self.face_detector = cv2.FaceDetectorYN_create(FACE_DETECTION_MODEL, "", INPUT_SIZE, 0.8, MIN_CONFIDENCE)
        self.embedder = tflite.Interpreter(model_path=EMBEDDING_MODEL)
        self.embedder.allocate_tensors()
        self.input_details = self.embedder.get_input_details()
        self.output_details = self.embedder.get_output_details()
        self.normalizer = Normalizer(norm='l2')
        self.known_faces = self.load_known_faces()
        self.attendance_log = {}

    def load_known_faces(self):
        known = {}
        if not os.path.exists(r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Data\features.csv"):
            print("Run Registration.py first to generate features.csv")
            return known
        df = pd.read_csv(r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Data\features.csv", header=None)
        for _, row in df.iterrows():
            name = row[0]
            embedding = row[1:].values.astype(np.float32)
            if name not in known:
                known[name] = []
            known[name].append(embedding)
        return known

    def detect_faces(self, frame):
        self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.face_detector.detect(frame)
        return faces if faces is not None else []

    def align_face(self, frame, face):
        try:
            landmarks = face[4:14].reshape(5, 2).astype(np.int32)
            eye_left = landmarks[0]
            eye_right = landmarks[1]

            dY = eye_right[1] - eye_left[1]
            dX = eye_right[0] - eye_left[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            center = (face[0] + face[2]//2, face[1] + face[3]//2)
            scale = 1.5

            M = cv2.getRotationMatrix2D(center, angle, scale)
            aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)

            h, w, _ = aligned.shape
            x1, y1 = max(0, face[0]), max(0, face[1])
            x2, y2 = min(w, face[0] + face[2]), min(h, face[1] + face[3])

            face_crop = aligned[int(y1):int(y2), int(x1):int(x2)]
            return face_crop if face_crop.size > 0 else None
        except Exception as e:
            print(f"Alignment error: {e}")
            return None

    def get_embedding(self, face_img):
        try:
            face_resized = cv2.resize(face_img, (112, 112))
            face_resized = face_resized.astype(np.float32) / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            self.embedder.set_tensor(self.input_details[0]['index'], face_resized)
            self.embedder.invoke()
            embedding = self.embedder.get_tensor(self.output_details[0]['index']).flatten()
            return self.normalizer.transform([embedding])[0]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def recognize_face(self, embedding):
        if embedding is None:
            return "Unknown", 0
        min_dist = float('inf')
        identity = "Unknown"
        for name, embeddings in self.known_faces.items():
            distances = np.linalg.norm(embeddings - embedding, axis=1)
            person_min_dist = np.min(distances)
            if person_min_dist < min_dist:
                min_dist = person_min_dist
                identity = name
        confidence = 1 - (min_dist / 2)
        return (identity, confidence) if confidence > RECOGNITION_THRESHOLD else ("Unknown", confidence)

    def mark_attendance(self, name):
        attendance_file = r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Data\features.csv"
        now = time.time()
        if name in self.attendance_log and (now - self.attendance_log[name]) < ATTENDANCE_COOLDOWN:
            return False
        self.attendance_log[name] = now
        df = pd.read_csv(attendance_file) if os.path.exists(attendance_file) else pd.DataFrame(columns=["Name", "Timestamp"])
        new_entry = pd.DataFrame([[name, pd.Timestamp.now()]], columns=["Name", "Timestamp"])
        pd.concat([df, new_entry]).to_csv(attendance_file, index=False)
        return True

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            resized_frame = cv2.resize(frame, INPUT_SIZE)
            faces = self.detect_faces(resized_frame)
            original_height, original_width = frame.shape[:2]
            width_scale = original_width / INPUT_SIZE[0]
            height_scale = original_height / INPUT_SIZE[1]

            for face in faces:
                aligned_face = self.align_face(resized_frame, face)
                if aligned_face is None:
                    continue
                embedding = self.get_embedding(aligned_face)
                if embedding is None:
                    continue
                name, confidence = self.recognize_face(embedding)
                if name != "Unknown":
                    self.mark_attendance(name)

                x = int(face[0] * width_scale)
                y = int(face[1] * height_scale)
                w = int(face[2] * width_scale)
                h = int(face[3] * height_scale)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fr = FaceRecognizer()
    fr.run_camera()