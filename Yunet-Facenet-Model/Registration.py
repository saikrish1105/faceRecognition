import cv2
import numpy as np
import os
import csv
from sklearn.preprocessing import Normalizer
import tensorflow.lite as tflite

# Configuration
FACE_DETECTION_MODEL = r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Models\yunet.onnx"
EMBEDDING_MODEL = r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Models\mobile_facenet.tflite"
INPUT_SIZE = (320, 240)
FACES_DIR = r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Data\faces"
OUTPUT_CSV = r"C:\Users\TRIDNT\Documents\Codes_and_Stuff\Face_Recognition\Final_working_Codes\Data\features.csv"

class FaceRegistrar:
    def __init__(self):
        self.face_detector = cv2.FaceDetectorYN_create(FACE_DETECTION_MODEL, "", INPUT_SIZE, 0.8, 0.7)
        self.embedder = tflite.Interpreter(model_path=EMBEDDING_MODEL)
        self.embedder.allocate_tensors()
        self.input_details = self.embedder.get_input_details()
        self.output_details = self.embedder.get_output_details()
        self.normalizer = Normalizer(norm='l2')

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            return None, None
        
        img_resized = cv2.resize(img, INPUT_SIZE)
        faces = self.detect_faces(img_resized)
        
        # Check if faces is None (invalid scenario) or empty list (no faces)
        if faces is None or len(faces) == 0:  # Check both conditions
            print(f"No faces detected in {img_path}")
            return None, None
        
        try:
            aligned_face = self.align_face(img_resized, faces[0])
        except IndexError:
            print(f"Invalid face data in {img_path}")
            return None, None
        
        if aligned_face is None:
            return None, None
        
        embedding = self.get_embedding(aligned_face)
        if embedding is None:
            return None, None
        
        embedding = self.normalizer.transform([embedding])[0]
        return embedding, os.path.splitext(os.path.basename(img_path))[0]

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
            return self.embedder.get_tensor(self.output_details[0]['index']).flatten()
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def run(self):
        with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for entry in os.listdir(FACES_DIR):
                entry_path = os.path.join(FACES_DIR, entry)
                if os.path.isfile(entry_path) and entry.lower().endswith(('png', 'jpg', 'jpeg')):
                    embedding, name = self.process_image(entry_path)
                    if embedding is not None:
                        writer.writerow([name] + embedding.tolist())
                elif os.path.isdir(entry_path):
                    person_name = os.path.basename(entry_path)
                    for file in os.listdir(entry_path):
                        if file.lower().endswith(('png', 'jpg', 'jpeg')):
                            img_path = os.path.join(entry_path, file)
                            embedding, _ = self.process_image(img_path)
                            if embedding is not None:
                                writer.writerow([person_name] + embedding.tolist())

if __name__ == "__main__":
    registrar = FaceRegistrar()
    registrar.run()