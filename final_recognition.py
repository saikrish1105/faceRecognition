import os
import cv2
import face_recognition #for this you will have to download dlib-19.22.99-cp39-cp39-win_amd64.whl for python9... 
import pickle #install it using pip install 'filepath/dlib-19.22.99-cp39-cp39-win_amd64.whl for python9' and then install face_recognition
import cvzone #pip install cvzone
import numpy as np

file_path = r'C:\\Users\\TRIDNT\\Desktop\\face_detection\\Encodedfile.p' # put your own file path

# Load the data
with open(file_path, 'rb') as file:
        Known_Encoded_List_withnames = pickle.load(file)
Known_Encodings, Face_Names = Known_Encoded_List_withnames

print(Face_Names)

cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame,1)
    imgS = cv2.resize(frame, (0, 0), None, fx=0.25, fy=0.25) #scaling it down to 1/4th 
    img_rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Actual recognition
    cur_face_landmarks = face_recognition.face_locations(img_rgb)
    cur_face_encodings = face_recognition.face_encodings(img_rgb,cur_face_landmarks)

    for encodeFace, FaceLoc in zip(cur_face_encodings, cur_face_landmarks):
        matches = face_recognition.compare_faces(Known_Encodings, encodeFace)
        distance= np.linalg.norm(Known_Encodings-encodeFace,axis=1) #used np.linalg.norm to satisify my ego
        # distance_fr = face_recognition.face_distance(Known_Encodings, encodeFace) #uses the same algorithm as np.linalg.norm
        # print('Matching:', matches) #gives list of true or falses
        # print('Distance With Np', distance)
        # print('Distance with face_recog', distance_fr)

        matching_index = np.argmin(distance)
        # print(Face_Names[matching_index])
        if matches[matching_index]:
            # print(Face_Names[matching_index]) #prints the name of the person
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4,x2 * 4,y2 * 4,x1 * 4 #since we scaled it down to 0.25... we're multiplying by 4 to resize
            bbox = x1,y1,x2-x1,y2-y1
            frame = cvzone.cornerRect(frame,bbox,rt=0)
            cv2.putText(frame, f'{Face_Names[matching_index]} found', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4,x2 * 4,y2 * 4,x1 * 4 #since we scaled it down to 0.25... we're multiplying by 4 to resize
            bbox = x1,y1,x2-x1,y2-y1
            frame = cvzone.cornerRect(frame,bbox,rt=0)
            cv2.putText(frame, f'UnRegistered Person', (x1, y1-10), cv2.FONT_ITALIC, 0.9, (255, 0, 0), 2)
    cv2.imshow('hi',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
