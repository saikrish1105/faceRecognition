import cv2
import time
import os

name = input("Enter your name")

def register_face(face_img, face_id, save_folder):
    save_path = os.path.join(save_folder, f"{face_id}.jpg")
    cv2.imwrite(save_path, face_img)
    print(f"New Face Registered! Saved as {save_path}")
    
def capture_image():

    registered_faces_folder = r'C:\\Users\\TRIDNT\\Desktop\\face_detection\\Registered_faces'
    if not os.path.exists(registered_faces_folder):
        print(f"Creating folder: {registered_faces_folder}")



    prev_time = 0
    new_time = 0
    
    cam = cv2.VideoCapture(0)
    cam.set(3,700)
    cam.set(4,450)
    

    while True:
        _,frame = cam.read()
        frame = cv2.flip(frame, 1)

        cv2.imshow("Attendance With Face Recognition",frame)


        #fps display
        new_time = time.time()
        fps = 1/(new_time-prev_time) 
        prev_time = new_time

        fps = str(int(fps))

        cv2.putText(frame,fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA) 

        if cv2.waitKey(1) & 0xFF == ord('s'):
            register_face(frame,name,registered_faces_folder)
            print('New Face Registered!!')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


capture_image()