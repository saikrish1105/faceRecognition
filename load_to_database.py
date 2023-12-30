import cv2 
import face_recognition
import os
import pickle 

registered_faces_folder = r'C:\\Users\\TRIDNT\\Desktop\\face_detection\\Registered_faces'

def making_list(registered_faces_folder):
    registered_faces_folder_faces = os.listdir(registered_faces_folder)
    Faces_List=[]
    Faces_names = []
    for face in registered_faces_folder_faces:
        Faces_List.append(cv2.imread(os.path.join(registered_faces_folder,face)))
        Faces_names.append(os.path.splitext(face)[0]) 
    
    print('Encoding Started')
    Known_Encoded_List = encoder(Faces_List)
    print('Encoding Over')
    Known_Encoded_List_With_Name = [Known_Encoded_List,Faces_names]
    print(Faces_names)
    return Known_Encoded_List_With_Name

# print(len(Faces_List))

def encoder(images_list):
    encoded_list = []
    for face in images_list:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB) 
        encode = face_recognition.face_encodings(face)[0]
        encoded_list.append(encode)
    return encoded_list

def save_file():
    file = open('C:\\Users\\TRIDNT\\Desktop\\face_detection\\Encodedfile.p','wb')
    Encode_with_name = making_list(registered_faces_folder)
    print(Encode_with_name)
    pickle.dump(Encode_with_name,file)
    file.close()

    print('File Saved')

save_file()