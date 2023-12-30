# faceRecognition
My own code of face recognition system using face_recognition software which is based off on tensorflow's facenet.
The file **'save_image.py'** is for capturing a image and storing into your pc as .png
**'load_to_database.py'** reads for all images in that particular path, encodes them and then stores them in a pickle file
**'final_recogntion.py' **loads the data from pickle file and and uses it to check if you're live face is registered or not. 
Uses a method called _'distance'_ which gives the least dissimilarity between images and your image form opencv.
Lower the distance, higher the match. 


