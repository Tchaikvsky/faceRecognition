from cv2 import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#trained_face_data.load('/Users/rader/Documents/Python CS 313E/AI/haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    sucessful_frame_read, frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (50,50,250),5)
    
    cv2.imshow('Press q to quit', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()