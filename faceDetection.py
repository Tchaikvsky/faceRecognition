from cv2 import cv2

#pre-trained front face data from opencv
#classifier is a detecter and cascade is an algorithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect a face pathing will probably have to be updated on different machines
img = cv2.imread('/Users/rader/Documents/Data Science/AI/images/TexasHouse.jpeg')



#converts to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangles around the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (50,50,150),4)

#print(face_coordinates)


#shows the image
cv2.imshow('Face Detection',img)
#Pauses the running of your code
cv2.waitKey(0)
