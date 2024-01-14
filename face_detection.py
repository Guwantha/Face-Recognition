import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Image face detection
img = cv.imread('guwantha.jpg')
cv.imshow('Original', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Faces:{len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow('Detected Faces', img)

cv.waitKey(0) 


# Video Face detection
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)


    for(x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    print(f'Faces:{len(faces_rect)}')

    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()