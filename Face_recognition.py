import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Guwantha','Emma Stone', 'Chris Evans']

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


# In a Image
img = cv.imread('both.jpg')
#rez = cv.resize(img, (img.shape[1]//2, img.shape[0]//2))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Person', gray)

# Detect the face in the image

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for(x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label: {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow('Deteted Face', img)

cv.waitKey(0)

# In a Video

# capture = cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 10)

#     for(x,y,w,h) in faces_rect:
#         faces_roi = gray[y:y+h, x:x+w]

#         label, confidence = face_recognizer.predict(faces_roi)
#         #print(f'Label: {people[label]} with a confidence of {confidence}')

#         cv.putText(frame, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
#         cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

#     cv.imshow('Deteted Face', frame)

#     if cv.waitKey(20) & 0xFF==ord('s'):
#         break

# capture.release()
# cv.destroyAllWindows()





