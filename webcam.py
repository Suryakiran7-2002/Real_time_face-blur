import cv2 as cv
import numpy as np


vid = cv.VideoCapture(0)

while True:

    r,img = vid.read()

    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


    haar_cas = cv.CascadeClassifier('face_cascade.xml')

    faces = haar_cas.detectMultiScale(grey)

    img_blr = cv.GaussianBlur(img,(51,51),0)

    for x,y,w,h in faces:
        img_blr_crop = img_blr[y:y+w,x:x+w]

        img[y:y+w,x:x+w] = img_blr_crop

    cv.imshow('webcam',img)
    if cv.waitKey(1) & 0xFF == 27:
        break


