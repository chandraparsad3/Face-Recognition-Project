import cv2 as cv

img=cv.imread('worldcup.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
har_cascad=cv.CascadeClassifier('face_det.xml')
face_detect=har_cascad.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=0)

print(f'No of face detected:{len(face_detect)}')
for (x,y,w,h) in face_detect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
cv.imshow('Face Detected',img)

cv.waitKey(0)

