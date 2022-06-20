import cv2 as cv

face_cascade = cv.CascadeClassifier('haar_cascade.xml')
eye_cascade = cv.CascadeClassifier('haar_eye.xml')
capture = cv.VideoCapture(0)

# To use a video as an input
# capture = cv.VideoCapture('C:\\Users\\anwes\\Downloads\\Y2Mate.is - Let Me Down Slowly  Tommy and Grace  Peaky Blinders-8rQz_R6uvsM-240p-1647256925524.mp4')

while True:
    isTrue, frame = capture.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_found= face_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=4)
    eyes_found = eye_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)
    for (x,y,w,h) in faces_found:
        cv.rectangle(frame,(x, y),(x+w,y+h),(255,0,0),thickness=3)
    # for (x,y,w,h) in eyes_found:
        # cv.rectangle(frame,(x, y),(x+w,y+h),(0,255,0),thickness=2)
    cv.imshow('Face Detection in Live Web-Cam', frame)
    if cv.waitKey(20) & 0xFF == ord('e'):# Pressing e closes the video player
        break

capture.release()
cv.destroyAllWindows()
