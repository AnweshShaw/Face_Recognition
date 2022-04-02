import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_cascade.xml')

people = ['Anwesh Shaw']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread("C:\\Images_library\\Anwesh Shaw\\IMG-20220109-WA0102_3.jpg")

# For resizing the size of the image
def rescale_frame(frame, scale=0.85):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


frame_resize2 = rescale_frame(img)
grey = cv.cvtColor(frame_resize2, cv.COLOR_BGR2GRAY)

# For detecting the face present in the image
faces_found = haar_cascade.detectMultiScale(frame_resize2,scaleFactor=1.1,minNeighbors=4)

for (x, y, w, h) in faces_found:
    faces_roi = grey[y:y + h, x:x + w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(frame_resize2,str(people[label]),(120, 40),cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), thickness=2)
    cv.rectangle(frame_resize2, (x, y), (x + w, y + h), (0,0,255), thickness=3)

cv.imshow('Detected Face', frame_resize2)

cv.waitKey(0)
