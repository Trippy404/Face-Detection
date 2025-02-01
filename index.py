# 1. pip install opencv-python
# 2. for dowlode cascade file: haarcascade_frontalface_default.xml

import cv2

a = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

b = cv2.VideoCapture(0)

while True:
    c_rec, d_image = b.read()
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    f = a.detectMultiScale(e, 1.3, 6)

    for (x, y, w, h) in f:
        cv2.rectangle(d_image, (x, y), (x+w, y+h), (255, 0, 0), 8)
    
    cv2.imshow('Face Detection', d_image)
    h = cv2.waitKey(40) & 0xff
    if h == 40:
        break

b.release()
cv2.destroyAllWindows()
