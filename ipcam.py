import cv2
import numpy as np


host = '192.168.0.59:8080'
benutzer = 'admin'
passwort = '1234'

hoststr = 'http:{0}:{1}@{2}/video'.format(benutzer, passwort, host)
print('Streame ' + hoststr)

aufzeichnung = cv2.VideoCapture(hoststr)
augenerkennung = cv2.CascadeClassifier('objekte/haarcascade_eye.xml')
gesichtserkennung = cv2.CascadeClassifier('objekte/haarcascade_frontalface_default.xml')
while True:

    ret, frame = aufzeichnung.read()
    grau = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gesicht = gesichtserkennung.detectMultiScale(grau, 1.3, 5)

    for (x,y,w,h) in gesicht:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_grau = grau[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        augen = augenerkennung.detectMultiScale(roi_grau)
        for (ax, ay, aw, ah) in augen:
                        cv2.rectangle(roi_color, (ax, ay), (ax+aw, ay+ah), (0,255,0), 2)
                

    cv2.imshow('Gesichtserkennung', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Verlasse IP-Kamera')
aufzeichnung.release()
cv2.destroyAllWindows()
