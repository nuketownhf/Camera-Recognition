import cv2
import numpy as np

host = '192.168.0.59:8080'
benutzer = 'admin'
passwort = '1234'

hoststr = 'http:{0}:{1}@{2}/video'.format(benutzer, passwort, host)
print('Streame ' + hoststr)

aufzeichnung = cv2.VideoCapture(hoststr)

gesichtserkennung = cv2.CascadeClassifier('objekte/haarcascade_frontalface_default.xml')
while True:

	ret, img = aufzeichnung.read()
	grau = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gesicht = gesichtserkennung.detectMultiScale(grau, 1.3, 5)


	for (x,y,w,h) in gesicht:
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

	cv2.imshow('Gesichtserkennung', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print('Verlasse IP-Kamera')
aufzeichnung.release()
cv2.destroyAllWindows()
