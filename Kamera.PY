import cv2
import numpy as np

gesichter,Ids = getBilderUndText('dataset')
recognizer.train(gesichter, np.array(Ids))
recognizer.write('trainer/trainer.yml')
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
host = '192.168.0.59:8080'
benutzer = 'admin'
passwort = '1234'

hoststr = 'http:{0}:{1}@{2}/video'.format(benutzer, passwort, host)
print('Streame ' + hoststr)

kamera = cv2.VideoCapture(hoststr)
schrift = cv2.schrift_HERSHEY_SIMPLEX
while True:
    ret, im =kamera.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gesichter=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in gesichter:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<100):
            if(Id==1):
                Id="Can"
            elif(Id==2):
                Id="Can"
        else:
            Id="Unbekannt"
        
        cv2.putText(im, str(Id), (x,y), schrift, 1, (255,255,255), 3)
    cv2.imshow('im',im) 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
kamera.release()
cv2.destroyAllWindows()
