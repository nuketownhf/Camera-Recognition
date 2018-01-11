import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getBilderUndText(path):
    BilderPfad=[os.path.join(path,f) for f in os.listdir(path)] 
    Bildnummer=[]
    Ids=[]
    for BilderPfad in BilderPfad:
        if(os.path.split(BilderPfad)[-1].split(".")[-1]!='jpg'):
            continue
        pilImage=Image.open(BilderPfad).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(BilderPfad)[-1].split(".")[1])
        gesichter=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in gesichter:
            Bildnummer.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return Bildnummer,Ids

host = '192.168.0.59:8080'
benutzer = 'admin'
passwort = '1234'
cv2.setUseOptimized(True);
cv2.setNumThreads(4);
hoststr = 'http:{0}:{1}@{2}/video'.format(benutzer, passwort, host)
print('Streame ' + hoststr)
gesichter,Ids = getBilderUndText('dataset')
recognizer.train(gesichter, np.array(Ids))
recognizer.write('trainer/trainer.yml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


kamera = cv2.VideoCapture(hoststr)
schrift = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =kamera.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gesichter=faceCascade.detectMultiScale(gray, 1.1,10)
    for(x,y,w,h) in gesichter:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        cv2.putText(im, str(len(gesichter)), (20, 50), schrift, 1, (0,255,255), 2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        if(conf<100):
            if(Id==1):
                Id="Can Karaduman"
                
            elif(Id==2):
                Id="Zeynep Karaduman"
        else:
            Id="Unbekannt"
        
        cv2.putText(im, str(Id), (x,y), schrift, 1, (255,255,255), 3)
    cv2.imshow('im',im) 
    if cv2.waitKey(1) == 27: 
        break
kamera.release()
cv2.destroyAllWindows()
