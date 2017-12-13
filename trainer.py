import cv2

detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
host = '192.168.0.59:8080'
benutzer = 'admin'
passwort = '1234'

hoststr = 'http:{0}:{1}@{2}/video'.format(benutzer, passwort, host)
print('Streame ' + hoststr)

Id=input('ID eingeben \n')
Bildnummer=1
cam = cv2.VideoCapture(hoststr)

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gesichter = detector.detectMultiScale(gray, 1.05, 20)
    for (x,y,w,h) in gesichter:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',img)
        Bildnummer+=1
        cv2.imwrite("dataset/Benutzer."+Id +'.'+ str(Bildnummer) + ".jpg", gray[y:y+h,x:x+w])

        
    if cv2.waitKey(1) == 27: 
        break
    elif Bildnummer>50:
        break
cam.release()
cv2.destroyAllWindows()








