import os
import cv2
import numpy as np
import imutils

def detectface(img,net):

    img = imutils.resize(img, width=750)

    (h,w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),scalefactor=1.0,size=(300,300),
                                mean=(104.0,177.0,123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
    
    face = img[startY:endY,startX:endX]
    coordinates = [startX, startY, endX, endY]

    return coordinates,face

def recognizeface(face,rec):
    
    id = 0
    id, sim = rec.predict(gray)
    if sim < 85:
            if id == 1:
                id = 'Rahil {:.2f}'.format(sim)
            elif id == 2:
                id = 'Ankush {:.2f}'.format(sim)
            elif id == 3:
                id = 'Ikram {:.2f}'.format(sim)
    else:
        id = 'unknown'
    
    return id


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    dirpath = os.path.dirname(__file__)
    # print(dirpath)

    modelFile = f"{dirpath}/model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = f"{dirpath}/model/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(f"{dirpath}/recognizer/trainingdata.yml")

    while True:
        _, frame = cap.read()

        try:
            coordinates, face = detectface(frame,net)
        except:
            print('No face detected')
            continue

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        res = recognizeface(gray,rec)

        startX,startY,endX,endY = [i-100 for i in coordinates]


        cv2.rectangle(frame,(startX+30, startY+30),(endX+30,endY+30),(0,255,0),2)
        cv2.putText(frame,res,(startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
        cv2.imshow('frame', frame)
        print(res)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()