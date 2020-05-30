import cv2
import numpy as np
import time
import os
import imutils

def detectface(img,model):

    img = imutils.resize(img, width=750)

    (h,w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),scalefactor=1.0,size=(300,300),
                                mean=(104.0,177.0,123.0))

    model.setInput(blob)
    detections = model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
    
    face = img[startY:endY,startX:endX]
    coordinates = [startX, startY, endX, endY]

    return coordinates,face

cam = cv2.VideoCapture(0)

count = 0
id = input('Enter the id:')


dirpath = os.path.dirname(__file__)

modelFile = f"{dirpath}/model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = f"{dirpath}/model/deploy.prototxt.txt"
model = cv2.dnn.readNetFromCaffe(configFile, modelFile)

while True:
    
    _,frame = cam.read()
    
    try:
        coordinates, face = detectface(frame,model)
    except:
        print('No face detected')
        continue
    
    writepath = os.path.join('DATA','user')
    cv2.imwrite(f"{dirpath}/{writepath}{id}.{str(count)}.jpg", face)
    startX,startY,endX,endY = [i-100 for i in coordinates]
    cv2.rectangle(frame,(startX+30, startY+30),(endX+30,endY+30),(0,255,0),2)
        
    count += 1
    print(count)

    cv2.imshow("FACE",frame)
    if cv2.waitKey(1) & count==200:
        break
    time.sleep(0.2)

cam.release()
cv2.destroyAllWindows()