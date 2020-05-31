import cv2
import numpy as np
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
        id = 'Unknown'
    
    return id


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    modelFile = r"F:\dnn facerec\model\res10_300x300_ssd_iter_140000.caffemodel"
    configFile = r"F:\dnn facerec\model\deploy.prototxt.txt"
    model = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(r"recognizer\trainingdata.yml")

    while True:
        _, frame = cap.read()

        try:
            coordinates, face = detectface(frame,model)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        except:
            print('No face detected')
            continue

        res = recognizeface(gray,rec)

        print(res)

    cv2.destroyAllWindows()
    cap.release()