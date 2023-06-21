import cv2                        
import numpy as np                
from os import listdir            
from os.path import isfile,join
import pyttsx3

k = pyttsx3.init()
sound = k.getProperty('voices')
k.setProperty('voice',sound[0].id)
k.setProperty('rate',130)
k.setProperty('pitch',200)


def speak(text):
    k.say(text)
    k.runAndWait()



data_path = r"C:\Users\pilli\OneDrive\Documents\Face_Recognition\sample/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data,Labels = [],[]

for i,files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels,dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data),np.asarray(Labels))
print("... *_*...Congratulations model is TRAINED ... *_*...")

face_classifier = cv2.CascadeClassifier(r"C:\Users\pilli\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\cv2\data\haarcascade_frontalface_default.xml")

def face_detector(img,size = 0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi,(200,200))

    return img,roi

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        result = model.predict(face)

        if result[1] < 500:
            Confidence = int(100 * (1 - (result[1])/300))
            display_string = str(Confidence)+'% edi nv ey'
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

            if Confidence > 78:
                cv2.putText(image, "Namastey annao", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face Cropper", image)

            else:
                cv2.putText(image, "CAN'T RECOGNISE", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Face Cropper", image)

        else:
            cv2.putText(image, "CAN'T RECOGNISE", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Cropper", image)

        cv2.waitKey(1)  # move cv2.waitKey() inside the try block

    except:
        speak("face not found")
        cv2.putText(image, "Mokam chupiya Radeyy sigga..?", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Face Cropper", image)
        cv2.waitKey(1)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
