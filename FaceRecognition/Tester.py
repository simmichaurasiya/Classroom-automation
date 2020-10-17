import cv2
import os
import numpy as np
from datetime import datetime
import faceRecognition as fr


#Comment belows lines when running this program second time.Since it saves training.yml file in directory
#faces,faceID = fr.labels_for_training_data('trainingImages')
#face_recognizer = fr.train_classifier(faces,faceID)
#face_recognizer.write('trainingData.yml')

#Uncomment below line for subsequent runs


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')                        #use this to load training data for subsequent runs

name={0:"Dayma",1:"Simmi"}                                      #creating dictionary containing names for each label

cap = cv2.VideoCapture(0)

while True:
    success, test_img = cap.read()
    faces_detected, gray_img = fr.faceDetection(test_img)

    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h,x:x+h]
        label,confidence = face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        if(confidence>67):                #If confidence more than 37 then don't print predicted face text on screen
            continue
        fr.put_text(test_img,predicted_name,x,y)
        fr.markAttendance(predicted_name)

    #resized_img=cv2.resize(test_img,(1000,1000))
    cv2.imshow("Attendance System",test_img)
    cv2.waitKey(1)                             #Waits indefinitely until a key is pressed
    #cv2.destroyAllWindows




