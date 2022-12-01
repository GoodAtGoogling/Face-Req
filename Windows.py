# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:45:16 2022

@author: fondr
"""

import tkinter
from tkinter import messagebox
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join



face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')



window = tkinter.Tk()
window.title("Login form")
window.geometry('640x400')
window.configure(bg='#1c1c7c')

######################################

def face_extractor(img):

    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


######################################"

def face_detector(img, size=0.5):
    
   
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


#####################################

def login():
    username = "Alexis"
    password = "12345"
    if username_entry.get()==username and password_entry.get()==password:
        messagebox.showinfo(title="Login Success", message="You successfully logged in.")
    else:
        messagebox.showerror(title="Error", message="Invalid login.")
 
 #####################################       
def im_get():

    cap = cv2.VideoCapture(0)   
    count = 0
    
    
    while True:
    
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    
            file_name_path = 'C:/Users/fondr/Desktop/Face_detection/face_detection/face/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
    
          
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)
            
        else:
            print("Face not found")
            pass
    
        if cv2.waitKey(1) == ord("q") or count == 100: #13 is the Enter Key
            break
            
    cap.release()
    cv2.destroyAllWindows()      
    print("Collecting Samples Complete")
    
#######################################
    
def model():
    data_path = 'C:/Users/fondr/Desktop/Face_detection/face_detection/face/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    
    
    Training_Data, Labels = [], []
    
    
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    
    
    Labels = np.asarray(Labels, dtype=np.int32)
    
    
    model = cv2.face.LBPHFaceRecognizer_create()
    
    
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained sucessefully")
    
    cap = cv2.VideoCapture(0)
    
    while True:
    
        ret, frame = cap.read()
        
        image, face = face_detector(frame)
        
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
            results = model.predict(face)
            
            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% Confident it is User'
                
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
            
            if confidence > 88:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image )
    
        except:
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            pass
            
        if cv2.waitKey(1) == ord("q"): 
            break
            
    cap.release()
    cv2.destroyAllWindows()     



    
#########################################################

def inp_nb_photo():
    print("Indiquez le nombre de photo : ")
    nb_photo = input()
    print(nb_photo)
    
        

frame = tkinter.Frame(bg='#1c1c7c')

# Creating widgets
login_label = tkinter.Label(
    frame, text="Login", bg='#1c1c7c', fg="#8cccfc", font=("Arial", 30))
username_label = tkinter.Label(
    frame, text="Username", bg='#1c1c7c', fg="#8cccfc", font=("Arial", 16))
username_entry = tkinter.Entry(frame, font=("Arial", 16))
password_entry = tkinter.Entry(frame, show="*", font=("Arial", 16))
password_label = tkinter.Label(
    frame, text="Password", bg='#1c1c7c', fg="#8cccfc", font=("Arial", 16))


login_button = tkinter.Button(
    frame, text="Login", bg="#8cccfc", fg="#FFFFFF", font=("Arial", 16), command=login)


video_button = tkinter.Button(
    frame, text="Vidéo", bg="#8cccfc", fg="#FFFFFF", font=("Arial", 16),command=im_get).grid(row=3, column=2)

model_button = tkinter.Button(
    frame, text="Déverouiller votre PC", bg="#8cccfc", fg="#FFFFFF", font=("Arial", 16),command=model).grid(row=3, column=3)


nombre_button = tkinter.Button(
    frame, text="Nb Photo", bg="#8cccfc", fg="#FFFFFF", font=("Arial", 16),command= inp_nb_photo).grid(row=4, column=3)


# Placing widgets on the screen
login_label.grid(row=0, column=0, columnspan=2, sticky="news", pady=40)
username_label.grid(row=1, column=0)
username_entry.grid(row=1, column=1, pady=20)
password_label.grid(row=2, column=0)
password_entry.grid(row=2, column=1, pady=20)
login_button.grid(row=3, column=0, columnspan=2, pady=30)

frame.pack()

window.mainloop()