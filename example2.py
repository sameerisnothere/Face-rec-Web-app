import os
import numpy as np
import cv2
import face_recognition

def extract_face_encodings(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings

def prepare_dataset(data_directory):
    face_encodings = []
    labels = []
    
    for label, person_name in enumerate(os.listdir(data_directory)):
        person_folder = os.path.join(data_directory, person_name)
        if os.path.isdir(person_folder):
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                encodings = extract_face_encodings(image_path)
                if encodings:
                    face_encodings.extend(encodings)
                    labels.extend([person_name] * len(encodings))
    
    return np.array(face_encodings), np.array(labels)

# Path to the directory containing individual folders of images
data_directory = r'C:\Users\GDIT\Desktop\GIKI\6th Semester\CS 354\Face Recognition Project\Assets'

# Prepare dataset
face_encodings, labels = prepare_dataset(data_directory)
