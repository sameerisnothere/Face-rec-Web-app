import os
import numpy as np
import face_recognition
import joblib

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(img_path)
            images.append(image)
    return images

def train_facial_recognition_model(data_dirs):
    face_encodings = []
    labels = []

    for label, data_dir in data_dirs.items():
        images = load_images_from_folder(data_dir)
        for image in images:
            face_encodings_single = face_recognition.face_encodings(image)
            if len(face_encodings_single) > 0:
                face_encoding = face_encodings_single[0]
                face_encodings.append(face_encoding)
                labels.append(label)
            else:
                print(f"No face detected in {image}")

    if len(face_encodings) == 0:
        raise ValueError("No valid face encodings found. Please check your input data.")

    # Convert lists to numpy arrays
    face_encodings = np.array(face_encodings)
    labels = np.array(labels)

    # Train a more advanced classifier (e.g., SVM)
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(face_encodings, labels)

    return classifier

# Directories containing training images for each individual
data_directories = {
    'Abdullah(ABD)': 'Abdullah(ABD)/',
    'Durrani': 'Abdur Rehman Durani/',
    'Sajid': 'Abdur Rehman Sajid/',
    'Adeen':'Adeen Amir/',
    'Affan':'Affan Ali Khan/',
    'Abid':'Ahmad Ali Abid/',
    'Sukhera':'Ahmad Fareed Sukhera/',
    'Ali Inayat':'Ali Inayat/',
    'Arsal': 'Arsal Sheikh/',
    'Basim':'Basim Mehmood/',
    'Eman':'Eman Anjum/',
    'Faizan':'Faizan Haq/',
    'Farwa':'Farwa Toor/',
    'Hammad Anwar':'Hammad Anwar/',
    'Hamza Ahmed Zuberi':'Hamza Ahmed Zuberi/',
    'Hamza Wajid':'Hamza Wajid/',
    'Haya Noor':'Haya Noor/',
    'itba': 'Itba Malahat/',
    'Lailoma Noor': 'Lailoma Noor/',
    'Mia Akbar Jaan': 'Mia Akbar Jaan/',
    'Mujtaba': 'Mujtaba/',
    'Omar khan': 'Omar Khan/',
    'raja':'Raja/',
    'rehan': 'Rehan Riaz/',
    'Saadullah': 'Saadullah/',
    'sameer': 'Sameer Shehzad/',
    'Sheharyar Sadiq': 'Sheharyar Sadiq/',
    'Sherry': 'Sherry/',
    'Syed Ibrahim hamza': 'Syed Ibrahim Hamza/',
    'Talha Wajid': 'Talha Wajid/',
    'Tehrim': 'Tehrim Ahmed/',
    'Umair': 'Umair/',
    'Umer Tayyab': 'Umer Tayyab/',
    'Zaid muzzamil': 'Zaid Bin Muzammil/',
    'Zaid Dandia': 'Zaid Dandia/'
}

# Train the facial recognition model
classifier = train_facial_recognition_model(data_directories)

# Save the trained model
model_output_file = 'facial_recognition_model.joblib'
joblib.dump(classifier, model_output_file)

print(f"Facial recognition model trained successfully and saved to {model_output_file}")
