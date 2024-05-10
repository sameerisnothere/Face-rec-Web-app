import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow import keras

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to detect faces using MTCNN
def detect_faces(image):
    detector = MTCNN()
    result = detector.detect_faces(image)
    return result

# Function to preprocess detected faces
def preprocess_faces(image, face_data_list, target_size=(100, 100)):
    preprocessed_faces = []
    
    for face_data in face_data_list:
        bounding_box = face_data['box']
        x, y, w, h = bounding_box
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        face = face / 255.0
        preprocessed_faces.append(face)
    
    return preprocessed_faces

# Function to recognize faces in the image using Siamese Network
def recognize_faces_in_image(image_path, model, data_directory):
    image = load_and_preprocess_image(image_path)
    face_data_list = detect_faces(image)
    
    if not face_data_list:
        print("No faces detected in the image.")
        return
    
    preprocessed_faces = preprocess_faces(image, face_data_list)
    
    if not preprocessed_faces:
        print("No faces preprocessed.")
        return
    
    recognized_names = []
    classes = sorted(os.listdir(data_directory))
    
    for face in preprocessed_faces:
        known_pairs = []
        
        for class_name in classes:
            class_folder = os.path.join(data_directory, class_name)
            images = os.listdir(class_folder)
            
            for image_name in images:
                image_path = os.path.join(class_folder, image_name)
                known_face = load_and_preprocess_image(image_path)
                known_pairs.append([face, known_face])
        
        similarities = model.predict([np.array(known_pairs)[:, 0], np.array(known_pairs)[:, 1]])
        recognized_idx = np.argmax(similarities)
        recognized_class = classes[recognized_idx]
        recognized_names.append(recognized_class)
    
    annotated_image = image.copy()
    
    for i, face_data in enumerate(face_data_list):
        bounding_box = face_data['box']
        name = recognized_names[i]
        
        cv2.rectangle(annotated_image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)
        
        cv2.putText(annotated_image, name,
                    (bounding_box[0], bounding_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 255), 2)
    
    cv2.imshow("Recognized Faces", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Paths and parameters
    image_path = 'image1.jpg'
    data_directory = 'Assets'
    model_path = 'siamese.h5'  # Siamese Network model
    
    # Load the Siamese Network model
    siamese_model = keras.models.load_model(model_path)
    
    # Perform face recognition on the input image
    recognize_faces_in_image(image_path, siamese_model, data_directory)
