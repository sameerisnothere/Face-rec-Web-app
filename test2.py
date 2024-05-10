import cv2
from mtcnn import MTCNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import load_and_preprocess_image  # Import image preprocessing function

# Initialize the MTCNN detector
detector = MTCNN()

# Load the Siamese model
siamese_model = load_model('siamese.h5', compile=False)  # Load model without compiling

# Load and preprocess the image
image = cv2.cvtColor(cv2.imread("image1.jpg"), cv2.COLOR_BGR2RGB)

# Detect faces and facial keypoints
result = detector.detect_faces(image)

if result:
    # Load classmates' known images (for comparison)
    classmates = ['Affan', 'John', 'Mary']  # List of classmate names (corresponding to folder names)
    known_embeddings = {}

    for class_name in classmates:
        image_path = f"Assets/{class_name}/sample.jpg"  # Use a sample image for each classmate
        img = load_and_preprocess_image(image_path)  # Preprocess image
        embedding = siamese_model.predict([np.expand_dims(img, axis=0), np.expand_dims(img, axis=0)])
        known_embeddings[class_name] = embedding

    # Loop through each detected face
    for face_data in result:
        bounding_box = face_data['box']

        # Extract face region
        face_img = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                         bounding_box[0]:bounding_box[0] + bounding_box[2]]
        
        # Preprocess extracted face image
        face_img = cv2.resize(face_img, (100, 100))  # Resize to match Siamese model input size
        face_img = face_img.astype(np.float32) / 255.0  # Normalize

        # Obtain face embedding using the Siamese model
        face_embedding = siamese_model.predict([np.expand_dims(face_img, axis=0), np.expand_dims(face_img, axis=0)])

        # Compare face embedding with known embeddings to recognize the face
        min_distance = float('inf')
        recognized_classmate = None

        for class_name in known_embeddings:
            distance = np.linalg.norm(face_embedding - known_embeddings[class_name])
            if distance < min_distance:
                min_distance = distance
                recognized_classmate = class_name

        # Draw bounding box around the face with recognized label
        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)
        cv2.putText(image, recognized_classmate, (bounding_box[0], bounding_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 155, 255), 2)

    # Display the annotated image with recognized faces
    cv2.imshow("Recognized Images", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No faces detected in the image.")
