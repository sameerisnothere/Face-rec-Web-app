import cv2
from mtcnn import MTCNN

# Initialize the MTCNN detector
detector = MTCNN()

# Load and preprocess the image
image = cv2.cvtColor(cv2.imread("image1.jpg"), cv2.COLOR_BGR2RGB)

# Detect faces and facial keypoints
result = detector.detect_faces(image)

if result:
    # Loop through each detected face
    for face_data in result:
        bounding_box = face_data['box']
        keypoints = face_data['keypoints']

        # Draw bounding box around the face
        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)

        # Draw facial keypoints (eyes, nose, mouth)
        cv2.circle(image, keypoints['left_eye'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['right_eye'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['nose'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['mouth_left'], 2, (0, 155, 255), 2)
        cv2.circle(image, keypoints['mouth_right'], 2, (0, 155, 255), 2)

    # Display the annotated image with detected faces and keypoints
    cv2.imshow("Recognized Images", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces detected in the image.")