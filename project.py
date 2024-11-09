import face_recognition
import os
import cv2
from mtcnn import MTCNN
import sys
import xlsxwriter

sys.stdout.reconfigure(encoding='utf-8')

print(os.getcwd())

def load_and_encode_samples(folder_path, scale_factor=1.0):
    samples = {}  # Dictionary to store person name to face encoding mapping

    # Iterate over each subfolder (person) in the training directory
    for person_folder in sorted(os.listdir(folder_path)):
        person_folder_path = os.path.join(folder_path, person_folder)

        if os.path.isdir(person_folder_path):
            # Initialize list to store face encodings for this person
            person_encodings = []

            # Iterate over each image file in the person's folder
            for filename in sorted(os.listdir(person_folder_path)):
                image_path = os.path.join(person_folder_path, filename)
                try:
                    # Load the image using OpenCV
                    image = cv2.imread(image_path)
                    
                    # Resize the image if scale_factor is provided
                    if scale_factor != 1.0:
                        image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
                    
                    # Convert to RGB format for face recognition
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in the image
                    face_locations = face_recognition.face_locations(rgb_image)
                    
                    if len(face_locations) > 0:
                        # Encode all faces found in the image
                        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        person_encodings.extend(face_encodings)
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
            
            # Store the list of face encodings for this person in the dictionary
            if person_encodings:
                samples[person_folder] = person_encodings
            else:
                print(f"No valid face found in images for {person_folder}")

    return samples


# Example usage:
train_dir = r'C:\Users\samee.WINDOWS-2L0L316\OneDrive\Desktop\DL project\DL_Dataset\Train1'  # Update with the path to your Train1 directory
sample_encodings = load_and_encode_samples(train_dir, scale_factor=1.5)

# Display the number of loaded samples
print(f"Number of loaded samples: {sum(len(encodings) for encodings in sample_encodings.values())}")

# Initialize the MTCNN detector
detector = MTCNN()

def main(image_path):
    try:
        # Load and preprocess the target image (image1.jpg)
        target_image_path = image_path # Update with the path to your target image
        print("target image path: " + target_image_path)
        target_image = cv2.imread(target_image_path)
        print("code reached after cv2.imread")
        if target_image is None:
            print("Error: Unable to load the image.")
            return
        target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        print("code reached after cv2.cvtColor")
        # Detect faces using MTCNN
        result = detector.detect_faces(target_image_rgb)
        print("code reached after 3")
        # Initialize lists to store face locations and encodings
        face_locations = []
        face_encodings = []

        if result:
            # Extract face locations from MTCNN results
            for face_data in result:
                bounding_box = face_data['box']
                face_locations.append((bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3], bounding_box[0]))

            # Extract face encodings from the target image
            face_encodings = face_recognition.face_encodings(target_image_rgb, face_locations)
        print("code reached after 4")
        # Load and encode sample faces from the Train1 dataset
        def load_and_encode_samples(folder_path):
            samples = {}  # Dictionary to store person name to face encoding mapping

            # Iterate over each subfolder (person) in the training directory
            for person_folder in sorted(os.listdir(folder_path)):
                person_folder_path = os.path.join(folder_path, person_folder)

                if os.path.isdir(person_folder_path):
                    # Initialize list to store face encodings for this person
                    person_encodings = []

                    # Iterate over each image file in the person's folder
                    for filename in sorted(os.listdir(person_folder_path)):
                        image_path = os.path.join(person_folder_path, filename)
                        try:
                            # Load the image using face_recognition (RGB format)
                            image = face_recognition.load_image_file(image_path)
                            
                            # Detect faces in the image
                            face_locations = face_recognition.face_locations(image)
                            
                            if len(face_locations) == 1:
                                # Encode the face (assuming one face per image)
                                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                                
                                # Append the face encoding to the list
                                person_encodings.append(face_encoding)
                        except Exception as e:
                            print(f"Error processing image {filename}: {str(e)}")
                    
                    # Store the list of face encodings for this person in the dictionary
                    if person_encodings:
                        samples[person_folder] = person_encodings
                    else:
                        print(f"No valid face found in images for {person_folder}")

            return samples

        # Path to the extracted Train1 directory
        train_dir = r'C:\Users\samee.WINDOWS-2L0L316\OneDrive\Desktop\DL project\DL_Dataset\Train1'  # Update with the path to your Train1 directory

        # Load and encode sample faces from the Train1 dataset
        sample_encodings = load_and_encode_samples(train_dir)
        print("code reached after 5")
        # Initialize a dictionary to store recognized names for each face
        recognized_names = {}

        # Compare each detected face encoding with sample face encodings
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Initialize the recognized name as 'Unknown' by default
            recognized_name = 'Unknown'

            # Initialize minimum distance to a large value
            min_distance = float('inf')

            # Compare the face encoding with sample face encodings
            for name, encodings_list in sample_encodings.items():
                for sample_encoding in encodings_list:
                    # Calculate the distance between the detected face and the sample face
                    distance = face_recognition.face_distance([sample_encoding], face_encoding)[0]

                    # Update recognized name if distance is below threshold
                    if distance < min_distance:
                        min_distance = distance
                        recognized_name = name

            # Store recognized name for this face
            recognized_names[(top, right, bottom, left)] = recognized_name
            print("code reached after 6")
            # Print recognized name along with bounding box coordinates
            print(f"Recognized: {recognized_name}, Bounding Box: (Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left})")

            # Draw rectangle around the face
            cv2.rectangle(target_image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw label with name below the face
            label_size, _ = cv2.getTextSize(recognized_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(target_image, (left, bottom - label_size[1] - 10), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(target_image, recognized_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        workbook = xlsxwriter.Workbook('attendance.xlsx')
        worksheet = workbook.add_worksheet()

        # Write the headers to the Excel file
        worksheet.write(0, 0, 'Present')
        worksheet.write(0, 1, 'Bounding Box (Top)')
        worksheet.write(0, 2, 'Bounding Box (Right)')
        worksheet.write(0, 3, 'Bounding Box (Bottom)')
        worksheet.write(0, 4, 'Bounding Box (Left)')

        # Write the recognized names and bounding box coordinates to the Excel file
        row = 1
        for face_location, recognized_name in recognized_names.items():
            top, right, bottom, left = face_location
            worksheet.write(row, 0, recognized_name)
            worksheet.write(row, 1, top)
            worksheet.write(row, 2, right)
            worksheet.write(row, 3, bottom)
            worksheet.write(row, 4, left)
            row += 1

        # Close the workbook
        workbook.close()
        # Display the annotated image with recognized faces
        cv2.imshow("Recognized Faces", target_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("main function started")
    # Check if an image path is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python facial_recognition_script.py <image_path>")
        sys.exit(1)

    # Get the image path from command-line arguments
    image_path = sys.argv[1].encode('utf-8').decode('unicode_escape')
    print("image path: " + image_path)

    # Call the main function with the image path
    main(image_path)
