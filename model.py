# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers

# # Function to load and preprocess an image
# def load_and_preprocess_image(image_path, target_size=(100, 100)):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, target_size)
#     image = image / 255.0  # Normalize to [0, 1]
#     return image

# # Function to create pairs of images for Siamese Network
# def create_pairs(data_directory):
#     classes = sorted(os.listdir(data_directory))
#     pairs = []
#     labels = []
    
#     for class_idx, class_name in enumerate(classes):
#         class_folder = os.path.join(data_directory, class_name)
#         images = os.listdir(class_folder)
        
#         for i in range(len(images)):
#             for j in range(i+1, len(images)):
#                 img1_path = os.path.join(class_folder, images[i])
#                 img2_path = os.path.join(class_folder, images[j])
                
#                 img1 = load_and_preprocess_image(img1_path)
#                 img2 = load_and_preprocess_image(img2_path)
                
#                 pairs.append([img1, img2])
#                 labels.append(1 if i == j else 0)  # 1 if same person, else 0
    
#     return np.array(pairs), np.array(labels)

# # Siamese Network model architecture
# def build_siamese_model(input_shape):
#     input_a = tf.keras.Input(shape=input_shape)
#     input_b = tf.keras.Input(shape=input_shape)
    
#     base_network = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu')
#     ])
    
#     processed_a = base_network(input_a)
#     processed_b = base_network(input_b)
    
#     distance = tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([processed_a, processed_b])
#     output = layers.Dense(1, activation='sigmoid')(distance)
    
#     model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
#     return model

# if __name__ == "__main__":
#     # Paths and parameters
#     data_directory = 'Assets'
#     model_path = 'siamese_model.h5'  # Path to save the trained Siamese Network model
    
#     # Create pairs of images and their labels
#     pairs, labels = create_pairs(data_directory)
    
#     # Shuffle and split the data into train and validation sets
#     indices = np.arange(len(pairs))
#     np.random.shuffle(indices)
#     pairs, labels = pairs[indices], labels[indices]
#     split = int(0.8 * len(pairs))
    
#     train_pairs, train_labels = pairs[:split], labels[:split]
#     val_pairs, val_labels = pairs[split:], labels[split:]
    
#     # Build the Siamese Network model
#     input_shape = train_pairs.shape[1:]
#     siamese_model = build_siamese_model(input_shape)
#     siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
#     # Train the Siamese Network
#     siamese_model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels,
#                       validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
#                       epochs=10, batch_size=32)
    
#     # Save the trained Siamese Network model
#     siamese_model.save(model_path)
#     print("Siamese Network trained and saved successfully.")



import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

def load_and_preprocess_image(image_path, target_size=(100, 100)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def create_pairs(data_directory):
    classes = os.listdir(data_directory)
    pairs = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(data_directory, class_name)
        images = os.listdir(class_folder)
        
        for img1_name in images:
            img1_path = os.path.join(class_folder, img1_name)
            img1 = load_and_preprocess_image(img1_path)
            
            for j in range(5):  # Number of positive pairs for each anchor image
                # Choose a random image from the same class
                img2_name = random.choice(images)
                img2_path = os.path.join(class_folder, img2_name)
                img2 = load_and_preprocess_image(img2_path)
                
                pairs.append([img1, img2])
                labels.append(1)  # Positive pair
                
                # Choose a random image from a different class
                negative_class = random.choice(classes)
                while negative_class == class_name:
                    negative_class = random.choice(classes)
                
                negative_folder = os.path.join(data_directory, negative_class)
                negative_images = os.listdir(negative_folder)
                img3_name = random.choice(negative_images)
                img3_path = os.path.join(negative_folder, img3_name)
                img3 = load_and_preprocess_image(img3_path)
                
                pairs.append([img1, img3])
                labels.append(0)  # Negative pair
    
    return np.array(pairs), np.array(labels)

def create_siamese_model(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_network = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(128, activation='relu')
    ])

    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    # Calculate Euclidean distance between the two encodings
    distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1)))([encoded_a, encoded_b])

    siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
    return siamese_model

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

if __name__ == "__main__":
    data_directory = 'Assets'
    pairs, labels = create_pairs(data_directory)

    input_shape = (100, 100, 3)

    siamese_model = create_siamese_model(input_shape)
    siamese_model.compile(optimizer=Adam(learning_rate=0.001), loss=contrastive_loss)


    img1 = pairs[:, 0]
    img2 = pairs[:, 1]

    siamese_model.fit([img1, img2], labels, batch_size=32, epochs=10)

    model_path = 'siamese.h5'
    siamese_model.save(model_path)
    print("Model saved successfully.")
