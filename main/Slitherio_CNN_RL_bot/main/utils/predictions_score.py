import numpy as np
import cv2
import tensorflow as tf

def predict_score(frame):
    # Preprocess the frame for the model
    processed_frame = preprocess_frame(frame)
    
    # Load the trained model (assuming the model is saved in the same directory)
    model = tf.keras.models.load_model('path_to_your_model.h5')
    
    # Make predictions
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))
    
    # Assuming the model outputs a score directly
    score = predictions[0][0]
    
    return score

def preprocess_frame(frame):
    # Resize the frame to the input size of the model
    frame_resized = cv2.resize(frame, (224, 224))  # Example size, adjust as needed
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    
    return frame_normalized