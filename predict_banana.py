"""
Banana Ripeness Prediction Script
- Uses the trained model to predict the ripeness of a single banana image
"""

import os
import sys
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from banana_ripeness_classifier import predict_image

def main():
    # Check if model exists
    model_path = "models/banana_ripeness_classifier_best.h5"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        sys.exit(1)
    
    # Load model
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Get image path from command line
    if len(sys.argv) < 2:
        print("Please provide an image path")
        print("Usage: python predict_banana.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        sys.exit(1)
    
    # Make prediction
    class_name, probability, img = predict_image(image_path, model)
    
    # Display results
    print(f"Prediction: {class_name}")
    print(f"Confidence: {probability:.2%}")
    
    # Display the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {class_name} ({probability:.2%})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()