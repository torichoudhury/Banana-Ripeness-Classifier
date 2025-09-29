"""
Banana Ripeness Classifier
- Uses a CNN to classify bananas into ripe, overripe, rotten and unripe categories
- Loads dataset from 'D:\Banana Ripeness Classification Dataset'
- Dataset is organized in train, test, and valid folders with class subfolders
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set paths
DATASET_PATH = r"D:\Banana Ripeness Classification Dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
VALID_PATH = os.path.join(DATASET_PATH, "valid")

# Set parameters
IMG_SIZE = 224  # Common size for transfer learning models
BATCH_SIZE = 32
EPOCHS = 20
CLASSES = ["overripe", "ripe", "rotten", "unripe"]
NUM_CLASSES = len(CLASSES)

def create_data_generators():
    """
    Create data generators for train, validation and test sets with augmentation for train set
    """
    print("Creating data generators...")
    
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test sets
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and augment training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load validation data
    valid_generator = valid_test_datagen.flow_from_directory(
        VALID_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load test data
    test_generator = valid_test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, valid_generator, test_generator

def create_model():
    """
    Create a CNN model for banana ripeness classification
    """
    print("Creating the model...")
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')  # 4 classes: overripe, ripe, rotten, unripe
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, valid_generator):
    """
    Train the model and save the best model
    """
    print("Training the model...")
    
    # Create directories to save model and checkpoints
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        "models/banana_ripeness_classifier_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save the final model
    model.save("models/banana_ripeness_classifier_final.h5")
    
    return history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test set and print classification report
    """
    print("Evaluating the model...")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true classes
    y_true = test_generator.classes
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASSES))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.show()

def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss
    """
    print("Plotting training history...")
    
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def predict_image(image_path, model):
    """
    Make prediction for a single image
    """
    from tensorflow.keras.preprocessing import image
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Get class name and probability
    class_name = CLASSES[predicted_class]
    probability = prediction[0][predicted_class]
    
    return class_name, probability, img

def main():
    """
    Main function to train and evaluate the model
    """
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please check the path.")
        return
        
    print(f"Dataset found at {DATASET_PATH}")
    
    # Create data generators
    train_generator, valid_generator, test_generator = create_data_generators()
    
    # Create model
    model = create_model()
    model.summary()
    
    # Train model
    history = train_model(model, train_generator, valid_generator)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, test_generator)
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()