import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import pathlib
import pickle  # Added for saving training history

# -------------------------------
# Use absolute paths
# -------------------------------
base_dir = r"D:\Banana Ripeness Classification Dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

print(f"Training directory: {train_dir}")
print(f"Validation directory: {val_dir}")
print(f"Test directory: {test_dir}")

# -------------------------------
# Parameters (for full training)
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32     # Good balance for training speed and memory
EPOCHS_HEAD = 10    # Increased epochs for better initial training
EPOCHS_FINE = 15    # Increased fine-tuning epochs for full training

# -------------------------------
# Data Generators with correct preprocessing
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Use all available training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------
# Class Weights for imbalance
# -------------------------------
class_weights = class_weight.compute_class_weight(
    "balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -------------------------------
# Build EfficientNetB0 model
# -------------------------------
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Stage 1: freeze backbone

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

# Compile (Stage 1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Create project directory for saving outputs
project_dir = pathlib.Path(__file__).parent.absolute()
model_dir = os.path.join(project_dir, "models")
history_dir = os.path.join(project_dir, "history")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

# -------------------------------
# Stage 1: Train classifier head
# -------------------------------
print("\n--- Stage 1: Training classifier head ---\n")
history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    verbose=1
)

# -------------------------------
# Stage 2: Fine-tune backbone
# -------------------------------
print("\n--- Stage 2: Fine-tuning EfficientNet ---\n")
base_model.trainable = True  # unfreeze

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    verbose=1
)

# -------------------------------
# Save model
# -------------------------------
model_path = os.path.join(model_dir, "banana_ripeness_efficientnet.keras")
model.save(model_path)
print(f"Model saved to: {model_path}")

# -------------------------------
# Save training history
# -------------------------------
# Combine both training histories
full_history = {
    'head_training': {
        'accuracy': history_head.history['accuracy'],
        'loss': history_head.history['loss'],
        'val_accuracy': history_head.history['val_accuracy'],
        'val_loss': history_head.history['val_loss']
    },
    'fine_tuning': {
        'accuracy': history_fine.history['accuracy'],
        'loss': history_fine.history['loss'],
        'val_accuracy': history_fine.history['val_accuracy'],
        'val_loss': history_fine.history['val_loss']
    }
}

# Save history as pickle file
history_path = os.path.join(history_dir, "training_history.pkl")
with open(history_path, 'wb') as file:
    pickle.dump(full_history, file)
print(f"Training history saved to: {history_path}")

# Save training configuration for reference
config = {
    'img_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'epochs_head': EPOCHS_HEAD,
    'epochs_fine': EPOCHS_FINE,
    'class_weights': class_weights,
    'model_architecture': 'EfficientNetB0',
    'num_classes': train_generator.num_classes,
    'class_indices': train_generator.class_indices
}

config_path = os.path.join(history_dir, "training_config.pkl")
with open(config_path, 'wb') as file:
    pickle.dump(config, file)
print(f"Training configuration saved to: {config_path}")

# -------------------------------
# Evaluate the model
# -------------------------------
print("\n--- Model Evaluation ---\n")

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Get predictions
print("Generating predictions for confusion matrix...")
test_generator.reset()
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = test_generator.classes

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Generate classification report
report = classification_report(
    y_true, 
    y_pred_classes,
    target_names=class_labels,
    digits=4
)
print("\nClassification Report:")
print(report)

# Save evaluation results
evaluation_results = {
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'confusion_matrix': cm,
    'classification_report': report,
    'y_true': y_true,
    'y_pred_classes': y_pred_classes
}

eval_path = os.path.join(history_dir, "evaluation_results.pkl")
with open(eval_path, 'wb') as file:
    pickle.dump(evaluation_results, file)
print(f"Evaluation results saved to: {eval_path}")

print("\n✅ Training completed successfully.")