import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import tensorflow as tf
# Configure TensorFlow for CPU optimization
tf.config.set_visible_devices([], 'GPU')  # Disable GPU
tf.config.threading.set_intra_op_parallelism_threads(4)  # Optimize CPU threads
tf.config.threading.set_inter_op_parallelism_threads(4)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Use absolute paths
# -------------------------------
base_dir = r"D:\Banana Ripeness Classification Dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")  # Added test directory

print(f"Training directory: {train_dir}")
print(f"Validation directory: {val_dir}")
print(f"Test directory: {test_dir}")

# -------------------------------
# Parameters (optimized for CPU)
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32   # Reduced for CPU efficiency
EPOCHS_HEAD = 5     # stage 1
EPOCHS_FINE = 2     # stage 2

print("\n" + "="*60)
print("RUNNING ON CPU - Optimized Configuration")
print("="*60)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Image Size: {IMG_SIZE}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Available devices: {tf.config.list_physical_devices()}")
print("="*60 + "\n")

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
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # No augmentation for test data

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
    shuffle=False  # Don't shuffle validation for confusion matrix
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # Don't shuffle test data for consistent evaluation
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

# -------------------------------
# Build MobileNetV2 model
# -------------------------------
base_model = MobileNetV2(
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
print("\n--- Stage 2: Fine-tuning MobileNetV2 ---\n")
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
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "banana_ripeness_mobilenet.keras")
model.save(model_path)

# -------------------------------
# Visualize Training History
# -------------------------------
print("\n--- Generating Training History Plots ---\n")

# Create directory for plots
plots_dir = os.path.join(base_dir, "plots_mobilenet")
os.makedirs(plots_dir, exist_ok=True)

# Combine histories from both stages
combined_history = {
    'accuracy': history_head.history['accuracy'] + history_fine.history['accuracy'],
    'val_accuracy': history_head.history['val_accuracy'] + history_fine.history['val_accuracy'],
    'loss': history_head.history['loss'] + history_fine.history['loss'],
    'val_loss': history_head.history['val_loss'] + history_fine.history['val_loss']
}

# Plot training & validation accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(combined_history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(combined_history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.axvline(x=EPOCHS_HEAD-1, color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(combined_history['loss'], label='Training Loss', marker='o')
plt.plot(combined_history['val_loss'], label='Validation Loss', marker='s')
plt.axvline(x=EPOCHS_HEAD-1, color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
history_plot_path = os.path.join(plots_dir, "training_history.png")
plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
print(f"Training history plot saved to: {history_plot_path}")
plt.show()


# -------------------------------
# Evaluate on validation data
# -------------------------------
print("\n--- Validation Set Evaluation ---\n")
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# -------------------------------
# Calculate validation confusion matrix
# -------------------------------
print("\n--- Validation Set Confusion Matrix ---\n")

# Reset the validation generator to ensure we get all samples
val_generator.reset()

# Get predictions
val_Y_pred = model.predict(val_generator)
val_y_pred = np.argmax(val_Y_pred, axis=1)

# Get true labels
val_steps = len(val_generator)
val_y_true = []

val_generator.reset()
for i in range(val_steps):
    _, y_batch = next(val_generator)
    val_y_true.extend(np.argmax(y_batch, axis=1))

# Trim to match prediction length (in case of incomplete last batch)
val_y_true = val_y_true[:len(val_y_pred)]

# Calculate confusion matrix
val_cm = confusion_matrix(val_y_true, val_y_pred)
print("\nValidation Confusion Matrix:")
print(val_cm)

# Get class names for better interpretation
class_names = list(train_generator.class_indices.keys())
print("\nClass Indices:", train_generator.class_indices)

# -------------------------------
# Visualize Validation Confusion Matrix
# -------------------------------
print("\n--- Generating Validation Confusion Matrix Plot ---\n")

plt.figure(figsize=(10, 8))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Validation Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
val_cm_plot_path = os.path.join(plots_dir, "validation_confusion_matrix.png")
plt.savefig(val_cm_plot_path, dpi=300, bbox_inches='tight')
print(f"Validation confusion matrix plot saved to: {val_cm_plot_path}")
plt.show()

# -------------------------------
# Visualize Validation Confusion Matrix (Normalized)
# -------------------------------
val_cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(val_cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Percentage'})
plt.title('Validation Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
val_cm_norm_plot_path = os.path.join(plots_dir, "validation_confusion_matrix_normalized.png")
plt.savefig(val_cm_norm_plot_path, dpi=300, bbox_inches='tight')
print(f"Validation normalized confusion matrix plot saved to: {val_cm_norm_plot_path}")
plt.show()


# -------------------------------
# Evaluate on test data
# -------------------------------
print("\n--- Test Set Evaluation ---\n")

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Get predictions on test data
test_generator.reset()
test_Y_pred = model.predict(test_generator)
test_y_pred = np.argmax(test_Y_pred, axis=1)

# Get true labels for test data
test_steps = len(test_generator)
test_y_true = []

test_generator.reset()
for i in range(test_steps):
    _, y_batch = next(test_generator)
    test_y_true.extend(np.argmax(y_batch, axis=1))

# Trim to match prediction length
test_y_true = test_y_true[:len(test_y_pred)]

# Calculate test confusion matrix
test_cm = confusion_matrix(test_y_true, test_y_pred)
print("\nTest Confusion Matrix:")
print(test_cm)

# Get class names for better interpretation
class_names = list(train_generator.class_indices.keys())
print("\nClass Indices:", train_generator.class_indices)

# -------------------------------
# Visualize Test Confusion Matrix
# -------------------------------
print("\n--- Generating Test Confusion Matrix Plot ---\n")

plt.figure(figsize=(10, 8))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Test Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
test_cm_plot_path = os.path.join(plots_dir, "test_confusion_matrix.png")
plt.savefig(test_cm_plot_path, dpi=300, bbox_inches='tight')
print(f"Test confusion matrix plot saved to: {test_cm_plot_path}")
plt.show()

# -------------------------------
# Visualize Test Confusion Matrix (Normalized)
# -------------------------------
test_cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(test_cm_normalized, annot=True, fmt='.2%', cmap='RdPu',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Percentage'})
plt.title('Test Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
test_cm_norm_plot_path = os.path.join(plots_dir, "test_confusion_matrix_normalized.png")
plt.savefig(test_cm_norm_plot_path, dpi=300, bbox_inches='tight')
print(f"Test normalized confusion matrix plot saved to: {test_cm_norm_plot_path}")
plt.show()


# Generate classification report for test data
test_report = classification_report(
    test_y_true, 
    test_y_pred,
    target_names=class_names,
    digits=4
)
print("\nTest Classification Report:")
print(test_report)

# -------------------------------
# Visualize Classification Report
# -------------------------------
print("\n--- Generating Classification Report Visualization ---\n")

# Parse classification report to dictionary
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1_score, support = precision_recall_fscore_support(
    test_y_true, test_y_pred, labels=range(len(class_names))
)

# Create bar plot for classification metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Precision by class
axes[0, 0].bar(class_names, precision, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Precision', fontsize=12)
axes[0, 0].set_ylim([0, 1.1])
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(precision):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Recall by class
axes[0, 1].bar(class_names, recall, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Recall', fontsize=12)
axes[0, 1].set_ylim([0, 1.1])
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(recall):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# F1-Score by class
axes[1, 0].bar(class_names, f1_score, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('F1-Score', fontsize=12)
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(f1_score):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Support by class
axes[1, 1].bar(class_names, support, color='plum', edgecolor='black')
axes[1, 1].set_title('Support (Sample Count) by Class', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(support):
    axes[1, 1].text(i, v + max(support)*0.02, f'{int(v)}', ha='center', fontweight='bold')

plt.tight_layout()
metrics_plot_path = os.path.join(plots_dir, "classification_metrics.png")
plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
print(f"Classification metrics plot saved to: {metrics_plot_path}")
plt.show()

# -------------------------------
# Create Combined Metrics Comparison
# -------------------------------
x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral', edgecolor='black')
bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='lightgreen', edgecolor='black')

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Classification Metrics Comparison by Class', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
comparison_plot_path = os.path.join(plots_dir, "metrics_comparison.png")
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"Metrics comparison plot saved to: {comparison_plot_path}")
plt.show()

# -------------------------------
# Overall Performance Summary
# -------------------------------
overall_accuracy = np.sum(test_y_true == test_y_pred) / len(test_y_true)
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1_score)

print("\n" + "="*60)
print("OVERALL TEST PERFORMANCE SUMMARY")
print("="*60)
print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Macro-Average Precision: {macro_precision:.4f}")
print(f"Macro-Average Recall: {macro_recall:.4f}")
print(f"Macro-Average F1-Score: {macro_f1:.4f}")
print("="*60)

# Create summary visualization
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
metrics_values = [overall_accuracy, macro_precision, macro_recall, macro_f1]
colors = ['gold', 'skyblue', 'lightcoral', 'lightgreen']

bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=2)
ax.set_ylim([0, 1.1])
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Overall Model Performance on Test Set', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, metrics_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.4f}\n({val*100:.2f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
summary_plot_path = os.path.join(plots_dir, "overall_performance_summary.png")
plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
print(f"Overall performance summary plot saved to: {summary_plot_path}")
plt.show()


print("\n✅ Training finished. Model saved at:", model_path)
print(f"✅ All visualization plots saved in: {plots_dir}")
print("\nGenerated plots:")
print(f"  1. Training History: {history_plot_path}")
print(f"  2. Validation Confusion Matrix: {val_cm_plot_path}")
print(f"  3. Validation Confusion Matrix (Normalized): {val_cm_norm_plot_path}")
print(f"  4. Test Confusion Matrix: {test_cm_plot_path}")
print(f"  5. Test Confusion Matrix (Normalized): {test_cm_norm_plot_path}")
print(f"  6. Classification Metrics: {metrics_plot_path}")
print(f"  7. Metrics Comparison: {comparison_plot_path}")
print(f"  8. Overall Performance Summary: {summary_plot_path}")
