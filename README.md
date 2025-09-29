# Banana Ripeness Classifier

A deep learning model to classify bananas into four categories:
- Ripe
- Overripe
- Rotten
- Unripe

## Dataset

The dataset is located at `D:\Banana Ripeness Classification Dataset` and is organized as follows:
- train/
  - overripe/
  - ripe/
  - rotten/
  - unripe/
- test/
  - overripe/
  - ripe/
  - rotten/
  - unripe/
- valid/
  - overripe/
  - ripe/
  - rotten/
  - unripe/

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python banana_ripeness_classifier.py
   ```
   This will create a `models` directory with the trained model and visualization results.

3. Predict ripeness for a new image:
   ```
   python predict_banana.py path/to/your/banana/image.jpg
   ```

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:
- 4 convolutional blocks with increasing filter sizes (32, 64, 128, 256)
- Max pooling after each convolutional layer
- Dropout regularization to prevent overfitting
- Output layer with 4 classes (overripe, ripe, rotten, unripe)

## Evaluation

The model is evaluated using:
- Test accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix