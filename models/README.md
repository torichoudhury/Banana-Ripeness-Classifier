# Model Files Directory

Place your three trained PyTorch model files here:

## Required Files

1. `banana_ripeness_efficientnet_pytorch.pth`
   - EfficientNet-B0 model
   - Generated from `banana_ripeness_classifier_ENB0_pytorch.py`

2. `banana_ripeness_resnet_pytorch.pth`
   - ResNet50 model
   - Generated from `banana_ripeness_classifier_ResNet_pytorch.py`

3. `banana_ripeness_mobilenet_pytorch.pth`
   - MobileNetV2 model
   - Generated from `banana_ripeness_classifier_MobileNet_pytorch.py`

## File Format

These should be PyTorch checkpoint files (`.pth`) containing:
- `model_state_dict`: The trained model weights
- `class_names`: List of class labels
- `num_classes`: Number of output classes

## Location

Copy the model files from your training directory to this folder:

```bash
cp path/to/your/models/*.pth ./models/
```

## Verification

After placing the files, you should have:
```
models/
├── banana_ripeness_efficientnet_pytorch.pth
├── banana_ripeness_resnet_pytorch.pth
├── banana_ripeness_mobilenet_pytorch.pth
└── README.md (this file)
```

The backend server will automatically load these models when it starts.
