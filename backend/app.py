from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
from pathlib import Path

app = FastAPI(title="Banana Ripeness Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(EfficientNetModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class ResNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNetModel, self).__init__()
        self.base_model = models.resnet50(pretrained=pretrained)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class MobileNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(MobileNetModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_dict = {}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_models():
    base_path = Path(__file__).parent.parent / "models"

    model_configs = {
        "EfficientNet-B0": {
            "path": base_path / "banana_ripeness_efficientnet_pytorch.pth",
            "class": EfficientNetModel
        },
        "ResNet50": {
            "path": base_path / "banana_ripeness_resnet_pytorch.pth",
            "class": ResNetModel
        },
        "MobileNetV2": {
            "path": base_path / "banana_ripeness_mobilenet_pytorch.pth",
            "class": MobileNetModel
        }
    }

    for model_name, config in model_configs.items():
        try:
            if config["path"].exists():
                checkpoint = torch.load(config["path"], map_location=device)
                num_classes = checkpoint['num_classes']
                class_names = checkpoint['class_names']

                model = config["class"](num_classes=num_classes)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

                models_dict[model_name] = {
                    "model": model,
                    "class_names": class_names
                }
                print(f"✅ Loaded {model_name}")
            else:
                print(f"⚠️  Model file not found: {config['path']}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)}")

load_models()

@app.get("/")
async def root():
    return {"message": "Banana Ripeness Classifier API", "available_models": list(models_dict.keys())}

@app.get("/models")
async def get_models():
    return {
        "models": [
            {
                "name": name,
                "classes": data["class_names"]
            }
            for name, data in models_dict.items()
        ]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not models_dict:
        raise HTTPException(status_code=503, detail="No models available")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        input_tensor = transform(image).unsqueeze(0).to(device)

        results = {}

        with torch.no_grad():
            for model_name, model_data in models_dict.items():
                model = model_data["model"]
                class_names = model_data["class_names"]

                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_class = class_names[predicted_idx.item()]

                all_confidences = {
                    class_names[i]: float(probabilities[i].item())
                    for i in range(len(class_names))
                }

                results[model_name] = {
                    "predicted_class": predicted_class,
                    "confidence": float(confidence.item()),
                    "all_confidences": all_confidences
                }

        return JSONResponse(content={
            "success": True,
            "predictions": results
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
