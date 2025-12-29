import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------
# Early Stopping
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# -------------------------------
# Build EfficientNet-B0 model
# -------------------------------
class BananaRipenessClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(BananaRipenessClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
    
    def freeze_backbone(self):
        for name, param in self.base_model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / total, correct / total, all_preds, all_labels

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_dir = r"D:\Banana Ripeness Classification Dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS_HEAD = 20
    EPOCHS_FINE = 15

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"\nClasses: {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_labels = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)

    model = BananaRipenessClassifier(num_classes=num_classes).to(device)
    model.freeze_backbone()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print("\n" + "="*60)
    print("STAGE 1: Training classifier head")
    print("="*60)

    history_head = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping_head = EarlyStopping(patience=5, min_delta=0.05)

    for epoch in range(EPOCHS_HEAD):
        print(f"\nEpoch {epoch+1}/{EPOCHS_HEAD}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_head, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history_head['train_loss'].append(train_loss)
        history_head['train_acc'].append(train_acc)
        history_head['val_loss'].append(val_loss)
        history_head['val_acc'].append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        early_stopping_head(val_loss)
        if early_stopping_head.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    print("\n" + "="*60)
    print("STAGE 2: Fine-tuning")
    print("="*60)

    model.unfreeze_backbone()
    optimizer_fine = optim.Adam(model.parameters(), lr=1e-5)
    history_fine = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping_fine = EarlyStopping(patience=5, min_delta=0.05)

    for epoch in range(EPOCHS_FINE):
        print(f"\nEpoch {epoch+1}/{EPOCHS_FINE}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_fine, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history_fine['train_loss'].append(train_loss)
        history_fine['train_acc'].append(train_acc)
        history_fine['val_loss'].append(val_loss)
        history_fine['val_acc'].append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        early_stopping_fine(val_loss)
        if early_stopping_fine.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "banana_ripeness_efficientnet_pytorch.pth")
    torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names, 'num_classes': num_classes}, model_path)
    print(f"\n✅ Model saved: {model_path}")

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    combined_history = {
        'train_loss': history_head['train_loss'] + history_fine['train_loss'],
        'train_acc': history_head['train_acc'] + history_fine['train_acc'],
        'val_loss': history_head['val_loss'] + history_fine['val_loss'],
        'val_acc': history_head['val_acc'] + history_fine['val_acc']
    }

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(combined_history['train_acc'], label='Train Acc', marker='o')
    plt.plot(combined_history['val_acc'], label='Val Acc', marker='s')
    plt.axvline(x=EPOCHS_HEAD-1, color='red', linestyle='--')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(combined_history['train_loss'], label='Train Loss', marker='o')
    plt.plot(combined_history['val_loss'], label='Val Loss', marker='s')
    plt.axvline(x=EPOCHS_HEAD-1, color='red', linestyle='--')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_history.png"), dpi=300)
    plt.close()

    print("\n--- Test Evaluation ---")
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    test_cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "test_confusion_matrix.png"), dpi=300)
    plt.close()

    print(classification_report(test_labels, test_preds, target_names=class_names))
    print(f"\n✅ Plots saved in: {plots_dir}")
