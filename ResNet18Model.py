import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image   
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FER2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        self.images = []
        self.labels = []

        data_dir = os.path.join(root_dir, split)

        for class_idx, emotion in enumerate(self.classes):
            class_dir = os.path.join(data_dir, emotion)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)

        # Calculate class weights based on the inverse frequency of the classes
        class_counts = torch.zeros(len(self.classes))  # to store the count of each class
        for label in self.labels:
            class_counts[label] += 1

        total_samples = len(self.labels)
        
        # Compute class weights: total samples / class count
        class_weights = total_samples / class_counts

        # Optionally, you can normalize weights, but not to the sum of all weights.
        # Instead, you can divide each weight by the maximum weight to keep the scale manageable.
        max_weight = class_weights.max()
        self.class_weights = class_weights / max_weight

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label


class PrepareDataset:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 2.0), value='random')
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

        self.train_dataset = FER2013Dataset(root_dir='archive',transform=self.train_transform,split='train')
        self.val_dataset = FER2013Dataset(root_dir='archive',transform=self.val_transform,split='val')
        self.test_dataset = FER2013Dataset(root_dir='archive',transform=self.val_transform,split='test') 
        
        self.train_loader = DataLoader(self.train_dataset, 64, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, 64, shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, 64, shuffle=False, num_workers=4, pin_memory=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = ((1 - p) ** self.gamma) * logp
        return loss.mean()

        
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(ResNet18Model, self).__init__()
        self.dropout_rate = dropout_rate

        # Load pre-trained EfficientNet
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the classifier head
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    def TrainModel(self, train_loader, val_loader, device):
        weights = train_loader.dataset.class_weights.to(device)
        criterion = FocalLoss(weight=weights)
        optimizer = optim.AdamW(self.parameters(), lr=2e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,           
            steps_per_epoch=len(train_loader),
            epochs=20,
        )

        best_val_acc = 0.0  
        patience = 6
        patience_counter = 0

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(20):
            self.model.train()
            train_loss, train_correct, total = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, preds = outputs.max(1)
                train_correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_losses.append(train_loss / len(train_loader))
            train_accuracies.append(train_correct / total)

            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_correct / val_total)

            # Early stopping check
            if val_accuracies[-1] > best_val_acc:
                best_val_acc = val_accuracies[-1]
                patience_counter = 0  # Reset patience counter when we find a better model
                torch.save(self.state_dict(), "best_model1.pth")
                print("Model saved as 'best_model.pth'")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Print epoch results after all operations
            print(f"Epoch {epoch + 1}/{20}, Train Loss: {train_losses[-1]:.4f}, "
                  f"Train Acc: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
                  f"Val Acc: {val_accuracies[-1]:.4f}")

        return train_losses, train_accuracies, val_losses, val_accuracies
    

def visualize_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            # Get indices of correct and incorrect predictions
            correct_mask = preds == labels
            incorrect_mask = ~correct_mask
            
            # Store correct predictions
            for i in range(len(images)):
                if len(correct_samples) < num_samples and correct_mask[i]:
                    correct_samples.append({
                        'image': images[i].cpu(),
                        'pred': preds[i].item(),
                        'true': labels[i].item()
                    })
                
                if len(incorrect_samples) < num_samples and incorrect_mask[i]:
                    incorrect_samples.append({
                        'image': images[i].cpu(),
                        'pred': preds[i].item(),
                        'true': labels[i].item()
                    })
                
                if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                    break
            
            if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                break
    
    # Create visualization
    plt.figure(figsize=(20, 8))
    
    # Plot correct predictions
    plt.subplot(2, 1, 1)
    plt.title('Correct Predictions')
    for i, sample in enumerate(correct_samples):
        plt.subplot(2, num_samples, i + 1)
        img = sample['image'].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        plt.imshow(img)
        plt.title(f'Pred: {test_loader.dataset.classes[sample["pred"]]}\nTrue: {test_loader.dataset.classes[sample["true"]]}')
        plt.axis('off')
    
    # Plot incorrect predictions
    plt.subplot(2, 1, 2)
    plt.title('Incorrect Predictions')
    for i, sample in enumerate(incorrect_samples):
        plt.subplot(2, num_samples, num_samples + i + 1)
        img = sample['image'].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        plt.imshow(img)
        plt.title(f'Pred: {test_loader.dataset.classes[sample["pred"]]}\nTrue: {test_loader.dataset.classes[sample["true"]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.close()
    print("\n✅ Sample predictions visualization saved as 'prediction_samples.png'")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset
    dataset = PrepareDataset()
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    test_loader = dataset.test_loader

    # Initialize model
    model = ResNet18Model(num_classes=5).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Train model
    print("Starting training...")
    train_losses, train_accuracies, val_losses, val_accuracies = model.TrainModel(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Load the best saved model
    model.load_state_dict(torch.load("best_model1.pth"))
    model.eval()

    #Evaluate on test set
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            test_correct += preds.eq(labels).sum().item()
            test_total += labels.size(0)

        test_accuracy = test_correct / test_total
        print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")

        # Create confusion matrix for per-class accuracy
        # Get all predictions and true labels
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Get the actual classes present in the data
        present_classes = []
        for i, acc in enumerate(class_accuracy):
            if acc > 0:  # Only include classes that have samples
                present_classes.append(dataset.test_dataset.classes[i])
        
        # Create figure with two subplots
        plt.figure(figsize=(15, 6))
        
        # Plot confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=present_classes,
                   yticklabels=present_classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Plot per-class accuracy
        plt.subplot(1, 2, 2)
        plt.bar(present_classes, class_accuracy[class_accuracy > 0])
        plt.title('Per-Class Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        
        # Add accuracy values on top of bars
        for i, acc in enumerate(class_accuracy[class_accuracy > 0]):
            plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center')
        
        plt.tight_layout()
        plt.savefig('class_accuracy1.png')
        plt.close()
        print("\n✅ Class accuracy visualization saved as 'class_accuracy.png'")

    # After evaluating on test set and before creating confusion matrix
    print("\nVisualizing sample predictions...")
    visualize_predictions(model, test_loader, device)