import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
import numpy as np

# Step 1: Define how to convert image
def fix_emnist_rotation(x):
    x = torch.rot90(x, 1, [1, 2])  # Rotate
    x = torch.flip(x, [1]) # Flip
    return x
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(fix_emnist_rotation)])

# Step 2: Download EMNIST dataset
train_data = datasets.EMNIST(
    root='./data',
    split='byclass', #Full dataset
    train=True,
    download=True,
    transform=transform)

# Step 3: Convert label number to actual character
def label_to_char(label):
    if 0 <= label <= 9:
        return chr(label + ord('0'))            # '0' to '9'
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))       # 'A' to 'Z'
    elif 36 <= label <= 61:
        return chr(label - 36 + ord('a'))       # 'a' to 'z'
    else:
        return "?"  # just in case
    
# Step 4: Show multiple images with characters
plt.figure(figsize=(12, 6))
for i in range(15): # first 15 images
    img, label = train_data[i]
    char = label_to_char(label)
    plt.subplot(3, 5, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"'{char}'")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 5: Split dataset into training and validation sets
train_size = int(0.9 * len(train_data)) # 54,000 train, 6,000 val (EMNIST byclass has 60,000 train images)
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# Step 6: Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

# Step 6: Build a simple neural network
class CNNHandwrittenCharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: 1x28x28 → 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # → 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # → 64x7x7
        )
        self.fc = nn.Sequential(  # 🔧 New final layers
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 62)
        )
            
    def forward(self, x):
        x = self.conv_block(x)
        return self.fc(x)
# Create the model
model = CNNHandwrittenCharModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 7: Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # model learns using Adam

# Step 8: Train the model
def train_model (model, train_loader, val_loder , loss_fn, optimizer, epochs=5):
    history = {'train_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    all_preds = []
    all_labels = []
    for epoch in range(epochs):  # Run for a few full passes over the dataset
        model.train()  # Put model in training mode
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Move images and labels to GPU if available
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: get predictions
            outputs = model(images)
            # Calculate loss
            loss = loss_fn(outputs, labels)
            # Backward pass: compute gradients
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
            # Track accuracy
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
    
        model.eval()  # Put model in evaluation mode
        val_correct = 0
        val_total = 0
        all_preds.clear()
        all_labels.clear()

        with torch.no_grad():  # No gradient calculation needed during evaluation
           for images, labels in val_loader:
              images = images.to(device)
              labels = labels.to(device)

              outputs = model(images)
              _, predicted = torch.max(outputs, 1)
              val_correct += (predicted == labels).sum().item()
              val_total += labels.size(0)

              # 🔥 Only during final epoch — collect for confusion matrix
              if epoch == epochs - 1:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        val_accuracy = 100 * val_correct /  val_total

        # Save for graph
        history['train_loss'].append(avg_loss)
        history['train_accuracy'].append(accuracy)
        history['val_accuracy'].append(val_accuracy)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | Loss: {avg_loss:.4f}")
     #After training ends, return history and final predictions
    return history, all_preds, all_labels

# Step 12: Start training
history , all_preds, all_labels = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=5)

# Step 13: Confusion Matrix
label_names = [] # Build readable character labels
# Digits 0-9
for i in range(10):
    label_names.append(chr(ord('0') + i))
# Uppercase A-Z
for i in range(26):
    label_names.append(chr(ord('A') + i))
# Lowercase a-z
for i in range(26):
    label_names.append(chr(ord('a') + i))
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels=label_names)
plt.figure(figsize=(30, 30))
disp.plot(cmap='Blues', xticks_rotation=45 , values_format='d')
plt.title("Confusion Matrix - Final Validation")
plt.tight_layout()
plt.show()

# Plot loss and accuracy
plt.figure(figsize=(12, 5))
# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.legend()
# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_accuracy'], label='Train Acc', marker='o')
plt.plot(history['val_accuracy'], label='Val Acc', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
