import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. CSV 로딩 및 라벨 처리
df = pd.read_csv('base_data.csv')
df = df[df['Finding Labels'].notna()]
df['Pneumothorax'] = df['Finding Labels'].apply(lambda x: 1 if 'Pneumothorax' in x else 0)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Pneumothorax'])

# 2. Dataset 클래스 정의
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])
        image = Image.open(img_path).convert("L")
        label = torch.tensor(row['Finding Labels'].map({'No Finding': 0, 'Pneumothorax': 1}), dtype=torch.int32)
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. Train & Eval 함수 정의
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# 4. Main 함수
def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on NIH Chest X-ray for Pneumothorax detection")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr

    print(f"Batch size: {batch_size}, Epochs: {epochs}, Learning Rate: {learning_rate}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = ChestXrayDataset(train_df, "./images", transform)
    val_dataset = ChestXrayDataset(val_df, "./images", transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.conv1.in_channels = 1
    model.conv1.weight.data = model.state_dict()['conv1.weight'].mean(dim=1, keepdim=True)  # fits grayscale
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, criterion)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
