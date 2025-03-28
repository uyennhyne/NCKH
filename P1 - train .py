import os
import torch
import timm
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import cv2
import json
from skimage.metrics import structural_similarity as ssim   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị được sử dụng: {device}")

disease_classes = ['curl', 'healthy', 'leaf_spot', 'pear_slug', 'test1', 'test2']
severity_classes = ['0', '1', '2', '3', '4', 'test']

class DiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseModel, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class SeverityModel(nn.Module):
    def __init__(self, num_classes):
        super(SeverityModel, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.disease_labels = []
        self.severity_labels = []

        print(f"Khởi tạo dataset từ thư mục: {root_dir}")

        disease_dir = os.path.join(root_dir, 'disease')
        severity_dir = os.path.join(root_dir, 'severity')

        print("Bắt đầu thu thập dữ liệu gốc...")
        for disease_label in disease_classes:
            disease_folder = os.path.join(disease_dir, disease_label)
            if os.path.exists(disease_folder):
                images = [f for f in os.listdir(disease_folder) if f.endswith((".jpg", ".png"))]
                print(f"Tìm thấy {len(images)} ảnh trong thư mục {disease_label}")
                for filename in images:
                    self.image_paths.append(os.path.join(disease_folder, filename))
                    self.disease_labels.append(disease_classes.index(disease_label))
                    self.severity_labels.append(5)

        for severity_label in severity_classes:
            severity_folder = os.path.join(severity_dir, severity_label)
            if os.path.exists(severity_folder):
                images = [f for f in os.listdir(severity_folder) if f.endswith((".jpg", ".png"))]
                print(f"Tìm thấy {len(images)} ảnh trong thư mục {severity_label}")
                for filename in images:
                    self.image_paths.append(os.path.join(severity_folder, filename))
                    self.disease_labels.append(5)
                    self.severity_labels.append(severity_classes.index(severity_label))

        print(f"Tổng số ảnh trong dataset: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.disease_labels[idx], self.severity_labels[idx]

def show_random_images(dataset):
    print("Hiển thị ảnh ngẫu nhiên từ dataset...")
    idx = random.randint(0, len(dataset) - 1)
    image, _, _ = dataset[idx]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    original_image = Image.open(dataset.image_paths[idx]).convert("RGB")
    plt.imshow(original_image)
    plt.title("Ảnh gốc")
    plt.axis('off')

    transformed_image = image.cpu().numpy().transpose((1, 2, 0))
    transformed_image = (transformed_image * 0.5) + 0.5  
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.title("Ảnh sau Transform")
    plt.axis('off')

    plt.show()
    print("Đã hiển thị ảnh xong")

def train_disease_model(model, dataloader, criterion, optimizer, epochs=15, model_save_path="disease_model.pth"):
    print("Bắt đầu huấn luyện mô hình phân loại bệnh...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_disease = 0
        total_samples = 0
        
        for batch_idx, (images, disease_labels, severity_labels) in enumerate(dataloader):
            valid_mask = severity_labels == 5
            images, disease_labels = images[valid_mask], disease_labels[valid_mask]

            if images.size(0) == 0:  
                print(f"⚠️ Batch {batch_idx + 1} bị bỏ qua vì không có mẫu hợp lệ!")
                continue

            images, disease_labels = images.to(device), disease_labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, disease_labels)  
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_disease += (preds == disease_labels).sum().item()
            total_samples += disease_labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1} - Loss: {loss.item():.4f}, Accuracy: {correct_disease/total_samples*100:.2f}%")

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Disease Acc: {correct_disease/total_samples*100:.2f}%\n")
    
    print(f"Lưu mô hình bệnh vào {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)

def train_severity_model(model, dataloader, criterion, optimizer, epochs=25, model_save_path="severity_model.pth"):
    print("Bắt đầu huấn luyện mô hình phân loại mức độ nghiêm trọng...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_severity = 0
        total_samples = 0
        
        for batch_idx, (images, disease_labels, severity_labels) in enumerate(dataloader):
            valid_mask = disease_labels == 5
            images, severity_labels = images[valid_mask], severity_labels[valid_mask]

            if images.size(0) == 0:  
                print(f"⚠️ Batch {batch_idx + 1} bị bỏ qua vì không có mẫu hợp lệ!")
                continue

            images, severity_labels = images.to(device), severity_labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, severity_labels) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_severity += (preds == severity_labels).sum().item()
            total_samples += severity_labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1} - Loss: {loss.item():.4f}, Accuracy: {correct_severity/total_samples*100:.2f}%")

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Severity Acc: {correct_severity/total_samples*100:.2f}%\n")
    
    print(f"Lưu mô hình mức độ nghiêm trọng vào {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)

def train_and_save_models():
    dataset_path = r"C:\Users\ADMIN\Desktop\MSE\NCKH\DiaMOS"
    print(f"Đường dẫn dataset: {dataset_path}")

    print("Tạo dataset huấn luyện...")
    train_dataset = CustomDataset(
        root_dir=os.path.join(dataset_path, "train"),
        transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
    
    disease_model = DiseaseModel(num_classes=len(disease_classes)).to(device)
    print("Khởi tạo SeverityModel...")
    severity_model = SeverityModel(num_classes=len(severity_classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_disease = optim.Adam(disease_model.parameters(), lr=0.001)
    optimizer_severity = optim.Adam(severity_model.parameters(), lr=0.001)
    print("Đã thiết lập criterion và optimizers")

    train_disease_model(disease_model, train_dataloader, criterion, optimizer_disease, model_save_path="disease_model.pth")
    train_severity_model(severity_model, train_dataloader, criterion, optimizer_severity, model_save_path="severity_model.pth")
    print("Huấn luyện hoàn tất và mô hình đã được lưu.")

if __name__ == "__main__":
    print("Bắt đầu chương trình huấn luyện...")
    train_and_save_models()
    print("Chương trình huấn luyện hoàn tất!")

