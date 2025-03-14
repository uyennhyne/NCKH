import os
import torch
import timm
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_classes = ['curl', 'healthy', 'leaf_spot', 'pear_slug', 'test1', 'test2']
severity_classes = ['0', '1', '2', '3', '4', 'test']

def segment_leaf(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang định dạng RGB
    
    # Create an initial mask with the same size as the image
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Define a rectangle for GrabCut (có thể tùy chỉnh tọa độ)
    # Ở đây tôi dùng một vùng mặc định, bạn có thể thay đổi hoặc thêm tương tác chuột
    height, width = image.shape[:2]
    rect = (50, 50, width-100, height-100)  # Ví dụ: cách lề 50px
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify the mask: 0 and 2 are background, 1 and 3 are foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to the original image
    segmented_image = image * mask2[:, :, np.newaxis]
    
    # Convert back to PIL Image for compatibility with the rest of the code
    return Image.fromarray(segmented_image)

# Dataset tùy chỉnh
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.disease_labels = []
        self.severity_labels = []

        disease_dir = os.path.join(root_dir, 'disease')
        severity_dir = os.path.join(root_dir, 'severity')

        for disease_label in disease_classes:
            disease_folder = os.path.join(disease_dir, disease_label)
            if os.path.exists(disease_folder):
                for filename in os.listdir(disease_folder):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        self.image_paths.append(os.path.join(disease_folder, filename))
                        self.disease_labels.append(disease_classes.index(disease_label))
                        self.severity_labels.append(5)

        for severity_label in severity_classes:
            severity_folder = os.path.join(severity_dir, severity_label)
            if os.path.exists(severity_folder):
                for filename in os.listdir(severity_folder):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        self.image_paths.append(os.path.join(severity_folder, filename))
                        self.disease_labels.append(5)
                        self.severity_labels.append(int(severity_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.disease_labels[idx], self.severity_labels[idx]

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

print("Loading datasets...")
dataset_path = r"C:\Users\ADMIN\Desktop\MSE\NCKH\DiaMOS"
# Dataset và DataLoader
train_dataset = CustomDataset(root_dir=os.path.join(dataset_path, "train"), transform=transform)
val_dataset = CustomDataset(root_dir=os.path.join(dataset_path, "val"), transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

# Mô hình bệnh
class DiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseModel, self).__init__()
        self.base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.base_model.reset_classifier(num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# Mô hình mức độ nghiêm trọng
class SeverityModel(nn.Module):
    def __init__(self, num_classes):
        super(SeverityModel, self).__init__()
        self.base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.base_model.reset_classifier(num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# Huấn luyện mô hình bệnh
def train_disease_model(model, dataloader, criterion, optimizer, epochs=1):
    print("Starting disease model training...")
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

            print(f"Batch {batch_idx + 1} - Loss: {loss.item():.4f}, Accuracy: {correct_disease/total_samples*100:.2f}%")

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Disease Acc: {correct_disease/total_samples*100:.2f}%\n")

# Huấn luyện mô hình mức độ nghiêm trọng
def train_severity_model(model, dataloader, criterion, optimizer, epochs=1):
    print("Starting severity model training...")
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

            print(f"Batch {batch_idx + 1} - Loss: {loss.item():.4f}, Accuracy: {correct_severity/total_samples*100:.2f}%")

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Severity Acc: {correct_severity/total_samples*100:.2f}%\n")

def suggest_treatment(disease, severity):
    treatments = {
        'curl': ["Không cần điều trị", "Tưới nước đúng cách", "Sử dụng phân bón", "Phun thuốc bảo vệ thực vật", "Cắt tỉa phần bị hỏng"],
        'healthy': ["Cây khỏe mạnh, tiếp tục chăm sóc bình thường"],
        'leaf_spot': ["Kiểm tra cây", "Tăng cường dinh dưỡng", "Phun thuốc trị nấm", "Cắt bỏ lá bị bệnh", "Cách ly cây bị bệnh"],
        'pear_slug': ["Rửa sạch lá", "Dùng thuốc trừ sâu nhẹ", "Cắt bỏ lá bị hại", "Dùng thuốc trừ sâu mạnh"]
    }
    return treatments[disease][severity]

# Dự đoán và gợi ý điều trị
def predict_with_segmentation(image_path, disease_model, severity_model):
    original_image = Image.open(image_path).convert("RGB")  # Ảnh gốc
    segmented_image = segment_leaf(image_path)  # Tách lá bằng GrabCut
    
    # Hiển thị ảnh gốc và ảnh sau khi xử lý
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Ảnh gốc")
    ax[0].axis("off")

    ax[1].imshow(segmented_image)
    ax[1].set_title("Ảnh sau khi xử lý")
    ax[1].axis("off")

    plt.show()

    # Tiếp tục tiền xử lý ảnh để phân loại bệnh
    transformed_image = transform(segmented_image).unsqueeze(0).to(device)

    disease_model.eval()
    severity_model.eval()
    with torch.no_grad():
        disease_output = disease_model(transformed_image)
        severity_output = severity_model(transformed_image)

        _, predicted_disease = torch.max(disease_output, 1)
        _, predicted_severity = torch.max(severity_output, 1)

        disease_name = disease_classes[predicted_disease.item()]
        severity_level = predicted_severity.item()

        print(f"Dự đoán bệnh: {disease_name}")
        print(f"Mức độ nghiêm trọng: {severity_level}")

        treatment = suggest_treatment(disease_name, severity_level)
        print(f"Hướng xử lý: {treatment}")

# Kiểm tra mô hình
if __name__ == '__main__':
    disease_model = DiseaseModel(num_classes=len(disease_classes)).to(device)
    severity_model = SeverityModel(num_classes=len(severity_classes)).to(device)
    
    optimizer_disease = optim.Adam(disease_model.parameters(), lr=0.001)
    optimizer_severity = optim.Adam(severity_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_disease_model(disease_model, train_dataloader, criterion, optimizer_disease, epochs=1)
    train_severity_model(severity_model, train_dataloader, criterion, optimizer_severity, epochs=1)
    
    test_image_path = r"C:\Users\ADMIN\Desktop\MSE\NCKH\DiaMOS\test\disease\curl\aug_curl_0_3276.jpg"
    predict_with_segmentation(test_image_path, disease_model, severity_model)