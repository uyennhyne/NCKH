import os
import torch
import timm
import time
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_classes = ['curl', 'healthy', 'leaf_spot', 'pear_slug']
severity_classes = ['0', '1', '2', '3', '4']

print("Initializing dataset...")

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.disease_labels = []
        self.severity_labels = []

        
        severity_dir = os.path.join(root_dir, 'severity') 
        
        for severity_label in severity_classes:
            severity_folder = os.path.join(severity_dir, severity_label)
            if os.path.exists(severity_folder):
                for filename in os.listdir(severity_folder):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        image_path = os.path.join(severity_folder, filename)
                        self.image_paths.append(image_path)
                        self.disease_labels.append(0)  
                        self.severity_labels.append(int(severity_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        disease_label = self.disease_labels[idx]
        severity_label = self.severity_labels[idx]
        
        return image, disease_label, severity_label

# Transform chuẩn cho ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Chuẩn hóa theo ViT
])

dataset_path = "D:\\MSE\\NCKH\\DiaMOS"
train_dataset = CustomDataset(root_dir=os.path.join(dataset_path, "train"), transform=transform)
val_dataset = CustomDataset(root_dir=os.path.join(dataset_path, "val"), transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

# Model cải tiến
class MultiTaskModel(nn.Module):
    def __init__(self, num_disease_classes, num_severity_classes):
        super(MultiTaskModel, self).__init__()
        self.base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.base_model.reset_classifier(0)
        
        self.disease_fc = nn.Linear(self.base_model.num_features, num_disease_classes)
        self.severity_fc = nn.Sequential(
            nn.Linear(self.base_model.num_features, 128),  # Thêm hidden layer
            nn.ReLU(),
            nn.Linear(128, num_severity_classes)
        )
    
    def forward(self, x):
        features = self.base_model(x)
        disease_output = self.disease_fc(features)
        severity_output = self.severity_fc(features)
        return disease_output, severity_output

model = MultiTaskModel(num_disease_classes=len(disease_classes), num_severity_classes=len(severity_classes))
model = model.to(device)

criterion_disease = nn.CrossEntropyLoss()
criterion_severity = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Giảm learning rate để ổn định

def train_model(model, dataloader, criterion_disease, criterion_severity, optimizer, epochs=2):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        running_loss_disease = 0.0
        running_loss_severity = 0.0
        correct_disease = 0
        correct_severity = 0
        total_samples = 0

        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, disease_labels, severity_labels) in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
            optimizer.zero_grad()
            
            disease_output, severity_output = model(images)
            
            loss_disease = criterion_disease(disease_output, disease_labels)
            loss_severity = criterion_severity(severity_output, severity_labels)
            
            loss = loss_disease + loss_severity
            loss.backward()
            optimizer.step()
            
            running_loss_disease += loss_disease.item()
            running_loss_severity += loss_severity.item()
            
            _, predicted_disease = torch.max(disease_output, 1)
            _, predicted_severity = torch.max(severity_output, 1)
            
            correct_disease += (predicted_disease == disease_labels).sum().item()
            correct_severity += (predicted_severity == severity_labels).sum().item()
            total_samples += images.size(0)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} sec, Disease Acc: {correct_disease/total_samples:.4f}, Severity Acc: {correct_severity/total_samples:.4f}\n")
                       

def suggest_treatment(disease, severity):
    treatments = {
        'curl': ["Không cần điều trị", "Tưới nước đúng cách", "Sử dụng phân bón", "Phun thuốc bảo vệ thực vật", "Cắt tỉa phần bị hỏng"],
        'healthy': ["Cây khỏe mạnh, tiếp tục chăm sóc bình thường"],
        'leaf_spot': ["Kiểm tra cây", "Tăng cường dinh dưỡng", "Phun thuốc trị nấm", "Cắt bỏ lá bị bệnh", "Cách ly cây bị bệnh"],
        'pear_slug': ["Rửa sạch lá", "Dùng thuốc trừ sâu nhẹ", "Cắt bỏ lá bị hại", "Dùng thuốc trừ sâu mạnh"]
    }
    return treatments[disease][severity]

def predict_and_suggest(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        disease_output, severity_output = model(image)
        
        print("Raw logits (disease):", disease_output.cpu().numpy())  # Debug log
        print("Raw logits (severity):", severity_output.cpu().numpy())  # Debug log
        
        _, predicted_disease = torch.max(disease_output, 1)
        _, predicted_severity = torch.max(severity_output, 1)
        
        disease_name = disease_classes[predicted_disease.item()]
        severity_level = predicted_severity.item()
        treatment = suggest_treatment(disease_name, severity_level)
        
        
        print(f"Dự đoán bệnh: {disease_name}")
        print(f"Mức độ nghiêm trọng: {severity_level}")
        print(f"Hướng xử lý: {treatment}")

if __name__ == '__main__':
    train_model(model, train_dataloader, criterion_disease, criterion_severity, optimizer, epochs=2)
    test_image_path = r"D:\MSE\NCKH\DiaMOS\test\severity\2\1446_aug_0_u1143.jpg"
    predict_and_suggest(test_image_path)