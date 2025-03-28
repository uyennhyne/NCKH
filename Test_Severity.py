import os
import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn as nn
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

severity_classes = ['0', '1', '2', '3', '4', 'test']

class SeverityModel(nn.Module):
    def __init__(self, num_classes):
        super(SeverityModel, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

severity_model = SeverityModel(num_classes=len(severity_classes)).to(device)
severity_model.load_state_dict(torch.load("severity_model_retrained.pth", map_location=device))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def grabcut_segmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    
    return Image.fromarray(segmented_image)

class CustomDataset(Dataset):
    def __init__(self, severity_dir, severity_classes, transform=None):
        self.severity_classes = severity_classes
        self.transform = transform
        self.image_paths = []
        self.severity_labels = []

        for severity_label in severity_classes:
            severity_folder = os.path.join(severity_dir, severity_label)
            if os.path.exists(severity_folder):
                images = [f for f in os.listdir(severity_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                print(f"Tìm thấy {len(images)} ảnh trong thư mục {severity_label}")
                for filename in images:
                    self.image_paths.append(os.path.join(severity_folder, filename))
                    self.severity_labels.append(severity_classes.index(severity_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = grabcut_segmentation(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        severity_label = self.severity_labels[idx]
        return image, severity_label

test_directory = r"C:\Users\heine\Desktop\py\PlantVillage\test"
dataset = CustomDataset(severity_dir=os.path.join(test_directory, "severity"),
                        severity_classes=severity_classes,
                        transform=transform)
dataloader = DataLoader(dataset, batch_size=100, num_workers=12)

def evaluate_test_set(test_loader):
    severity_model.eval()
    severity_true_labels = []
    severity_pred_labels = []

    with torch.no_grad():
        for images, severity_labels in test_loader:
            images = images.to(device)
            severity_output = severity_model(images)
            _, predicted_severity = torch.max(severity_output, 1)
            severity_true_labels.extend(severity_labels.cpu().numpy())
            severity_pred_labels.extend(predicted_severity.cpu().numpy())

    severity_precision, severity_recall, severity_f1, _ = precision_recall_fscore_support(
        severity_true_labels, severity_pred_labels, average='weighted', labels=range(len(severity_classes)), zero_division=1)
    severity_accuracy = accuracy_score(severity_true_labels, severity_pred_labels)

    print(f"Severity Precision: {severity_precision:.4f}")
    print(f"Severity Recall: {severity_recall:.4f}")
    print(f"Severity F1-score: {severity_f1:.4f}")
    print(f"Severity Accuracy: {severity_accuracy:.4f}")

if __name__ == '__main__':
    evaluate_test_set(dataloader)