import os
import torch
import cv2
import timm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn as nn
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_classes = ['curl', 'healthy', 'leaf_spot', 'pear_slug', 'test1', 'test2']

class DiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseModel, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

disease_model = DiseaseModel(num_classes=len(disease_classes)).to(device)
disease_model.load_state_dict(torch.load("disease_model_retrained.pth", map_location=device))

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
    def __init__(self, disease_dir, disease_classes, transform=None):
        self.disease_classes = disease_classes
        self.transform = transform
        self.image_paths = []
        self.disease_labels = []

        for disease_label in disease_classes:
            disease_folder = os.path.join(disease_dir, disease_label)
            if os.path.exists(disease_folder):
                images = [f for f in os.listdir(disease_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                print(f"Tìm thấy {len(images)} ảnh trong thư mục {disease_label}")
                for filename in images:
                    self.image_paths.append(os.path.join(disease_folder, filename))
                    self.disease_labels.append(disease_classes.index(disease_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            segmented_image = grabcut_segmentation(image_path)
            
            if self.transform:
                segmented_image = self.transform(segmented_image)
            
            disease_label = self.disease_labels[idx]
            return segmented_image, disease_label
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
            return None, None

test_directory = r"C:\Users\heine\Desktop\py\PlantVillage\test"
dataset = CustomDataset(disease_dir=os.path.join(test_directory, "disease"),
                        disease_classes=disease_classes,
                        transform=transform)
dataloader = DataLoader(dataset, batch_size=100, num_workers=12)

def evaluate_test_set(test_loader):
    disease_true_labels = []
    disease_pred_labels = []

    disease_model.eval()
    with torch.no_grad():
        for images, disease_labels in test_loader:
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            if not valid_indices:
                continue
            images = torch.stack([images[i] for i in valid_indices]).to(device)
            disease_labels = [disease_labels[i] for i in valid_indices]

            disease_output = disease_model(images)
            _, predicted_disease = torch.max(disease_output, 1)

            for i in range(len(valid_indices)):
                disease_name = disease_classes[predicted_disease[i].item()]
                true_label = disease_classes[disease_labels[i]]
                disease_true_labels.append(true_label)
                disease_pred_labels.append(disease_name)

    disease_precision, disease_recall, disease_f1, _ = precision_recall_fscore_support(
        disease_true_labels, disease_pred_labels, average='weighted', labels=disease_classes, zero_division=1)
    disease_accuracy = accuracy_score(disease_true_labels, disease_pred_labels)

    print(f"Disease Precision: {disease_precision:.4f}")
    print(f"Disease Recall: {disease_recall:.4f}")
    print(f"Disease F1-score: {disease_f1:.4f}")
    print(f"Disease Accuracy: {disease_accuracy:.4f}")

if __name__ == '__main__':
    evaluate_test_set(dataloader)