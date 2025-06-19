# src/evaluate.py
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def evaluate_model(data_dir=r'E:\2025\PROJECT_AI\data_split\test',
                   model_path="models/resnet50_best.pth",
                   batch_size=16,
                   cm_save_path="outputs/confusion_matrix.png",
                   acc_plot_path="outputs/class_accuracy.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_dataset = ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(test_dataset.classes))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    #Đánh giá theo tưng lớp
    print("Đánh giá của từng lớp:")
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
    for cls in test_dataset.classes:
        acc = report[cls]['recall']
        print(f"{cls}: Accuracy (Recall) = {acc:.2f}")

    #ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes, cmap="Blues")
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(cm_save_path)
    print(f"Đã lưu: {cm_save_path}")
    plt.close()

    #độ chính xác cảu từng lớp
    class_accuracies = [report[cls]['recall'] * 100 for cls in test_dataset.classes]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=test_dataset.classes, y=class_accuracies, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    for i, acc in enumerate(class_accuracies):
        plt.text(i, acc + 1, f"{acc:.1f}%", ha='center', fontsize=9)
    plt.ylim(0, 110)
    os.makedirs(os.path.dirname(acc_plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(acc_plot_path)
    print(f"Đã lưu: {acc_plot_path}")
    plt.close()


if __name__ == '__main__':
    evaluate_model()
