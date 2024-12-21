import json
import torch
import yaml
from sklearn.metrics import classification_report
from train import AnimalsClassifier, CustomDataset, load_images_from_folder
from torch.utils.data import DataLoader

import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import yaml

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


transform = transforms.Compose([
    transforms.Resize((params['train']['image_size'],
                      params['train']['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


foxes_dir = "data/foxes_images/"
butterflies_dir = "data/butterflies_images/"
fox_tensors, fox_targets = load_images_from_folder(foxes_dir, 0)
butterfly_tensors, butterfly_targets = load_images_from_folder(
    butterflies_dir, 1)

all_tensors = fox_tensors + butterfly_tensors
all_targets = fox_targets + butterfly_targets

all_tensors = torch.stack(all_tensors)
all_targets = torch.tensor(all_targets)
indices = torch.randperm(len(all_tensors))
all_tensors = all_tensors[indices]
all_targets = all_targets[indices]

train_tensors, test_tensors, train_targets, test_targets = train_test_split(
    all_tensors, all_targets, test_size=0.2, random_state=42
)

test_dataset = CustomDataset(test_tensors, test_targets)
test_loader = DataLoader(
    test_dataset, batch_size=params['train']['batch_size'], shuffle=True)


def test(net, test_dataloader, device):
    net.to(device)

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Точность на тестовых данных: {:.2f} %'.format(accuracy))

    report = classification_report(all_labels, all_predictions, target_names=[
                                   "0", "1"], output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    return accuracy


def load_model(path="model.pth"):
    model = AnimalsClassifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model("model.pth")


test(model, test_loader, device)
