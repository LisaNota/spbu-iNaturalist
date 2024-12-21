import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import yaml

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


class CustomDataset(Dataset):
    def __init__(self, tensors, targets):
        self.tensors = tensors
        self.targets = targets

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.targets[idx]


def load_images_from_folder(folder: str, target: int):
    tensors = []
    targets = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".jpg"):
            image = Image.open(file_path).convert("RGB")
            tensor = transform(image)
            tensors.append(tensor)
            targets.append(target)
    return tensors, targets


class AnimalsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.train_block(3, 8, 3, 1, 2)
        self.block2 = self.train_block(8, 16, 3, 1, 2)
        self.block3 = self.train_final_block(32 * 32 * 16, 128, 0.2)

    def train_block(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size - 1, stride=stride),
        )

    def train_final_block(self, in_channels, out_channels, drop_out):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(out_channels, 2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(
            m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
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

    train_dataset = CustomDataset(train_tensors, train_targets)
    train_loader = DataLoader(
        train_dataset, batch_size=params['train']['batch_size'], shuffle=True)

    def train_model(model, train_loader, device, num_epochs=params['train']['num_epochs'], learning_rate=params['train']['learning_rate']):
        model.to(device)
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        acc_history = []

        with tqdm(total=len(train_loader) * num_epochs) as pbar:
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0

                for batch_num, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    pbar.set_description(
                        f"Epoch: {epoch}, Loss: {running_loss:.2f}")
                    pbar.update()

                epoch_acc = correct / total
                acc_history.append(epoch_acc)
            pbar.close()

        return acc_history

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnimalsClassifier()
    history = train_model(model, train_loader, device)

    torch.save(model.state_dict(), "model.pth")
    plt.plot(history)
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy_plot.png")
