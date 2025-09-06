import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

train_data= datasets.CIFAR10(root="data", train=True, transform=train_tf)
test_data= datasets.CIFAR10(root="data", train=False, transform=test_tf)

train_loader= DataLoader(train_data, batch_size =64, shuffle=True, pin_memory=True)
test_loader=DataLoader(test_data, batch_size=64, shuffle=False,pin_memory=True)

model = nn.Sequential(

    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Dropout(p=0.3),
    nn.Linear(256,10)
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

epochs = 30
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
device = torch.device("cuda")
model = model.to(device)

for epoch in range(1, epochs+1):
    model.train()
    running_loss=0

    for X, y in train_loader:
        X= X.to(device, non_blocking=True)
        y =y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X)
        loss= criterion(logits, y)

        loss.backward()
        optimizer.step()
        running_loss+= loss.item()
    avg_loss = running_loss/len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

    model.eval()

    train_correct=0 
    train_total = 0
    with torch.no_grad():
       for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds = model(X).argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
    train_acc = 100.0 * train_correct / train_total
    print(f"Train accuracy: {train_acc:.2f}%")

    correct = 0 
    total =0
    with torch.no_grad():
        for X, y in test_loader:
            X =X.to(device, non_blocking=True)
            y=y.to(device,non_blocking=True)
            logits = model(X)
            result = logits.argmax(dim=1)
            correct+= (result==y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    scheduler.step()
    print(f"Test accuracy: {acc:.2f}%")
