import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from zoo.dataset.mnist import MNISTloader
from zoo.model.mnist.simplenet import SimpleNet


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(device, lr, model, optimizer, criterion, train_loader):

    train_loss_running, train_acc_running = 0, 0

    model.train().cuda() if torch.cuda.is_available() else model.train()

    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        _, predictions = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_running += loss.item() * inputs.shape[0]
        train_acc_running += torch.sum(predictions == labels.data)

    train_loss = train_loss_running / len(train_loader.sampler)
    train_acc = train_acc_running / len(train_loader.sampler)

    return train_loss, train_acc


if __name__ == "__main__":
    seed_everything(42)
    lr = 0.043
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, _, _ = MNISTloader(train_val_split=0.95).load()

    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            device, lr, model, optimizer, criterion, train_loader
        )
        info = "Epoch: {:3}/{} \t train_Loss: {:.3f} \t train_acc: {:.3f}"
        print(info.format(epoch + 1, num_epochs, train_loss, train_acc))
