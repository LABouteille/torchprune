import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

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


def evaluate(device, model, criterion, val_loader):

    val_loss_running, val_acc_running = 0, 0

    model.eval().cuda() if torch.cuda.is_available() else model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, dim=1)
            val_loss_running += loss.item() * inputs.shape[0]
            val_acc_running += torch.sum(predictions == labels.data)

        val_loss = val_loss_running / len(val_loader.sampler)
        val_acc = val_acc_running / len(val_loader.sampler)

    return val_loss, val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", help="train", nargs="?", type=bool, const=True, default=False
    )
    args = parser.parse_args()

    seed_everything(42)
    lr = 0.02
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, _, _ = MNISTloader(train_val_split=0.95).load()

    if args.train:
        summary(model, input_size=(1, 1, 32, 32))

        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                device, lr, model, optimizer, criterion, train_loader
            )
            info = "Epoch: {:3}/{} \t train_loss: {:.3f} \t train_acc: {:.3f}"
            print(info.format(epoch + 1, num_epochs, train_loss, train_acc))

        torch.save(model.state_dict(), "./model.pt")
    else:
        model.load_state_dict(torch.load("./model.pt"))

        # set the qconfig for PTQ
        qconfig = torch.quantization.get_default_qconfig("qnnpack")
        # or, set the qconfig for QAT
        qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
        # set the qengine to control weight packing
        torch.backends.quantized.engine = "qnnpack"

        model_quant = torch.quantization.quantize_dynamic(
            model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
        )

        start = time.time()
        val_loss, val_acc = evaluate(device, model, criterion, train_loader)
        end = time.time()

        start_q = time.time()
        val_loss_q, val_acc_q = evaluate(device, model_quant, criterion, train_loader)
        end_q = time.time()

        print("Floating point FP32")
        print(model)
        print(f"val_loss: {val_loss:.3f} \t val_acc: {val_acc:.3f}")
        print(f"Latency: {end - start}")

        print()

        print("Quantized INT8")
        print(model_quant)
        print(f"val_loss_q: {val_loss_q:.3f} \t val_acc_q:{val_acc_q:.3f}")
        print(f"Latency: {end_q - start_q}")
