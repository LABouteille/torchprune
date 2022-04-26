import argparse
import csv
import os
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import torchcompress as tc
from zoo.dataset.mnist import MNISTloader
from zoo.model.mnist.simplenet_bn import SimpleNetBN


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def evaluate_model(device, model, dataloader):
    criterion = nn.CrossEntropyLoss()

    running_loss, running_acc = 0, 0

    model.eval().cuda() if torch.cuda.is_available() else model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.shape[0]
            running_acc += torch.sum(predictions == labels.data)

        loss = running_loss / len(dataloader.sampler)
        acc = running_acc / len(dataloader.sampler)

    return loss, acc


def sensitivity_analysis(device, model, sparsities, dataloader):

    sensitivities = OrderedDict()
    x = torch.randn(2, 1, 32, 32)

    for i, (module_name, module) in enumerate(list(model.named_modules())[:-1]):
        if (
            not isinstance(module, torch.nn.Conv2d)
            and not isinstance(module, torch.nn.Linear)
        ) or ():
            continue

        tmp = OrderedDict()

        for sparsity in sparsities:

            model_cpy = deepcopy(model)
            module_cpy = list(model_cpy.named_modules())[i][1]
            DG = tc.DependencyGraph(model_cpy)
            DG.build_dependency_graph(inputs=x)
            pruner = tc.Pruner(DG=DG, dummy_input=x)

            pruner.run(
                layer=module_cpy, criteria=tc.random_criteria, amount_to_prune=sparsity
            )

            loss, acc = evaluate_model(device, model_cpy, dataloader)

            print(f"{module_name} sparsity: {sparsity} | loss = {loss} | acc = {acc}")

            tmp[sparsity] = (loss, acc.item())

            del DG

        sensitivities[module_name] = tmp

    return sensitivities


def dump_sensitivity_to_csv(sensitivities, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "sparsity", "loss", "top1_acc"])
        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform sensitivity analysis")
    parser.add_argument(
        "--model_filepath", required=True, type=str, help="path to trained model."
    )
    parser.add_argument("--csv_name", required=True, type=str, help="csv name.")
    args = parser.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    _, _, test_loader = MNISTloader(shuffle=True).load()

    # Load model
    model = SimpleNetBN()
    model.load_state_dict(torch.load(args.model_filepath, map_location=device))

    # Evaluate model
    test_loss, test_acc = evaluate_model(device, model, test_loader)
    print("Test Loss = {:.3f} | Test Acc = {:.3f}".format(test_loss, test_acc))

    # Perform sensitivity analysis
    sparsities = np.arange(0.0, 1.0, 0.05)
    sensitivities = sensitivity_analysis(device, model, sparsities, test_loader)
    dump_sensitivity_to_csv(sensitivities, args.csv_name)
