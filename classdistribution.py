import torch

def class_distribution(dataset, classes):
    class_list = torch.zeros(classes,dtype=torch.long)
    for _, mask, _ in dataset:
        mask = mask.flatten()
        counts = torch.bincount(mask, minlength=classes)
        class_list += counts

    total = class_list.sum().item()
    for index in range(classes):
        print(f'Class {index}: {(class_list[index].item() / total * 100):.4f}%')


