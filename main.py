import numpy as np
import os
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
from resnet_class import ResnetClass
import torch.nn as nn
from PIL import Image
from visuals_class import plot_loss, stitch_images
from classdistribution import class_distribution
from dataset import MyDataset
from tqdm import tqdm
import sys


def machine_learning(epochs, classes, patches = True, layers = 1, back_bone = True):

    # -------------------------------------------------------------------------------

    dataset = MyDataset(stride = 16, patch_size = 32, patches = patches)
    class_distribution(dataset, classes)

    device_status = torch.cuda.is_available()
    device = torch.device('cuda' if device_status else 'cpu')

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size])

    train_loaded_set = DataLoader(train_set, batch_size=4, shuffle=True)
    validation_loaded_set = DataLoader(validation_set, batch_size=4, shuffle=True)

    loss1 = nn.CrossEntropyLoss()
    resnet = ResnetClass(classes = classes, additional_layers = layers, fpn_back_bone = back_bone).to(device)
    weight_manager = torch.optim.Adam(resnet.parameters(), lr=0.001)
    loss_train, loss_eval = [0.0], [0.0]

    # -------------------------------------------------------------------------------

    for epoch in range(epochs):

        loss_validation_epoch, loss_train_epoch = [], []
        print(f'Epoch {epoch + 1}/{epochs}:')

        scores = []

        # -------------------------------------------------------------------------------

        resnet.train()

        with tqdm(total=len(train_loaded_set), desc='Training', file=sys.stdout) as pbar:
            for number, (image, mask, file) in enumerate(train_loaded_set):
                image = image.to(device)
                mask = mask.long().to(device)
                weight_manager.zero_grad()
                output = resnet.forward(image)
                loss = loss1(output, mask)
                loss.backward()
                weight_manager.step()
                loss_train_epoch.append(loss.item())
                pbar.update(1)


        loss_train.append(sum(loss_train_epoch) / len(loss_train_epoch))

        # -------------------------------------------------------------------------------

        resnet.eval()

        with tqdm(total=len(validation_loaded_set), desc='Validating', file=sys.stdout) as pbar:
            with torch.no_grad():
                for number, (image, mask, file) in enumerate(validation_loaded_set):

                    image = image.to(device) #(1, H, W)
                    mask = mask.long().to(device) #(1, H, W)
                    output = resnet.forward(image) #(1, C, H, W)
                    loss = loss1(output, mask) #tensor scalar


                    #-------------------------------------------------------------------------------
                    count = 0
                    pred_class = torch.argmax(output, dim=1) #(B, H, W)
                    for images in range(len(pred_class)):
                        pred_class_np = pred_class[images].cpu().numpy().astype('uint8')
                        pred_class_np *= 50
                        os.makedirs('saved_images', exist_ok=True)
                        if number == count == 0:
                            stitch_images(file[images][-8:], pred_class_np, 'delete')
                        else:
                            stitch_images(file[images][-8:], pred_class_np)
                        count += 1
                    # -------------------------------------------------------------------------------

                    output_array = torch.argmax(output, dim=1).flatten().cpu().numpy() #(1 * H * W)
                    mask_array = mask.flatten().cpu().numpy() #(1 * H * W)
                    scores.append(f1_score(y_true=mask_array, y_pred=output_array, average = 'macro'))

                    loss_validation_epoch.append(loss.item())
                    pbar.update(1)

        loss_eval.append(sum(loss_validation_epoch) / len(loss_validation_epoch))

        # -------------------------------------------------------------------------------

        plot_loss(loss_train, loss_eval, epoch + 1)
        macro_f1 = np.mean(np.array(scores))
        print(f'Epoch {epoch + 1}: {macro_f1:.4f}')

# -------------------------------------------------------------------------------

machine_learning(epochs = 10, classes = 3, patches = False, layers = 1, back_bone = True)










