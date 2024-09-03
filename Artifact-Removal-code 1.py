#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random

patch_size = 24
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_patches_from_rgb_image(image_path: str, patch_size: int):
    patches, patch_numbers = [], []
    if not os.path.exists(image_path):
        print(f"Warning: File {image_path} does not exist.")
        return [], []

    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    image_array = np.array(image)
    patch_number = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_array[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]),
                               (0, patch_size - patch.shape[1]), (0, 0)), 'constant')
            patches.append(patch)
            patch_numbers.append(patch_number)
            patch_number += 1

    return patches, patch_numbers


def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)

    all_original_patches = []
    all_denoised_patches = []
    all_scores = []
    denoised_image_names = []
    all_patch_numbers = []

    for _, row in df.iterrows():
        original_file_name = f"original_{row['image_name']}.png"
        denoised_file_name = f"denoised_{row['image_name']}.png"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)

        # denoised_patches, denoised_patch_numbers = extract_patches_from_rgb_image(denoised_path, patch_size)

        # all_denoised_patches.extend(denoised_patches)
        denoised_patches, denoised_patch_numbers = extract_patches_from_rgb_image(denoised_path, patch_size)
        all_denoised_patches.extend(denoised_patches)
        denoised_image_names.extend([row['image_name']] * len(denoised_patches))
        all_patch_numbers.extend(denoised_patch_numbers)

        patch_scores = row['patch_score'].strip('[]').replace(',', ' ').split()
        scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])

        all_scores.extend(scores)

    # 리스트를 NumPy 배열로 변환하여 다차원 인덱싱 가능하게 변경
    all_denoised_patches_np = np.array(all_denoised_patches)

    # 8x8 중심 패치를 추출
    denoised_patches_24 = all_denoised_patches_np * 255
    denoised_patchs_8 = all_denoised_patches_np[:, 6:18, 6:18] * 255

    return all_original_patches, denoised_patches_24, denoised_patchs_8, all_scores, denoised_image_names, all_patch_numbers


original_dir = '../m-gaid-dataset-high-frequency/original'
denoised_dir = '../m-gaid-dataset-high-frequency/denoised'
csv_path = '../m-gaid-dataset-high-frequency/classified_label.csv'


original_patches, denoised_patches_24, denoised_patches_8, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(
    csv_path, original_dir, denoised_dir)

num_additional_labels = len(denoised_patches_24) - len(labels)
additional_labels = [random.choice([0, 1]) for _ in range(num_additional_labels)]
labels.extend(additional_labels)


X_train, X_temp, X_train_denoised, X_temp_denoised, y_train, y_temp = train_test_split(
    denoised_patches_24, denoised_patches_8, labels, test_size=0.2, random_state=42)
X_val, X_test, X_val_denoised, X_test_denoised, y_val, y_test = train_test_split(
    X_temp, X_temp_denoised, y_temp, test_size=0.5, random_state=42)


class PatchDataset(Dataset):
    def __init__(self, original_patches, denoised_patches, labels, transform=None):
        self.original_patches = original_patches
        self.denoised_patches = denoised_patches
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.original_patches)

    def __getitem__(self, idx):
        original_patch = self.original_patches[idx]
        denoised_patch = self.denoised_patches[idx]
        label = self.labels[idx]

        if self.transform:
            original_patch = self.transform(original_patch)
            denoised_patch = self.transform(denoised_patch)

        original_patch = torch.tensor(original_patch).permute(2, 0, 1).float() / 255.0
        denoised_patch = torch.tensor(denoised_patch).permute(2, 0, 1).float() / 255.0

        return denoised_patch, original_patch, label


class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        color = 3
        k = 64

        self.conv_1 = nn.Conv2d(color, k, (3, 5), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(k)

        self.layer_1 = BottleNeck(k, k)
        self.layer_2 = BottleNeck(k, k)

        self.conv_2 = nn.Conv2d(k, k*2, (3, 5), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(k*2)

        self.layer_3 = BottleNeck(k*2, k*2)

        self.conv_3 = nn.Conv2d(k*2, k*4, (1, 5), (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(k*4)

        self.layer_4 = BottleNeck(k*4, k*4)
        self.layer_5 = BottleNeck(k*4, k*4)

        self.conv_4 = nn.Conv2d(k*4, k*8, (1, 1), (1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(k*8)

        self.layer_6 = BottleNeck(k*8, k*8)

        self.conv_5 = nn.Conv2d(k*8, k*4, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(k*4)

        self.layer_7 = BottleNeck(k*4, k*4)

        self.conv_6 = nn.Conv2d(k*4, k*2, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(k*2)

        self.layer_8 = BottleNeck(k*2, k*2)

        self.conv_7 = nn.Conv2d(k*2, k, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(k)

        self.layer_9 = BottleNeck(k, k)

        self.conv_8 = nn.Conv2d(k, color, 1, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.relu(self.bn2(self.conv_2(x)))
        x = self.layer_3(x)
        x = self.relu(self.bn3(self.conv_3(x)))
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.relu(self.bn4(self.conv_4(x)))
        x = self.layer_6(x)
        x = self.relu(self.bn5(self.conv_5(x)))
        x = self.layer_7(x)
        x = self.relu(self.bn6(self.conv_6(x)))
        x = self.layer_8(x)
        x = self.relu(self.bn7(self.conv_7(x)))
        x = self.layer_9(x)
        x = self.conv_8(x)
        x = self.sig(x)
        x = x * 255.0

        return x


def calculate_metrics(original, restored):
    original = original.squeeze().permute(1, 2, 0).cpu().numpy()
    restored = restored.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    ssim_val = ssim(original, restored, win_size=7, channel_axis=-1, data_range=1.0)
    psnr_val = psnr(original, restored, data_range=1.0)

    original_tensor = torch.tensor(original).permute(2, 0, 1).unsqueeze(0)
    restored_tensor = torch.tensor(restored).permute(2, 0, 1).unsqueeze(0)

    return ssim_val, psnr_val


train_dataset = PatchDataset(X_train, X_train_denoised, y_train)
val_dataset = PatchDataset(X_val, X_val_denoised, y_val)
test_dataset = PatchDataset(X_test, X_test_denoised, y_test)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)


model = CNN_Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)
criterion = nn.MSELoss()


epochs = 10

# Training
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for _, (data_24, data_8) in enumerate(train_loader):
        optimizer.zero_grad()
        input = data_24[:, :, 0:16, :].clone().detach().float()
        input[:, :, 8:16, 8:24] = input[:, :, 0:8, 0:24].mean(
            dim=-1, keepdim=True).mean(dim=-2, keepdim=True).expand_as(input[:, :, 8:16, 8:24])
        target = data_8.clone().detach().float()
        out = model(input)
        loss = criterion(out, denoised_patches_8)
        loss.backward()
        optimizer.step()
        print('Train i: {} \tLoss: {:.6f}'.format(loss.item()))

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader)}")

    # for i, (denoised_patches_24, denoised_patches_8, _) in enumerate(train_loader):
    #     denoised_patches_24 = denoised_patches_24.to(device)
    #     denoised_patches_8 = denoised_patches_8.to(device)

    #     optimizer.zero_grad()
    #     restored_patches = model(denoised_patches_24)
    #     loss = criterion(restored_patches, denoised_patches_8)
    #     loss.backward()
    #     optimizer.step()

    #     train_loss += loss.item()

    # print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader)}")

    # # Validation
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for i, (denoised_patches, original_patches, _) in enumerate(val_loader):
    #         denoised_patches = denoised_patches.to(device)
    #         original_patches = original_patches.to(device)

    #         restored_patches = model(denoised_patches)
    #         loss = criterion(restored_patches, original_patches)
    #         val_loss += loss.item()

    #         if i == 0:
    #             ssim_val, psnr_val = calculate_metrics(original_patches[0], restored_patches[0])
    #             print(f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}")

    # print(f"Validation Loss: {val_loss/len(val_loader)}")


def test_and_visualize(model, test_loader):
    model.eval()

    with torch.no_grad():
        displayed = 0
        for i, (denoised_patches, original_patches, labels) in enumerate(test_loader):
            if labels.item() == 1:
                denoised_patches = denoised_patches
                original_patches = original_patches

                restored_patches = model(denoised_patches)
                ssim_val, psnr_val = calculate_metrics(original_patches[0], restored_patches[0])

                print(f"Image {i + 1}: SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

                visualize_restoration(original_patches[0], denoised_patches[0], restored_patches[0])

                displayed += 1
                if displayed == 5:
                    break


def visualize_restoration(original, denoised, restored):

    print(f"Original patch min-max: {original.min().item()}-{original.max().item()}")
    print(f"Denoised patch min-max: {denoised.min().item()}-{denoised.max().item()}")
    print(f"Restored patch min-max: {restored.min().item()}-{restored.max().item()}")

    original = (original * 255).clamp(0, 255).byte()
    denoised = (denoised * 255).clamp(0, 255).byte()
    restored = (restored * 255).clamp(0, 255).byte()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original")
    axs[1].imshow(denoised.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Denoised")
    axs[2].imshow(restored.permute(1, 2, 0).cpu().numpy())
    axs[2].set_title("Restored")
    for ax in axs:
        ax.axis('off')
    plt.show()


test_dataset = PatchDataset(X_test, X_test_denoised, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


test_and_visualize(model, test_loader)
