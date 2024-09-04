import os
import cv2
import torch
import shutil
import logging
from glob import glob
import numpy as np
import torch.nn as nn
from PIL import Image
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

IMAGE_SIZE = 224
PATCH_SIZE = 224
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
weight_decay = 1e-4
EPOCHS = 10
COLOR_CHANNELS = 3
RESULTS_DIR = '/ghosting-artifact-metric/Code/'
CHECKPOINT_INTERVAL = 5


if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class CustomDataset(Dataset):
    def __init__(self, original_dir, denoised_dir, csv_path, transform=None):
        self.original_dir = original_dir
        self.denoised_dir = denoised_dir
        self.transform = transform

        self.all_original_patches, self.all_denoised_patches = load_data_from_csv(csv_path, original_dir, denoised_dir)

    def __len__(self):
        return len(self.all_original_patches)

    def __getitem__(self, idx):
        original_patch = self.all_original_patches[idx]
        denoised_patch = self.all_denoised_patches[idx]

        original_patch = Image.fromarray(original_patch)
        denoised_patch = Image.fromarray(denoised_patch)

        if self.transform:
            original_patch = self.transform(original_patch)
            denoised_patch = self.transform(denoised_patch)

        return original_patch, denoised_patch


def extract_patches_from_rgb_image(image_path: str, patch_size: int = 224):
    patches = []

    if not os.path.exists(image_path):
        print(f"Warning: File {image_path} does not exist.")
        return [], []

    image = Image.open(image_path)
    if image.mode != 'RGB':
        print(f"Warning: Expected an RGB image, got {image.mode}.")
        return [], []

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

    return patches


def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)

    all_original_patches = []
    all_denoised_patches = []

    for _, row in df.iterrows():

        original_file_name = f"original_{row['image_name']}.png"
        denoised_file_name = f"denoised_{row['image_name']}.png"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)

        original_patches = extract_patches_from_rgb_image(original_path)
        denoised_patches = extract_patches_from_rgb_image(denoised_path)

        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)

    return all_original_patches, all_denoised_patches


transform = transforms.Compose([
    transforms.ToTensor(),
])

original_dir = '../m-gaid-dataset-high-frequency/original'
denoised_dir = '../m-gaid-dataset-high-frequency/denoised'
csv_path = '../m-gaid-dataset-high-frequency/classified_label.csv'

dataset = CustomDataset(original_dir, denoised_dir, csv_path, transform=transform)

train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class DMCNN(nn.Module):
    def __init__(self):
        super(DMCNN, self).__init__()

        # DCT Domain Network
        self.dct_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dct_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dct_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dct_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dct_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Pixel Domain Network
        self.pixel_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pixel_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pixel_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pixel_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pixel_conv5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Batch Normalization
        self.bn = nn.BatchNorm2d(64)

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # DCT domain processing
        dct_x = self.relu(self.bn(self.dct_conv1(x)))
        dct_x = self.relu(self.bn(self.dct_conv2(dct_x)))
        dct_x = self.relu(self.bn(self.dct_conv3(dct_x)))
        dct_x = self.relu(self.bn(self.dct_conv4(dct_x)))
        dct_output = self.dct_conv5(dct_x)

        # Pixel domain processing
        pixel_x = self.relu(self.bn(self.pixel_conv1(x)))
        pixel_x = self.relu(self.bn(self.pixel_conv2(pixel_x)))
        pixel_x = self.relu(self.bn(self.pixel_conv3(pixel_x)))
        pixel_x = self.relu(self.bn(self.pixel_conv4(pixel_x)))
        pixel_output = self.pixel_conv5(pixel_x)

        # Combine outputs
        output = dct_output + pixel_output  # 결과를 가중치 합으로 결합
        return output


model = DMCNN()
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'Best_model.pth'))
        print(f"New best model saved with validation loss: {val_loss:.4f}")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

    scheduler.step(val_loss)


model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'Best_model.pth')))
model.eval()

psnr_scores, ssim_scores = [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        for i in range(len(outputs)):
            psnr_scores.append(psnr(targets[i], outputs[i]))

            patch_size = min(outputs[i].shape[0], outputs[i].shape[1])
            win_size = min(7, patch_size)

            if win_size >= 3:
                ssim_val = ssim(targets[i], outputs[i], win_size=win_size, channel_axis=-1, data_range=1.0)
                ssim_scores.append(ssim_val)
            else:
                print(f"Skipping SSIM for patch {i} due to insufficient size")

avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
