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
from tqdm import tqdm

IMAGE_SIZE = 224
PATCH_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
COLOR_CHANNELS = 3
RESULTS_DIR = "./results"
CHECKPOINT_INTERVAL = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Current cuda device:", torch.cuda.current_device())
print("Count of using GPUs:", torch.cuda.device_count())

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def append_to_file(text):
    with open("result.txt", "a") as file:
        file.write(text + "\n")


class CustomDataset(Dataset):
    def __init__(self, original_dir, denoised_dir, csv_path, transform=None):
        self.original_dir = original_dir
        self.denoised_dir = denoised_dir
        self.transform = transform
        self.all_original_patches, self.all_denoised_patches = load_data_from_csv(
            csv_path, original_dir, denoised_dir
        )

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
    if image.mode != "RGB":
        print(f"Warning: Expected an RGB image, got {image.mode}.")
        return [], []

    width, height = image.size
    image_array = np.array(image)
    patch_number = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_array[i: i + patch_size, j: j + patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(
                    patch,
                    (
                        (0, patch_size - patch.shape[0]),
                        (0, patch_size - patch.shape[1]),
                        (0, 0),
                    ),
                    "constant",
                )
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


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

original_dir = "./dataset/m-gaid-dataset-high-frequency/original"
denoised_dir = "./dataset/m-gaid-dataset-high-frequency/denoised"
# original_dir = "/Users/Yihyun_1/나/학교/연구실/ghosting-artifact-metric/dataset/dataset_patch_raw_ver3/original"
# denoised_dir = "/Users/Yihyun_1/나/학교/연구실/ghosting-artifact-metric/dataset/dataset_patch_raw_ver3/denoised"

csv_path = "./Non_Zeros_Classified_label_filtered.csv"

if not os.path.exists(original_dir):
    print(f"Original file {original_dir} does not exist.")
if not os.path.exists(denoised_dir):
    print(f"Denoised file {denoised_dir} does not exist.")

dataset = CustomDataset(original_dir, denoised_dir, csv_path, transform=transform)

train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

if not torch.cuda.is_available():
    print("CUDA is not available. Please check your GPU configuration.")
    exit()

print(f"Train dataset size: {len(train_data)}")


class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.relu(out)

        return out


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        k = 64
        self.conv_1 = nn.Conv2d(
            COLOR_CHANNELS, k, (3, 5), (1, 1), padding=(1, 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(k)

        self.layer_1 = BottleNeck(k, k)
        self.layer_2 = BottleNeck(k, k)

        self.conv_2 = nn.Conv2d(k, k * 2, (3, 5), (1, 1), padding=(1, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(k * 2)

        self.layer_3 = BottleNeck(k * 2, k * 2)

        self.conv_3 = nn.Conv2d(
            k * 2, k * 4, (1, 5), (1, 1), padding=(0, 2), bias=False
        )
        self.bn3 = nn.BatchNorm2d(k * 4)

        self.layer_4 = BottleNeck(k * 4, k * 4)
        self.layer_5 = BottleNeck(k * 4, k * 4)

        self.conv_4 = nn.Conv2d(
            k * 4, k * 8, (1, 1), (1, 1), padding=(0, 0), bias=False
        )
        self.bn4 = nn.BatchNorm2d(k * 8)

        self.layer_6 = BottleNeck(k * 8, k * 8)

        self.conv_5 = nn.Conv2d(k * 8, k * 4, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(k * 4)

        self.layer_7 = BottleNeck(k * 4, k * 4)

        self.conv_6 = nn.Conv2d(k * 4, k * 2, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(k * 2)

        self.conv_7 = nn.Conv2d(k * 2, COLOR_CHANNELS, 1, 1, 0, bias=False)

    def forward(self, x):
        x = x.squeeze(1)
        out = F.relu(self.bn1(self.conv_1(x)))
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = F.relu(self.bn2(self.conv_2(out)))
        out = self.layer_3(out)
        out = F.relu(self.bn3(self.conv_3(out)))
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = F.relu(self.bn4(self.conv_4(out)))
        out = self.layer_6(out)
        out = F.relu(self.bn5(self.conv_5(out)))
        out = self.layer_7(out)
        out = F.relu(self.bn6(self.conv_6(out)))
        out = torch.sigmoid(self.conv_7(out))

        return out


model = CNN_Net().cuda()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)

model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
early_stopping_patience = 10
best_val_loss = float("inf")
epochs_no_improve = 0

print("Training started.")

# for epoch in range(EPOCHS):
#     model.train()
#     train_loss = 0.0
#     print(len(train_loader))
#     for batch_idx, (inputs, targets) in enumerate(
#         tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
#     ):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}")

#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs, targets = inputs.cuda(), targets.cuda()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         early_stopping_counter = 0
#         torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
#         print(f"New best model saved with validation loss: {val_loss:.4f}")
#     else:
#         early_stopping_counter += 1

#     if early_stopping_counter >= early_stopping_patience:
#         print("Early stopping triggered.")
#         break

#     scheduler.step(val_loss)

model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth")))
model.eval()

denoised_patch, original_psnr_scores, psnr_scores, original_ssim_scores, ssim_scores = [], [], [], [], []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        for i in range(len(outputs)):
            # denoised_patch.append(cv2.GaussianBlur(targets[i], (5, 5), 0))
            denoised_patch.append(cv2.GaussianBlur(targets[i].astype(np.float32), (5, 5), 0))

            psnr_scores.append(
                psnr(denoised_patch[i], outputs[i])
            )  # denoise:output image
            inputs.cpu().numpy ()
            original_psnr_scores.append(
                psnr(inputs[i], denoised_patch[i])
            )  # original:denoise image

            patch_size = min(outputs[i].shape[0], outputs[i].shape[1])
            win_size = min(7, patch_size)

            if win_size >= 3:
                ssim_val = ssim(
                    targets[i],
                    outputs[i],
                    win_size=win_size,
                    channel_axis=-1,
                    data_range=1.0,
                )
                original_ssim_val = ssim(
                    inputs[i],
                    denoised_patch[i],
                    win_size=win_size,
                    channel_axis=-1,
                    data_range=1.0,
                )
                ssim_scores.append(ssim_val)
                original_ssim_scores.append(original_ssim_val)
            else:
                print(f"Skipping SSIM for patch {i} due to insufficient size")

            # output_image = outputs[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            # output_image = (output_image * 255).astype(np.uint8)  # [0, 255] 범위로 변환
            # pil_image = Image.fromarray(output_image)  # numpy 배열을 PIL 이미지로 변환

            # # 파일명 생성
            # output_filename = f"output_{i}_{batch_idx}.png"
            # output_path = os.path.join(RESULTS_DIR, output_filename)

            # # 이미지 저장
            # pil_image.save(output_path)
            # print(f"Saved output image: {output_path}")

avg_psnr = np.mean(psnr_scores)
avg_original_psnr = np.mean(original_psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
avg_original_ssim = np.mean(original_ssim_scores) if original_ssim_scores else 0

append_to_file(
    f"Average Filtered PSNR: {avg_original_psnr:.4f} | Average PSNR: {avg_psnr:.4f} | Average Filtered SSIM: {avg_original_ssim:.4f} | Average SSIM: {avg_ssim:.4f}")

print(f"Average Filtered PSNR: {avg_original_psnr:.4f}")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average Filtered SSIM: {avg_original_ssim:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
