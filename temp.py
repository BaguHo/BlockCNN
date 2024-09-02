import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from logger import Logger
import os
import argparse
import shutil
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
from torchvision import transforms
from model import CNN_Net

# Constants
PICTURE_SIZE = 24
BATCH_SIZE = 32
COLOR = 1
IN_SIZE = 8
OUT_SIZE = 24
ITERATE_SIZE = int((PICTURE_SIZE - (2 * IN_SIZE)) / IN_SIZE)
NUM_PER_PIC = int(((PICTURE_SIZE - 2 * IN_SIZE) ** 2) / (IN_SIZE**2))
TOTAL_PIC = 3000
TOTAL_PIC_TEST = 10
FILTER_K = 64
NUM_PIC_SHOW = 1
BEST_LOSS = 1e8
ROOT = "../high-frequency-datasets/m-gaid-dataset-high-frequency/denoised"
csv_path = "../high-frequency-datasets/m-gaid-dataset-high-frequency/classified_label.csv"


def extract_y_channel_from_yuv_with_patch_numbers(
    yuv_file_path: str, width: int, height: int
):
    y_size = width * height
    patches, patch_numbers = [], []

    if not os.path.exists(yuv_file_path):
        print(f"Warning: File {yuv_file_path} does not exist.")
        return [], []

    with open(yuv_file_path, "rb") as f:
        y_data = f.read(y_size)

    if len(y_data) != y_size:
        print(f"Warning: Expected {y_size} bytes, got {len(y_data)} bytes.")
        return [], []

    y_channel = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))

    for i in range(0, height, 24):
        for j in range(0, width, 24):
            patch = y_channel[i: i + 24, j: j + 24]
            if patch.shape[0] < 24 or patch.shape[1] < 24:
                patch = np.pad(
                    patch,
                    ((0, 24 - patch.shape[0]), (0, 24 - patch.shape[1])),
                    "constant",
                )
            patches.append(patch)

    return patches


def load_data_from_csv(csv_path, max_samples=300):
    df = pd.read_csv(csv_path)

    all_denoised_patches = []

    df = df.sample(n=min(max_samples, len(df))).reset_index(drop=True)

    for _, row in df.iterrows():
        denoised_file_name = f"denoised_{row['image_name']}.raw"
        denoised_path = os.path.join(ROOT, denoised_file_name)

        denoised_patches = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])

        all_denoised_patches.extend(denoised_patches)

    return all_denoised_patches


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Image Generating")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-o",
        "--save_dir",
        type=str,
        default="./save",
        help="Location for parameter checkpoints and samples",
    )
    return parser.parse_args()


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_np(x):
    return x.data.cpu().numpy()


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(os.path.dirname(filename), "model_best.pth.tar")
        )


# class My_dataloader(Dataset):
#     def __init__(self, data_24, transform):
#         self.data_24 = data_24
#         self.pathes_24 = list(glob(data_24))
#         self.transform = transform

#     def __len__(self):
#         return len(self.pathes_24)

#     def __getitem__(self, idx):
#         raw_image_path = self.pathes_24[idx]

#         width, height = PICTURE_SIZE, PICTURE_SIZE
#         with open(raw_image_path, "rb") as f:
#             img_data = np.frombuffer(f.read(), dtype=np.uint8)

#         img_24 = img_data.reshape((height, width, COLOR))

#         if self.transform:
#             img_24 = Image.fromarray(img_24)
#             img_24 = self.transform(img_24)
#             img_8 = img_24[:, 6:18, 6:18]

#         return img_24 * 255.0, img_8 * 255.0
#         # return img_24

class My_dataloader(Dataset):
    def __init__(self, patches, transform):
        self.patches = patches
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]

        # 데이터 변환 수행
        if self.transform:
            patch = Image.fromarray(patch)  # NumPy 배열을 PIL 이미지로 변환
            patch = self.transform(patch)   # 변환 적용
            img_8 = patch[:, 6:18, 6:18]    # 8x8 이미지 추출

        return patch * 255., img_8 * 255.


def main():
    args = parse_args()
    create_dir(args.save_dir)
    create_dir(os.path.join(args.save_dir, "Generated_Pic/train"))
    create_dir(os.path.join(args.save_dir, "Generated_Pic/test"))
    create_dir(os.path.join(args.save_dir, "checkpoint"))
    create_dir(os.path.join(args.save_dir, "logs"))

    model = CNN_Net()
    model.apply(weight_init)

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    transform = transforms.Compose([
        transforms.Resize((PICTURE_SIZE, PICTURE_SIZE)),
        transforms.ToTensor()
    ])

    all_denoised_patches = load_data_from_csv(csv_path, max_samples=100)
    train_data_24 = My_dataloader(all_denoised_patches, transform)
    train_loader = DataLoader(train_data_24, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    for epoch in range(1):
        model.train()
        num_batches = len(train_loader)
        print(f"Number of batches: {num_batches}")
        for i, (data_24, data_8) in enumerate(train_loader):
            optimizer.zero_grad()
            input = data_24[:, :, 0:16, :].clone().detach().float()
            input[:, :, 8:16, 8:24] = input[:, :, 0:8, 0:24].mean(
                dim=-1, keepdim=True).mean(dim=-2, keepdim=True).expand_as(input[:, :, 8:16, 8:24])
            target = data_8.clone().detach().float()
            out = model(input)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            print('Batch: {} \tLoss: {:.6f}'.format(i, loss.item()))

        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(args.save_dir, 'checkpoint', 'checkpoint.pth.tar'))


if __name__ == "__main__":
    main()
