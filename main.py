import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import argparse
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model import CNN_Net
from glob import glob

# Constants
BATCH_SIZE = 32
FILTER_K = 64
BEST_LOSS = 1e8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size_24 = 24
patch_size_8 = 8


def extract_patches_from_rgb_image(image_path: str, patch_size_24: int, patch_size_8: int):
    patches_24, patches_8, patch_numbers = [], [], []
    if not os.path.exists(image_path):
        print(f"Warning: File {image_path} does not exist.")
        return [], [], []

    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    image_array = np.array(image)
    patch_number = 0

    # Extract 24x24 patches
    for i in range(0, height, patch_size_24):
        for j in range(0, width, patch_size_24):
            patch_24 = image_array[i:i+patch_size_24, j:j+patch_size_24]
            if patch_24.shape[0] < patch_size_24 or patch_24.shape[1] < patch_size_24:
                patch_24 = np.pad(patch_24, ((0, patch_size_24 - patch_24.shape[0]),
                                             (0, patch_size_24 - patch_24.shape[1]), (0, 0)), 'constant')
            patches_24.append(patch_24)

            # Extract corresponding 8x8 patches from the center
            center_i, center_j = i + (patch_size_24 // 2) - (patch_size_8 // 2), j + \
                (patch_size_24 // 2) - (patch_size_8 // 2)
            patch_8 = image_array[center_i:center_i+patch_size_8, center_j:center_j+patch_size_8]
            if patch_8.shape[0] < patch_size_8 or patch_8.shape[1] < patch_size_8:
                patch_8 = np.pad(patch_8, ((0, patch_size_8 - patch_8.shape[0]),
                                           (0, patch_size_8 - patch_8.shape[1]), (0, 0)), 'constant')
            patches_8.append(patch_8)

            patch_numbers.append(patch_number)
            patch_number += 1

    return patches_24, patches_8, patch_numbers


def load_data_from_csv(csv_path, denoised_dir):
    df = pd.read_csv(csv_path)

    all_denoised_patches_24 = []
    all_denoised_patches_8 = []
    all_scores = []
    denoised_image_names = []
    all_patch_numbers = []

    for _, row in df.iterrows():
        denoised_file_name = f"denoised_{row['image_name']}.png"
        denoised_path = os.path.join(denoised_dir, denoised_file_name)

        denoised_patches_24, denoised_patches_8, denoised_patch_numbers = extract_patches_from_rgb_image(
            denoised_path, patch_size_24, patch_size_8)

        if len(denoised_patches_24) == 0:
            print(f"Warning: No patches found for {row['image_name']}. Skipping.")
            continue

        patch_scores = row['patch_score'].strip('[]').replace(',', ' ').split()
        scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])

        if len(scores) < len(denoised_patches_24):
            missing_count = len(denoised_patches_24) - len(scores)
            random_scores = np.random.choice([0, 1], size=missing_count)
            scores = np.concatenate([scores, random_scores])

        if len(scores) != len(denoised_patches_24):
            print(f"Error: Mismatch after filling scores for {row['image_name']} "
                  f"denoised patches: {len(denoised_patches_24)}, scores: {len(scores)}")
            continue

        all_denoised_patches_24.extend(denoised_patches_24)
        all_denoised_patches_8.extend(denoised_patches_8)
        denoised_image_names.extend([row['image_name']] * len(denoised_patches_24))
        all_patch_numbers.extend(denoised_patch_numbers)
        all_scores.extend(scores)

    return all_denoised_patches_24, all_denoised_patches_8, all_scores, denoised_image_names, all_patch_numbers


denoised_dir = '../high-frequency-datasets/m-gaid-dataset-high-frequency/denoised'
csv_path = '../high-frequency-datasets/m-gaid-dataset-high-frequency/classified_label.csv'

denoised_patches_24, denoised_patches_8, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(
    csv_path, denoised_dir)

# Continue with training the model using denoised_patches_24 and denoised_patches_8 as needed...


# Continue with training the model using denoised_patches_24 and denoised_patches_8 as needed...

ROOT = "/home/mlcm/Danial/Image_compression/dataset"

original_dir = '../high-frequency-datasets/m-gaid-dataset-high-frequency/original'
denoised_dir = '../high-frequency-datasets/m-gaid-dataset-high-frequency/denoised'
csv_path = '../high-frequency-datasets/m-gaid-dataset-high-frequency/classified_label.csv'

# X_train, X_temp, X_train_denoised, X_temp_denoised, y_train, y_temp = train_test_split(
#     original_patches, denoised_patches, labels, test_size=0.2, random_state=42)
# X_val, X_test, X_val_denoised, X_test_denoised, y_val, y_test = train_test_split(
#     X_temp, X_temp_denoised, y_temp, test_size=0.5, random_state=42)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Image Generating')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-o', '--save_dir', type=str, default='./save',
                        help='Location for parameter checkpoints and samples')
    return parser.parse_args()


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_np(x):
    return x.data.cpu().numpy()


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))


def main():
    # Parse arguments
    args = parse_args()

    # Prepare directories
    create_dir(args.save_dir)
    create_dir(os.path.join(args.save_dir, "Generated_Pic/train"))
    create_dir(os.path.join(args.save_dir, "Generated_Pic/test"))
    create_dir(os.path.join(args.save_dir, "checkpoint"))
    create_dir(os.path.join(args.save_dir, "logs"))

    # Load model
    model = CNN_Net(FILTER_K)
    model.apply(weight_init)

    # Load loss function
    loss_fn = nn.MSELoss()

    # Load optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Set up data set
    train_data_24 = denoised_patches_24
    train_loader = DataLoader(train_data_24, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Train model
    for epoch in range(500):
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

        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

    # Save checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(args.save_dir, 'checkpoint', 'checkpoint.pth.tar'))


if __name__ == "__main__":
    main()
