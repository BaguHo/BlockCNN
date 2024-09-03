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
from torchvision import transforms

# Constants
# TODO: image_height and image_width need to change
image_height, image_width = 24, 24
BATCH_SIZE = 32
FILTER_K = 64
BEST_LOSS = 1e8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size = 24
denoised_dir = '../high-frequency-datasets/m-gaid-dataset-high-frequency/denoised'
csv_path = '../high-frequency-datasets/m-gaid-dataset-high-frequency/classified_label.csv'

# transform = transforms.Compose([
#     transforms.Resize((PICTURE_SIZE, PICTURE_SIZE)),
#     transforms.ToTensor()
# ])


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


def load_data_from_csv(csv_path, denoised_dir):
    df = pd.read_csv(csv_path)

    all_denoised_patches = []
    all_scores = []
    denoised_image_names = []
    all_patch_numbers = []

    for _, row in df.iterrows():
        denoised_file_name = f"denoised_{row['image_name']}.png"

        denoised_path = os.path.join(denoised_dir, denoised_file_name)

        denoised_patches, denoised_patch_numbers = extract_patches_from_rgb_image(denoised_path, 24)

        all_denoised_patches.extend(denoised_patches)
        denoised_image_names.extend([row['image_name']] * len(denoised_patches))
        all_patch_numbers.extend(denoised_patch_numbers)

        patch_scores = row['patch_score'].strip('[]').replace(',', ' ').split()
        scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])

        all_scores.extend(scores)

    return all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers


class My_dataloader(Dataset):
    def __init__(self, data_24, transform):
        self.data_24 = data_24
        self.pathes_24 = list(glob(self.data_24))
        self.transform = transform

    def __len__(self):
        return len(self.pathes_24)

    def __getitem__(self, idx):
        raw_image_path = self.pathes_24[idx]

        width, height = 24, 24
        with open(raw_image_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), dtype=np.uint8)

        img_24 = img_data.reshape((height, width, 1))

        if self.transform:
            img_24 = Image.fromarray(img_24)
            img_24 = self.transform(img_24)
            img_8 = img_24[:, 6:18, 6:18]

        return img_24 * 255., img_8 * 255.




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


def training_model():
    # Parse arguments
    args = parse_args()

    # Prepare directories
    create_dir(args.save_dir)
    create_dir(os.path.join(args.save_dir, "Generated_Pic/train"))
    create_dir(os.path.join(args.save_dir, "Generated_Pic/test"))
    create_dir(os.path.join(args.save_dir, "checkpoint"))
    create_dir(os.path.join(args.save_dir, "logs"))

    # Load model
    model = CNN_Net().to(DEVICE)
    model.apply(weight_init).to(DEVICE)
    # Load loss function
    loss_fn = nn.MSELoss()

    # Load optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Set up data set
    train_data_24, _, _, _ = load_data_from_csv(csv_path, denoised_dir)
    # train_data_24 = My_dataloader(train_data_24)
    train_data_8 = train_data_24[:, :, 8:16, 8:16]
    train_loader = DataLoader(train_data_24, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Train model
    # epoch 500 to 10
    for epoch in range(10):
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
            print('Train i: {} \tLoss: {:.6f}'.format(i, loss.item()))

        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

    # Save checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(args.save_dir, 'checkpoint', 'checkpoint.pth.tar'))


if __name__ == '__main__':
    training_model().to(DEVICE)
