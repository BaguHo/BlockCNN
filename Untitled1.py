#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_dct


# In[2]:


def dct_transform(x):
    return torch_dct.dct_2d(x)

def idct_transform(x):
    return torch_dct.idct_2d(x)


# In[3]:


class DCTLayer(nn.Module):
    def forward(self, x):
        return dct_transform(x)

class IDCTLayer(nn.Module):
    def forward(self, x):
        return idct_transform(x)


# In[4]:


def DRU(x, CDCT, Q):
    lower_bound = CDCT - Q / 2
    upper_bound = CDCT + Q / 2
    return torch.clamp(x, lower_bound, upper_bound)


# In[5]:


class DCTAutoEncoder(nn.Module):
    def __init__(self, depth=9):
        super(DCTAutoEncoder, self).__init__()
        self.encoder = self._make_layers(depth, in_channels=3, dilations=[1, 2, 4])  # Adjust input channels to 3
        self.decoder = self._make_layers(depth, in_channels=64, dilations=[4, 2, 1])

    def _make_layers(self, depth, in_channels, dilations):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=dilations[i % len(dilations)], dilation=dilations[i % len(dilations)]))
            else:
                layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=dilations[i % len(dilations)], dilation=dilations[i % len(dilations)]))
            layers.append(nn.PReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# In[6]:


class PixelAutoEncoder(nn.Module):
    def __init__(self, depth=15):
        super(PixelAutoEncoder, self).__init__()
        self.encoder = self._make_layers(depth, in_channels=3, dilations=[1, 2, 4])  # Adjust input channels to 3
        self.decoder = self._make_layers(depth, in_channels=64, dilations=[4, 2, 1])

    def _make_layers(self, depth, in_channels, dilations):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=dilations[i % len(dilations)], dilation=dilations[i % len(dilations)]))
            else:
                layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=dilations[i % len(dilations)], dilation=dilations[i % len(dilations)]))
            layers.append(nn.PReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# In[7]:


class DMCNN(nn.Module):
    def __init__(self, r=0.5, lambda_param=0.9, theta_param=0.618):
        super(DMCNN, self).__init__()
        self.dct_layer = DCTLayer()
        self.idct_layer = IDCTLayer()
        self.dct_autoencoder = DCTAutoEncoder()
        self.pixel_autoencoder = PixelAutoEncoder()
        self.r = nn.Parameter(torch.tensor(r))
        self.lambda_param = lambda_param
        self.theta_param = theta_param

    def forward(self, x, CDCT, Q):
        dct_data = self.dct_layer(x)  # Apply DCT
        dct_output = self.dct_autoencoder(dct_data)  # Process in DCT domain
        dct_output = DRU(dct_output, CDCT, Q)  # Apply DRU
        pixel_input = self.idct_layer(dct_output)  # Convert back to pixel domain
        pixel_output = self.pixel_autoencoder(pixel_input)  # Process in pixel domain
        final_output = pixel_output + self.r * dct_output + (1 - self.r) * x
        return final_output, pixel_output, dct_output


# In[8]:


def loss_function(final_output, pixel_output, dct_output, target_image, lambda_param=0.9, theta_param=0.618):
    l_mmse = sum(theta_param ** i * nn.MSELoss()(pixel_output, target_image) for i in range(3))
    l_dct = lambda_param * nn.MSELoss()(dct_output, target_image)
    return l_mmse + l_dct


# In[9]:


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
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)), 'constant')
            patches.append(patch)
            patch_numbers.append(patch_number)
            patch_number += 1

    return patches, patch_numbers


# In[10]:


def load_data_from_csv(csv_path, original_dir, denoised_dir, patch_size):
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
        
        original_patches, original_patch_numbers = extract_patches_from_rgb_image(original_path, patch_size)
        denoised_patches, denoised_patch_numbers = extract_patches_from_rgb_image(denoised_path, patch_size)

        if len(original_patches) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches for {row['image_name']}")
            continue
        
        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)
        denoised_image_names.extend([row['image_name']] * len(denoised_patches))
        all_patch_numbers.extend(denoised_patch_numbers)

        patch_scores = row['patch_score'].strip('[]').split(', ')
#         scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])
        
#         if len(scores) != len(original_patches):
#             print(f"Error: Mismatch in number of patches and scores for {row['image_name']}")
#             continue
        
#         all_scores.extend(scores)

#     return all_original_patches, all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers
    return all_original_patches, all_denoised_patches, denoised_image_names, all_patch_numbers


# In[16]:


block_size = 8
batch_size = 32

num_epochs = 50
initial_patch_size = 56
final_patch_size = 224

previous_validation_loss = float('inf')


# In[17]:


original_dir = '/FINAL DATASET/maid-dataset-high-frequency/original'
denoised_dir = '/FINAL DATASET/maid-dataset-high-frequency/denoised'
csv_path = '/FINAL DATASET/Non_Zeros_Classified_label_filtered.csv'

model = DMCNN().to('cuda') 
optimizer = optim.Adam(model.parameters(), lr=0.001)



# In[18]:


standard_quantization_table = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=torch.float32)

def get_quantization_table(QF):
    if QF < 50 and QF > 1:
        scale = 5000 / QF
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        raise ValueError("Quality factor (QF) must be in the range [1, 99].")

    scale = scale / 100
    Q = torch.clamp((standard_quantization_table * scale + 0.5).int(), 1, 255)
    return Q


QF = 20
Q = get_quantization_table(QF).to('cuda')


# In[19]:


def quantize_dct(dct_coefficients, Q):
    return torch.round(dct_coefficients / Q)


# In[20]:


for epoch in range(num_epochs):
    patch_size = initial_patch_size + int((final_patch_size - initial_patch_size) * (epoch / num_epochs))
    
    original_patches, denoised_patches, denoised_image_names, all_patch_numbers = load_data_from_csv(
        csv_path, original_dir, denoised_dir, patch_size)
    
    model.train()
    for i in range(0, len(original_patches), batch_size):
        batch_original = torch.tensor(original_patches[i:i+batch_size]).float().permute(0, 3, 1, 2).to('cuda')
        batch_denoised = torch.tensor(denoised_patches[i:i+batch_size]).float().permute(0, 3, 1, 2).to('cuda')

        CDCT = torch.zeros_like(batch_denoised)

        for h in range(0, batch_denoised.shape[2], block_size):
            for w in range(0, batch_denoised.shape[3], block_size):
                block = batch_denoised[:, :, h:h+block_size, w:w+block_size]
                dct_block = dct_transform(block)
                quantized_block = quantize_dct(dct_block, Q)
                CDCT[:, :, h:h+block_size, w:w+block_size] = quantized_block
        
        optimizer.zero_grad()
        final_output, pixel_output, dct_output = model(batch_denoised, CDCT, Q)
        loss = loss_function(final_output, pixel_output, dct_output, batch_original, model.lambda_param, model.theta_param)
        loss.backward()
        optimizer.step()

    validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(original_patches), batch_size):
            batch_original = torch.tensor(original_patches[i:i+batch_size]).float().permute(0, 3, 1, 2).to('cuda')
            batch_denoised = torch.tensor(denoised_patches[i:i+batch_size]).float().permute(0, 3, 1, 2).to('cuda')
            
            final_output, pixel_output, dct_output = model(batch_denoised, CDCT, Q)
            loss = loss_function(final_output, pixel_output, dct_output, batch_original, model.lambda_param, model.theta_param)
            validation_loss += loss.item()

    if validation_loss > previous_validation_loss:
        for g in optimizer.param_groups:
            g['lr'] /= 3
    
    previous_validation_loss = validation_loss


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define a simple autoencoder for artifact removal
class GeneralArtifactRemovalAutoencoder(nn.Module):
    def __init__(self):
        super(GeneralArtifactRemovalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, optimizer, and loss functions
model = GeneralArtifactRemovalAutoencoder().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()

# Optionally, add perceptual loss (VGG19) to help preserve high-level features
vgg = models.vgg19(pretrained=True).features.to('cuda').eval()

def perceptual_loss(y, y_hat):
    y_vgg = vgg(y)
    y_hat_vgg = vgg(y_hat)
    return mse_loss(y_vgg, y_hat_vgg)

# Training loop
num_epochs = 50
batch_size = 16
previous_validation_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for batch_data in train_loader:  # Assume train_loader is already defined
        noisy_images, clean_images = batch_data
        noisy_images, clean_images = noisy_images.to('cuda'), clean_images.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(noisy_images)
        
        # Combine MSE and perceptual loss
        loss = mse_loss(outputs, clean_images) + perceptual_loss(outputs, clean_images)
        loss.backward()
        optimizer.step()
    
    # Validation loop (optional)
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for batch_data in val_loader:  # Assume val_loader is already defined
            noisy_images, clean_images = batch_data
            noisy_images, clean_images = noisy_images.to('cuda'), clean_images.to('cuda')
            
            outputs = model(noisy_images)
            val_loss = mse_loss(outputs, clean_images) + perceptual_loss(outputs, clean_images)
            validation_loss += val_loss.item()
    
    if validation_loss > previous_validation_loss:
        for g in optimizer.param_groups:
            g['lr'] /= 3
    previous_validation_loss = validation_loss

# Save the model


# In[ ]:




