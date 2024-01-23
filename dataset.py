'''
going to use a LabPicsChemistry dataset
waiting for download......
'''

# read image with cv2.imread(path)
# all channels where ignore > 0 (from ignore image) = 0
# open maps for each category as image: 
# map blue channel > 0

import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Compose, Resize, Normalize, Lambda, Pad
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np

IMAGE_SIZE = 512

class SemanticSegmentationDataset(Dataset):
    def __init__(self, img_dir, categories=['Vessel']):
        image_path = os.path.join('data', img_dir)
        single_entry_dir = img_dir if img_dir == 'Train' else 'Eval'
        self.image_paths = [os.path.join(image_path, str(i+1) + single_entry_dir) for i in range(len(os.listdir(image_path)))]
        self.length = len(self.image_paths)
        self.categories = categories
        self.transform = Compose([
            Lambda(lambda img: self.resize_and_maybe_pad(img, IMAGE_SIZE)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

    def resize_and_maybe_pad(self, img, target_size):
        # Resize the image
        img = Resize((target_size, target_size))(img)

        # Check if padding is needed
        pad_width = max(target_size - img.shape[1], 0)
        pad_height = max(target_size - img.shape[2], 0)
        if pad_width > 0 or pad_height > 0:
            # Calculate padding for left, top, right, bottom
            padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
            img = Pad(padding)(img)

        return img

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path+"/Image.jpg", ImageReadMode.RGB).to(torch.float)
        image = self.transform(image)


        mask = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.long)

        # Assuming each image folder contains a 'SemanticMaps/FullImage' directory with semantic maps
        maps_dir = os.path.join(image_path, "SemanticMaps", "FullImage")
        if not os.path.exists(maps_dir):
            raise FileNotFoundError(f"Semantic maps directory not found for index {idx}")
        
        available_categories = [os.path.splitext(map_file)[0] for map_file in os.listdir(maps_dir)]
        for i, category in enumerate(self.categories):
            if category in available_categories:
                map_path = os.path.join(maps_dir, category + ".png")  # assuming the file format is .png
                semantic_map = read_image(map_path, ImageReadMode.GRAY)  # convert to grayscale
                semantic_map = self.resize_and_maybe_pad(semantic_map, IMAGE_SIZE).squeeze(0)

                # Update the mask; for pixels in semantic_map with value > 0, set them to the category index
                mask[semantic_map > 0] = i + 1  # +1 if you want to reserve 0 for background

        return image, mask
    

    
# view one sample at index
def view_sample(idx=0, dataset='Train'):
    data_dir = f"data/{dataset}/{idx+1}{dataset}"
    maps = os.listdir(f"{data_dir}/SemanticMaps/FullImage/")
    num_images = len(maps) + 2  # +2 for the base and masked images

    # Calculate the number of rows required, assuming 2 columns for base and masked images
    num_rows = (num_images + 1) // 2  # +1 to handle odd number of images

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))  # Adjust the figure size as needed
    axs = axs.ravel()

    # Base Image
    axs[0].axis("off")
    axs[0].set_title("Base Image", color='white')
    image = read_image(f"{data_dir}/Image.jpg")
    axs[0].imshow(image.permute(1, 2, 0))

    # Masked Image
    axs[1].axis("off")
    axs[1].set_title("Masked Image", color='white')
    ignore = read_image(f"{data_dir}/Ignore.png")
    ignore = ignore.squeeze(0)
    ignore_im = image.clone()
    ignore_im[0][ignore > 0] = 0
    ignore_im[1][ignore > 0] = 0
    ignore_im[2][ignore > 0] = 0
    axs[1].imshow(ignore_im.permute(1, 2, 0))

    # Semantic Maps
    for i, map in enumerate(maps, start=2):
        temp_image = ignore_im.clone()
        axs[i].axis("off")
        axs[i].set_title(map, color='white')
        mask = read_image(f"{data_dir}/SemanticMaps/FullImage/{map}")
        mask = mask[0] > 0
        temp_image[0][mask] = 0
        temp_image[1][mask] = 0
        axs[i].imshow(temp_image.permute(1, 2, 0))

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()



# Call the function with your sample index
# view_sample(41)
    

def plot_sample(dataset, idx, categories, columns=4):
    # Load the image and semantic maps using the dataset class
    image, semantic_maps = dataset[idx]
    
    rows = int(np.ceil((1 + len(semantic_maps)) / float(columns)))
    fig, axs = plt.subplots(rows, columns, figsize=(12, 4*rows))

    # If axs is not an array, make it one for easy indexing
    if rows == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    # Plot the base image
    axs[0].set_title('Base Image', color='green')
    axs[0].axis('off')
    axs[0].imshow(image.permute(1, 2, 0).to(torch.int))

    # Plot each semantic map
    for i, semantic_map in enumerate(semantic_maps):
        # The semantic map tensor is converted to numpy array for display
        semantic_map = semantic_map.permute(1, 2, 0).to(torch.int)  # Remove channel dimension and convert to numpy array
        axs[i+1].imshow(semantic_map, cmap='tab20b')  # Use a colormap that clearly distinguishes classes
        axs[i+1].set_title(f'Semantic Map {categories[i]}', color='green')
        axs[i+1].axis('off')

    # Hide any unused subplots if they exist
    for i in range(1 + len(semantic_maps), len(axs)):
        axs[i].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# Test code to visually verify dataset
# import random
# categories = ["Cork", "Label", "Part", "Spike", "Valve", "MagneticStirer", "Thermometer", "Spatula", "Holder", "Filter", "PipeTubeStraw"]
# dataset = SemanticSegmentationDataset(
#     img_dir='Train',
#     categories=categories
# )
# for i in range(10):
#     plot_sample(dataset, random.randint(1, 5000), categories)