'''
going to use a LabPicsChemistry dataset
waiting for download......
'''

# read image with cv2.imread(path)
# all channels where ignore > 0 (from ignore image) = 0
# open maps for each category as image: 
# map blue channel > 0

import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch

class SemanticSegmentationDataset(Dataset):
    def __init__(self, img_dir, categories=['Vessel']):
        self.img_dir = "data/"+img_dir
        self.length = len(os.listdir("data/"+img_dir))
        self.categories = categories

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        folder_path = self.img_dir + str(idx+1)
        image = read_image(folder_path+"/Image.png")
        ignore = read_image(folder_path+"/Ignore.png")
        maps = []
        for map in os.listdir(self.img_dir+"/SemanticMaps/FullImage/"):
            if map in self.categories:
                maps.append(read_image(map))
        return image, maps
    
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
view_sample(41)