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
    def __init__(self, img_dir):
        self.img_dir = "data/"+img_dir
        self.length = len(os.listdir("data/"+img_dir))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        folder_path = self.img_dir + str(idx+1)
        image = read_image(folder_path+"/Image.png")
        ignore = read_image(folder_path+"/Ignore.png")
        maps = []
        for map in os.listdir(self.img_dir+"/SemanticMaps/FullImage/"):
            maps.append(read_image(map))
        return image, maps
    
# view one sample at index
def view_sample(idx=0, dataset='Train'):
    data_dir = f"data/{dataset}/{idx+1}{dataset}"
    maps = os.listdir(f"{data_dir}/SemanticMaps/FullImage/")
    num_images = len(maps) + 2  # +2 for the base and masked images

    fig = plt.figure(figsize=(8, 8 * num_images))

    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.title("Base Image")
    image = read_image(f"{data_dir}/Image.jpg")
    plt.imshow(image.permute(1,2,0))

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.title("Masked Image")
    ignore = read_image(f"{data_dir}/Ignore.png")
    ignore = ignore.squeeze(0)
    ignore_im = image.clone()
    ignore_im[0][ignore>0] = 0
    ignore_im[1][ignore>0] = 0
    ignore_im[2][ignore>0] = 0
    plt.imshow(ignore_im.permute(1,2,0))

    for i, map in enumerate(maps, start=3):
        temp_image = ignore_im.clone()
        plt.subplot(num_images, 2, i)
        plt.axis("off")
        plt.title(map)
        mask = read_image(f"{data_dir}/SemanticMaps/FullImage/{map}")
        mask = mask[0] > 0
        temp_image[0][mask] = 0
        temp_image[1][mask] = 0
        plt.imshow(temp_image.permute(1,2,0))

    # show full plot of images
    plt.show()

view_sample(1)