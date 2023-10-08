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
def view_sample():
    pass