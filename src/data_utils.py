from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision
from pathlib import Path

EXTENSIONS = ["*.png", "*.jpg"]

class PairedImages(Dataset):
    def __init__(self, orig_dir, recon_dir, transforms=None):
        self.orig_dir = Path(orig_dir)
        self.recon_dir = Path(recon_dir)
        self.orig_image_list = []
        self.recon_image_list = []
        for ext in EXTENSIONS:
            self.orig_image_list.extend(self.orig_dir.glob(ext))
            self.recon_image_list.extend(self.recon_dir.glob(ext))
        
        self.transforms = transforms

    def __len__(self):
        return len(self.orig_image_list)

    def __getitem__(self, idx):
        orig_img_path = self.orig_image_list[idx]
        img_basename = os.path.basename(orig_img_path).split(".")[0]
        for ext in EXTENSIONS:
            recon_img_path = os.path.join(self.recon_dir, f"{img_basename}{ext[1:]}")
            if os.path.exists(recon_img_path):
                break 
        
        if self.transforms is None:
            orig_image = torchvision.transforms.ToTensor()(Image.open(orig_img_path))
            if os.path.exists(recon_img_path):
                recon_image = torchvision.transforms.ToTensor()(Image.open(recon_img_path))
            else: 
                recon_image = None 
        else:
            orig_image = self.transforms(Image.open(orig_img_path))
            if os.path.exists(recon_img_path):
                recon_image = self.transforms(Image.open(recon_img_path))
            else: 
                recon_image = None 
        return orig_image, recon_image, img_basename.split(".")[0]


class PairedDuplicateImages(Dataset):
    def __init__(self, duplicate_dir, num_duplicates, recon_dir, transforms=None):
        """
        duplicate_dir corresponds to the original images directory, 
        num_duplicates corresponds to the numer of duplicates of samples
        in duplicate_dir that are reconstructed. 
        recon_dir corresponds to the directory where reconstructed images are stored. 
        """
        self.duplicate_dir = Path(duplicate_dir)
        self.recon_dir = Path(recon_dir)
        self.dup_image_list = []
        self.recon_image_list = []
        for ext in EXTENSIONS:
            self.dup_image_list.extend(self.duplicate_dir.glob(ext))
            self.recon_image_list.extend(self.recon_dir.glob(ext))
        
        assert len(self.dup_image_list) * num_duplicates == len(self.recon_image_list)
        
        self.transforms = transforms

    def __len__(self):
        return len(self.recon_image_list)

    def __getitem__(self, idx):
        recon_img_path = self.recon_image_list[idx]
        # recon_img_path is for instance Indian_90.png or Latino_Hispanic_90.png. 
        # First, we want to extract the ethnicity 
        img_basename = str(os.path.basename(recon_img_path).split(".")[0])
        if img_basename.startswith("Latino_Hispanic"):
            # Latino_Hispanic is the only ethnicity with a "_" within the name 
            img_basename = "Latino_Hispanic"
        else:
            # The other ethnicities can simply be extracted by taking the string 
            # before the first "_". 
            img_basename = img_basename.split("_")[0]
        for ext in EXTENSIONS:
            orig_img_path = os.path.join(self.duplicate_dir, f"{img_basename}{ext[1:]}")
            if os.path.exists(orig_img_path):
                break 
        
        if self.transforms is None:
            orig_image = torchvision.transforms.ToTensor()(Image.open(orig_img_path))
            if os.path.exists(recon_img_path):
                recon_image = torchvision.transforms.ToTensor()(Image.open(recon_img_path))
            else: 
                recon_image = None 
        else:
            orig_image = self.transforms(Image.open(orig_img_path))
            if os.path.exists(recon_img_path):
                recon_image = self.transforms(Image.open(recon_img_path))
            else: 
                recon_image = None 
        return orig_image, recon_image, img_basename.split(".")[0]
