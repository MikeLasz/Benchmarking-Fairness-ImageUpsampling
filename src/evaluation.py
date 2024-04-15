import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision
import torch.nn as nn
import lpips
import os

from tqdm import tqdm

from pytorch_msssim import SSIM  
import numpy as np
import pandas as pd


from src.data_utils import PairedImages
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src.losses import NIQE
from src.losses import LaplaceBlurriness

class RacePredictor(nn.Module):
    """I think I dont need this anymore?!""" # TODO 
    def __init__(self, version="resnet34_7", device="cuda"):
        super(RacePredictor, self).__init__()
        assert version in ["resnet34_7", "resnet34_4"], "Invalid version"
        if version == "resnet34_7":
            self.model_path = 'models/race_prediction/res34_fair_align_multi_7_20190809.pt'
        elif version == "resnet34_4":
            self.model_path = 'models/race_prediction/res34_fair_align_multi_4_20190809.pt'
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(device)
        self.model.eval()
        self.trans = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.version = version
    
    def forward(self, x):
        with torch.no_grad():
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = self.trans(x)
            outputs = self.model(x).cpu().numpy()
            if self.version == "resnet34_7":
                race_outputs = outputs[:, :7]
            elif self.version == "resnet34_4":
                race_outputs = outputs[:, :4]
            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
            race_pred = np.argmax(race_score, axis=1)
            race_pred_str = ["None"] * len(race_pred)
            for j in range(len(race_pred)):
                race_pred_str[j] = code_to_race(race_pred[j], version=self.version)
        return race_pred_str

class RaceDeltaLoss(nn.Module):
    def __init__(self, version="resnet34_7", loss="l2", device="cuda", verbose=False):
        super(RaceDeltaLoss, self).__init__()
        assert version in ["resnet34_7", "resnet34_4"], "Invalid version"
        assert loss in ["l2", "0-1", "cosine", "pred-0-1"], "Invalid loss"
        if version == "resnet34_7":
            self.model_path = 'models/race_prediction/res34_fair_align_multi_7_20190809.pt'
        elif version == "resnet34_4":
            self.model_path = 'models/race_prediction/res34_fair_align_multi_4_20190809.pt'
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(device)
        self.model.eval()
        
        self.trans = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.loss = loss
        self.version = version
        self.verbose = verbose 
        
    def forward(self, x, y):
        with torch.no_grad():
            if len(x.shape)==3:
                x = x.unsqueeze(0)
            if self.loss in ["l2", "cosine"]:
                if len(y.shape) == 3:
                    y = y.unsqueeze(0)
                # compute latent representation of x
                x = self.trans(x)
                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = torch.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                x = self.model.avgpool(x)

                # compute latent representation of y
                y = self.trans(y)
                y = self.model.conv1(y)
                y = self.model.bn1(y)
                y = torch.relu(y)
                y = self.model.maxpool(y)
                y = self.model.layer1(y)
                y = self.model.layer2(y)
                y = self.model.layer3(y)
                y = self.model.layer4(y)
                y = self.model.avgpool(y)

                if self.loss=="l2":
                    # average l2 loss between x and y
                    return MSELoss(x,y)
                elif self.loss=="cosine":
                    # cosine loss between x and y
                    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=1).squeeze()
                    cosine_distance = 1 - cosine_similarity # cosine distance=0 -> perfect, cosine_distance=1 -> worst
                    return cosine_distance
            elif self.loss in ["0-1", "pred-0-1"]:
                x = self.trans(x)
                outputs = self.model(x).cpu().numpy()
                if self.version=="resnet34_7":
                    race_outputs = outputs[:, :7]
                elif self.version=="resnet34_4":
                    race_outputs = outputs[:, :4]
                race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
                race_pred_recon = np.argmax(race_score, axis=1)
                if self.loss == "0-1":
                    race_pred_recon = [code_to_race(race_pred, version=self.version) for race_pred in race_pred_recon]
                    if self.verbose:
                        print(f"True race of original Image: {y}")
                        print(f"Predicted race of estimated Image: {race_pred_recon}")
                    loss = np.array(race_pred_recon) == np.array(y)
                    return loss, race_pred_recon 
                elif self.loss == "pred-0-1":
                    # Estimate the race of the original sample
                    if len(y.shape) == 3:
                        y = y.unsqueeze(0)
                    y = self.trans(y)
                    outputs = self.model(y).cpu().numpy()
                    if self.version == "resnet34_7":
                        race_outputs_orig = outputs[:, :7]
                    elif self.version == "resnet34_4":
                        race_outputs_orig = outputs[:, :4]
                    race_score_orig = np.exp(race_outputs_orig) / np.sum(np.exp(race_outputs_orig))
                    race_pred_orig = np.argmax(race_score_orig)
                    race_pred_recon = [code_to_race(race_pred, version=self.version) for race_pred in race_pred_recon]
                    race_pred_orig = [code_to_race(race_pred, version=self.version) for race_pred in race_pred_orig]
                    if self.verbose:
                        print(f"Predicted race of original Image: {race_pred_orig}")
                        print(f"Predicted race of estimated Image: {race_pred_recon}")
                    loss = np.array(race_pred_orig) == np.array(race_pred_recon)
                    return loss, race_pred_orig, race_pred_recon 


def code_to_race(code, version="resnet34_7"):
    if version=="resnet34_7":
        if code==0:
            return "White"
        elif code==1:
            return "Black"
        elif code==2:
            return "Latino_Hispanic"
        elif code==3:
            return "East Asian"
        elif code==4:
            return "Southeast Asian"
        elif code==5:
            return "Indian"
        elif code==6:
            return "Middle Eastern"
    elif version=="resnet34_4":
        if code==0:
            return "White"
        elif code==1:
            return "Black"
        elif code==2:
            return "Asian"
        elif code==3:
            return "Indian"

def race_to_code(race, version="resnet34_7"):
    if version=="resnet34_7":
        if race=="White":
            return 0
        elif race=="Black":
            return 1
        elif race=="Latino_Hispanic":
            return 2
        elif race=="East Asian":
            return 3
        elif race=="Southeast Asian":
            return 4
        elif race=="Indian":
            return 5
        elif race=="Middle Eastern":
            return 6
    elif version=="resnet34_4":
        if race=="White":
            return 0
        elif race=="Black":
            return 1
        elif race=="Asian":
            return 2
        elif race=="Indian":
            return 3


@torch.no_grad()
def compute_losses(orig_dir, recon_dir, labels_path):
    labels_df = pd.read_csv(labels_path)
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(128, antialias=True)
    ]) 
    ds = PairedImages(orig_dir, recon_dir, transforms=transforms) # ds contains tuples (orig_img, reconstruction, base file name)
    dataloader = DataLoader(ds, batch_size=128) 
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    
    # preprocess labels_df since we test only on 1400 samples. We can delete the rest 
    labels_df_columns = labels_df.columns 
    rows = []
    for path in tqdm(ds.orig_image_list):
        base_name = path.stem 
        row = labels_df.loc[[labels_df[labels_df["file"]==f"val/{base_name}.jpg"].index.item()]]
        rows.append(row)
    labels_df = pd.concat(rows)
    labels_df.columns = labels_df_columns

    
    LOSS_FCTs = {"lpips": lpips.LPIPS().to(device), 
                 "ssim": SSIMLoss(1),
                 "race_cos": RaceDeltaLoss(loss="cosine"),
                 "race_0-1": RaceDeltaLoss(loss="0-1"),
                 "niqe16": NIQE(block_size=16), 
                 "blur": LaplaceBlurriness(),
                 }
    columns_df = ["file", "race", "race_id", "race_recon", "race_recon_id"]
    for loss_name in LOSS_FCTs.keys():
        columns_df.append(loss_name)
        
    losses_rows = []
    for batch in tqdm(dataloader):
        orig_img = batch[0].to(device)
        recon_img = batch[1].to(device)
        base_names = batch[2]
        races = [] 
        races_id = [] 
        files = [] 
        for base_name in base_names: 
            file = f"val/{base_name}.jpg"
            race = labels_df["race"][labels_df["file"]==file].item()
            race_id = race_to_code(race)
            races.append(race)
            races_id.append(race_id)
            files.append(file)

        losses_batch = {"file": files, "race": races, "race_id": race_id}
        
        for loss_name in LOSS_FCTs:
            loss = LOSS_FCTs[loss_name]
            
            if loss_name == "lpips":
                # Images are in [0, 1], therefore set normalize flag! 
                loss_eval = loss(orig_img, recon_img, normalize=True).squeeze().cpu().numpy()
            elif loss_name == "race_0-1":
                loss_eval, race_pred = loss(recon_img, races)
            elif loss_name in ["niqe32", "niqe16", "niqe8", "blur"]: 
                # the lower the better
                loss_eval = loss(recon_img).cpu().numpy()
            else:
                loss_eval = loss(orig_img, recon_img).cpu().numpy()

            losses_batch.update({loss_name: loss_eval})
        
        race_pred_id = []
        for prediction in race_pred:
            race_pred_id.append(race_to_code(prediction))  
        losses_batch.update({"race_recon": race_pred, "race_recon_id": race_pred_id})
        # Update losses dataframe: 
        losses_rows.append(losses_batch)
    
    losses_df = pd.concat([pd.DataFrame(row) for row in losses_rows], ignore_index=True) 
    return losses_df 
         
from src.data_utils import PairedDuplicateImages  
@torch.no_grad()
def compute_diversity(orig_dir, recon_dir, labels_path, num_duplicates=100):
    labels_df = pd.read_csv(labels_path)
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(128, antialias=True)
    ])
    if num_duplicates==1:
        ds = PairedImages(orig_dir, recon_dir, transforms=transforms)
    else: 
        ds = PairedDuplicateImages(orig_dir, num_duplicates=num_duplicates, recon_dir=recon_dir, transforms=transforms)
    dataloader = DataLoader(ds, batch_size=128)
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
   
    loss = RaceDeltaLoss(loss="0-1")
        
    losses_rows = []
    for batch in tqdm(dataloader):
        recon_img = batch[1].to(device)
        races = batch[2]
        races_id = [] 
        for race in races:
            race_id = race_to_code(race)
            races_id.append(race_id)
 
        
        loss_eval, race_pred = loss(recon_img, races)
        
        # TODO Check that! Should I return race_id or races_id?! 
        losses_batch = {"race": races, "race_id": race_id, "race_0-1": loss_eval}
        
        race_pred_id = []
        for prediction in race_pred:
            race_pred_id.append(race_to_code(prediction))  
        losses_batch.update({"race_recon": race_pred, "race_recon_id": race_pred_id})
        # Update losses dataframe: 
        losses_rows.append(losses_batch)
    
    losses_df = pd.concat([pd.DataFrame(row) for row in losses_rows], ignore_index=True) 
    return losses_df 
 
class PSNRLoss(nn.Module):
    def __init__(self, max_val: float) -> None:
        super().__init__()
        self.max_val: float = max_val
        
    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        mse = MSELoss(batch1, batch2)
        psnr = 10.0 * torch.log10(self.max_val**2 / mse)
        return -1 * psnr 
    
class SSIMLoss(nn.Module):
    """Based on https://github.com/VainF/pytorch-msssim"""
    def __init__(self, max_val: int) -> None:
        super().__init__()
        self.max_val: float = max_val
        self.ssim = SSIM(data_range=self.max_val, channel=3, size_average=False)
        
    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        ssim_score = self.ssim(batch1, batch2)
        dssim = (1 - ssim_score) / 2  
        return dssim 
    
def MSELoss(batch1, batch2):
    mse_loss_elementwise = nn.MSELoss(reduction="none")
    loss_elementwise = mse_loss_elementwise(batch1, batch2)
    loss_per_sample = torch.mean(loss_elementwise.view(loss_elementwise.size(0), -1), dim=1)
    return loss_per_sample 
