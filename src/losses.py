import torch 
import scipy
import numpy as np 
from pyiqa.archs.func_util import diff_round
from pyiqa.utils.color_util import to_y_channel 
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.archs.niqe_arch import niqe 

from kornia.filters.laplacian import laplacian 
from kornia.color.gray import rgb_to_grayscale 
import torch.nn as nn 


class LaplaceBlurriness(nn.Module):
    """ 
    Returns a blurriness score based on https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ 
    Lower scores indicate less blurriness. 
    """
    def __init__(self, kernel_size=3, normalized=False, border_type="reflect"):
        super().__init__() 
        self.kernel_size = kernel_size
        self.normalized = normalized 
        self.border_type = border_type
        
    def forward(self, img): 
        img = rgb_to_grayscale(img)
        return -1 * laplacian(img, self.kernel_size, self.border_type, self.normalized).var(dim=(1,2,3))
    
"""Taken from https://iqa-pytorch.readthedocs.io/en/latest/_modules/pyiqa/archs/niqe_arch.html#niqe"""
def calculate_niqe(img: torch.Tensor,
                   block_size=96,
                   crop_border: int = 0,
                   test_y_channel: bool = True,
                   pretrained_model_path: str = "",
                   color_space: str = 'yiq',
                   **kwargs) -> torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (str): The pretrained model path.
    Returns:
        Tensor: NIQE result.
    """
    if pretrained_model_path == "":
        pretrained_model_path = load_file_from_url("https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/niqe_modelparameters.mat")
    params = scipy.io.loadmat(pretrained_model_path)
    mu_pris_param = np.ravel(params['mu_prisparam'])
    cov_pris_param = params['cov_prisparam']
    mu_pris_param = torch.from_numpy(mu_pris_param).to(img)
    cov_pris_param = torch.from_numpy(cov_pris_param).to(img)

    mu_pris_param = mu_pris_param.repeat(img.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(img.size(0), 1, 1)

    # NIQE only support gray image 
    if img.shape[1] == 3:
        img = to_y_channel(img, 255, color_space)
    elif img.shape[1] == 1:
        img = img * 255

    img = diff_round(img)
    img = img.to(torch.float64)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, block_size_h = block_size, block_size_w = block_size)

    return niqe_result


class NIQE(torch.nn.Module):
    """NIQE class with variable block size."""
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size 
        
    def forward(self, img):
        return calculate_niqe(img, block_size = self.block_size)