import torch
import PIL
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class utility():
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor()])
    def transform_image(self, image: torch.tensor, device: str = "cpu", batch: bool = False):
        if not batch:
            image = image.squeeze(0)
            image = image.squeeze(0)
            image = image.squeeze(0) # PARA ESTAR SEGUROS QUE NO QUEDAN DIMENSIONES RARAS
            image = image.unsqueeze(0)
        image = image.float()
        image = image.to(device)
        image = F.interpolate(image, size=(128, 128), mode='bilinear')
        return image