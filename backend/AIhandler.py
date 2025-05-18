import torch
import torch.nn.functional as F
from ai_models.vq_unet import pretrained_vqunet_encoder
from ai_models.seq2img import PretrainedModel
from utils.preproccess import utility

class AIhandler:
    def __init__(self, vq_unet_device: str="cuda", seq2img_device: str="cuda", vqunet_weights_path: str="pretrainedweights/finished_encoder.pth", seq2img_weights_path: str="pretrainedweights/finalseq.pth"):
        self.vq_unet_weights_path = vqunet_weights_path
        self.seq2img_weights_path = seq2img_weights_path
        self.vq_unet_device = torch.device(vq_unet_device if torch.cuda.is_available() else "cpu")
        self.seq2img_device = torch.device(seq2img_device if torch.cuda.is_available() else "cpu")
        if vq_unet_device != "cuda":
            self.vq_unet_device = vq_unet_device
        if seq2img_device != "cuda":
            self.seq2img_device = seq2img_device
        self.vq_unet = pretrained_vqunet_encoder()
        self.vq_unet.to(self.vq_unet_device)
        self.seq2img = PretrainedModel()
        self.seq2img.to(self.seq2img_device)
        self.utility = utility()

    def GenerateSeqCodebooks(self, text: str):
        skips = self.seq2img.forward([text])
        return skips
    def GenerateImageCodebooks(self, image, tensor: bool = False):
        if not tensor:
            image = self.utility.transform(image)
        image = self.utility.transform_image(image=image)
        with torch.no_grad():
            skips = self.vq_unet.forward(image)
        return skips

# DIFERENCIA EN L2, QUEDA PENDIENTE IMPLEMENTAR UN POSIBLE EXPONENTIAL MOVING AVERAGE.
    def FindDifferenceBetweenCodebooks(self, codebook1: list, codebook2: list, weight1: float =1, weight2: float = 1, weight3: float = 1, weight4: float = 1):
        d1,d2,d3,d4 = 0,0,0,0
        for i, x in enumerate(codebook1):
            with torch.no_grad():
                codebook1[i] = (F.mse_loss(x, codebook2[i]))
        d1,d2,d3,d4 = codebook1
        d1,d2,d3,d4 = d1*weight1, d2*weight2, d3*weight3, d4*weight4
        difference = (d1+d2+d3+d4).item()
        return difference