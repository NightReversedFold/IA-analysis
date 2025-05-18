import PIL.Image
import torch
from ai_models.vq_unet import pretrained_vqunet_encoder
from ai_models.seq2img import PretrainedModel
from utils.preproccess import utility
import PIL
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
        skips = self.vq_unet.forward(image)
        return skips