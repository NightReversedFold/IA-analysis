from seq2img import PretrainedModel
import torch

a = PretrainedModel()
a.to("mps")
d = a.forward(["coche"])
for x in d:
    print(x.shape)

from vq_unet import pretrained_vqunet_encoder
from utils.preproccess import utility

b = pretrained_vqunet_encoder()
import torch.nn.functional as F
h = utility()
def transformIMAGE(image):
    skips = b.forward(image)
    return skips
from PIL import Image
import torchvision.transforms as transforms
img = Image.open("prueba.jpeg")
img = img.convert("RGB")
img = transforms.ToTensor()(img)
c = transformIMAGE(img)
difference = 0
for i,x in enumerate(c):
  print(difference)
  difference += F.mse_loss(x, d[i])
  c[i] = 1-((d[i]-x)**2)
print((difference).item())