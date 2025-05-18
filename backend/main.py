
import torch
from PIL import Image


from AIhandler import AIhandler
a = AIhandler()
import torch.nn.functional as F
img = Image.open("prueba.jpeg")
img = img.convert("RGB")
codebooks1 = a.GenerateImageCodebooks(image=img, tensor=False)
codebooks2 = a.GenerateSeqCodebooks("coche")
print(len(codebooks1))
difference = 0
for i,x in enumerate(codebooks1):
  difference += F.mse_loss(x, codebooks2[i])
  codebooks1[i] = 1-((codebooks2[i]-x)**2)
print((codebooks1))