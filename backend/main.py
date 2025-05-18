
import torch
from PIL import Image


from AIhandler import AIhandler
a = AIhandler()

img = Image.open("prueba.jpeg")
img = img.convert("RGB")
codebooks1 = a.GenerateImageCodebooks(image=img, tensor=False)
codebooks2 = a.GenerateSeqCodebooks("perro blanco")
difrencia = a.FindDifferenceBetweenCodebooks(codebooks1, codebooks2)
print("Diferencia entre codebooks: ", difrencia)
