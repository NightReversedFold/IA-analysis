import lightning as L
import torch.nn.functional as F

import torch
import torchvision
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

import PIL

def transformIMAGE(image):

    image = image.float()
    
    image = image.unsqueeze(0)
    image = image.to(device)
    image = F.interpolate(image, size=config.IMAGE_SIZE, mode='bilinear')
    with torch.no_grad():
        reconstructions, vq_loss, skips, encoded_features = model(image)

    return skips, encoded_features

class seq2seq(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.embedding_dim = 512
        self.sentence_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        self.sentence_model.eval()
        self.commitment_cost = 0.25
        encoder_channel_dims = [512, 256, 128, 64]

        for param in self.sentence_model.parameters():
            param.requires_grad = False
        self.fc_layers = nn.Sequential(
               nn.Linear(512, 4096, bias=False),
               nn.ReLU(),
               nn.Linear(4096, 512 * 4 * 4)
           )
        self.upblocks = nn.ModuleList()
        self.loss = 0

        current_channels = encoder_channel_dims[0]
        for i in range(len(encoder_channel_dims)):
            is_last_block = (i == len(encoder_channel_dims) - 1)


            up_in_channels = current_channels
            if i == 0:
                 up_out_channels = encoder_channel_dims[i]
            else:
                 up_out_channels = encoder_channel_dims[i]


            upsample_layer = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=4, stride=2, padding=1)

            conv_in_channels = up_out_channels


            conv_out_channels = up_out_channels

            if is_last_block:
                upsample_layer = nn.ConvTranspose2d(up_in_channels, encoder_channel_dims[i], kernel_size=4, stride=2, padding=1)
                conv_layer = nn.Tanh()
            else:
                conv_layer = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_in_channels, up_out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            self.upblocks.append(nn.ModuleDict({
                'upsample': upsample_layer,
                'conv': conv_layer
            }))

            current_channels = up_out_channels
        self.codebooks = nn.ModuleList()

        for emb_dim in encoder_channel_dims:
            self.codebooks.append(nn.Embedding(512, emb_dim))


    def _quantize(self, z_e, codebook_idx):
        codebook = self.codebooks[codebook_idx]
        b, c, h, w = z_e.shape

        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, c)

        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + \
                    torch.sum(codebook.weight**2, dim=1) - \
                    2 * torch.matmul(z_e_flat, codebook.weight.t())

        indices = torch.argmin(distances, dim=1)
        z_q_flat = codebook(indices)
        z_q = z_q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        loss = self.commitment_cost * torch.mean((z_q.detach() - z_e)**2) + \
               torch.mean((z_q - z_e.detach())**2)

        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss # esto ni yo lo entiendo, se lo robe a un chino, pero entiendo lo que hace


    def forward(self, sentences_batch):
        x = self.sentence_model.encode(sentences_batch, convert_to_tensor=True)
        x = self.fc_layers(x)
        x = x.reshape(1, 512, 4, 4)
        skip_connections_quantized = []
        total_vq_loss = 0

        for i, val in enumerate(self.upblocks):
            upsample_layer = val['upsample']
            conv_block = val['conv']
            x = upsample_layer(x)
            x = conv_block(x)
            quantized_features, vq_loss = self._quantize(x, i)
            skip_connections_quantized.append(quantized_features)
            total_vq_loss += vq_loss

        return skip_connections_quantized, total_vq_loss




    def training_step(self, batch, batch_idx):
      try:
        if (batch_idx + 1) % 1500 == 0:
          torch.save(mod.state_dict(), f"drive/MyDrive/checkpoint_{batch_idx}.pth")
        input_sentences = batch["caption"]
        image = batch["image"]
        image = image.squeeze(0)

        skips, encoded_features = transformIMAGE(image)
        print(encoded_features.shape)
        for x in skips:
          x = x.squeeze(0)

        genskips, vq_loss = self.forward(input_sentences)
        loss = vq_loss
        for gen, target in zip(genskips, skips[::-1]):
          loss += F.mse_loss(gen, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        self.loss = loss
      except Exception as e:
        self.loss = None
        print(e)

      finally:
        return self.loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters() ,lr=1e-4)
        return optimizer
