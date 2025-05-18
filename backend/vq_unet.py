import torch.nn as nn
import torch
import lightning as L

# IMPLEMENTACION HECHA A MANO DE UNA ARQUITECTURA UNET
# NO HABIA CODIGO PARA ESTA, POR LO TANTO SE HIZO A MANO, Y SE ENTRENO DE 0
# https://arxiv.org/abs/2406.03117
# NOTA; SE USA UNA UNET ENCODER Y SE GUARDAN LOS CODEBOOKS DE LAS SKIP CONNECTIONS.
class VQUNet(L.LightningModule):
    def __init__(self, in_channels=3, codebook_size=512, encoder_channel_dims=[64, 128, 256, 512], commitment_cost=0.25):
        super(VQUNet, self).__init__()
        self.commitment_cost = commitment_cost
        self.encoder_channel_dims = encoder_channel_dims 
        self.enc_blocks = nn.ModuleList()
        current_channels = in_channels
        for channels in encoder_channel_dims:
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = channels
        self.codebooks = nn.ModuleList()
        for emb_dim in encoder_channel_dims:
            self.codebooks.append(nn.Embedding(codebook_size, emb_dim))
    def _quantize(self, z_e, codebook_idx):
        codebook = self.codebooks[codebook_idx]
        b, c, h, w = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, c)
        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + torch.sum(codebook.weight**2, dim=1) - 2 * torch.matmul(z_e_flat, codebook.weight.t())
        indices = torch.argmin(distances, dim=1)
        z_q_flat = codebook(indices)
        z_q = z_q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        z_q = z_e + (z_q - z_e).detach()
        return z_q
    def forward(self, x):
        skip_connections_quantized = []
        current_features = x
        for i, enc_block in enumerate(self.enc_blocks):
            encoded_features = enc_block(current_features)
            quantized_features = self._quantize(encoded_features, i)
            skip_connections_quantized.append(quantized_features)
            current_features = encoded_features
        return skip_connections_quantized