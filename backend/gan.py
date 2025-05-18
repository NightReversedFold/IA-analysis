import torch.nn as nn
import torch
import numpy as np

class VQUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, codebook_size=512, encoder_channel_dims=[64, 128, 256, 512], commitment_cost=0.25):
        super(VQUNet, self).__init__()
        self.commitment_cost = commitment_cost
        self.encoder_channel_dims = encoder_channel_dims 
        self.enc_blocks = nn.ModuleList()
        current_channels = in_channels
        for i, channels in enumerate(encoder_channel_dims):
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

      
        self.dec_blocks = nn.ModuleList()

    
        reversed_enc_dims = encoder_channel_dims[::-1] 

        current_channels = reversed_enc_dims[0] 
        for i in range(len(reversed_enc_dims)):
            is_last_block = (i == len(reversed_enc_dims) - 1)
            
          
            up_in_channels = current_channels
            if i == 0: 
                 up_out_channels = reversed_enc_dims[i] 
            else: 
                 up_out_channels = reversed_enc_dims[i]


            upsample_layer = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=4, stride=2, padding=1)
            
            conv_in_channels = up_out_channels
            if not is_last_block:
                 skip_channel_dim = reversed_enc_dims[i+1]
                 conv_in_channels += skip_channel_dim
            
            conv_out_channels = up_out_channels
            
            if is_last_block: 
       
                upsample_layer = nn.ConvTranspose2d(up_in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                conv_layer = nn.Tanh() 
            else:
                conv_layer = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_in_channels, up_out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

            self.dec_blocks.append(nn.ModuleDict({
                'upsample': upsample_layer,
                'conv': conv_layer
            }))
            
            current_channels = up_out_channels

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
        return z_q, loss

    def forward(self, x):
        skip_connections_quantized = []
        total_vq_loss = 0.0
        current_features = x
        for i, enc_block in enumerate(self.enc_blocks):
            encoded_features = enc_block(current_features)
            quantized_features, vq_loss = self._quantize(encoded_features, i)
            skip_connections_quantized.append(quantized_features)
            total_vq_loss += vq_loss
            current_features = encoded_features

    
        decoder_input = skip_connections_quantized[-1]
        reversed_quantized_skips_for_concat = skip_connections_quantized[:-1][::-1]

        current_features = decoder_input 
        
        for i, dec_m_dict in enumerate(self.dec_blocks):
            upsample_layer = dec_m_dict['upsample']
            conv_block = dec_m_dict['conv']

            current_features = upsample_layer(current_features)

            if i < len(reversed_quantized_skips_for_concat): 
                skip_to_concat = reversed_quantized_skips_for_concat[i]
            
                if current_features.shape[2:] != skip_to_concat.shape[2:]:
                   
                    print("AYUDA JESUCRISTO DIOS MIO")
                current_features = torch.cat((current_features, skip_to_concat), dim=1)
            
            current_features = conv_block(current_features) 

        reconstructed_x = current_features
        return reconstructed_x, total_vq_loss, skip_connections_quantized