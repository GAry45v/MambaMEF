import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_
from model.blocks import CBlock_ln, SwinTransformerBlock
from model.global_net import Global_pred

# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=1, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, in_dim, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, in_dim, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
            
            

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)

        return mul, add

class IAT(nn.Module):
    def __init__(self, in_dim=1, local_net_dim=16, global_embed_dim=64, global_num_heads=4, with_global=True, type='exp'): # type='exp' or 'lol' etc.
        super(IAT, self).__init__()
        
        # Local network for M and A maps
        self.local_net = Local_pred_S(in_dim=in_dim, dim=local_net_dim) # PEM channel dim is 16 by default in paper

        self.with_global = with_global
        if self.with_global:
            # Global network for gamma prediction
            self.global_net = Global_pred(in_channels=in_dim, embed_dim=global_embed_dim, num_heads=global_num_heads, type=type, num_gamma_queries=1)

    # The apply_color method is no longer needed for Option A with Y-channel only
    # def apply_color(self, image, ccm):
    #     shape = image.shape
    #     # This was for 3-channel image and 3x3 ccm
    #     # image = image.view(-1, 3) 
    #     # image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    #     # image = image.view(shape)
    #     return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        # img_low is (B, 1, H, W) for Y-channel
        
        # Local adjustments
        mul, add = self.local_net(img_low) # mul, add are (B, 1, H, W)
        
        # Apply local adjustments: Y' = Y_in * M + A
        # The paper's equation is I_t = (max( sum(W * (I_i * M + A)) , epsilon ) ) ^ gamma
        # For Y-channel and Option A, this becomes: Y_t = (max(Y_in * M + A, epsilon)) ^ gamma
        img_after_local = (img_low.mul(mul)).add(add)

        if not self.with_global:
            # If no global, maybe just return the locally adjusted image
            # The original returns mul, add, img_high (locally_adjusted)
            # We should clamp to ensure positivity if gamma is applied later, or if this is final output
            img_final = torch.clamp(img_after_local, min=1e-8) # Or a more sophisticated mapping
            return mul, add, img_final # Or just img_final
        
        else:
            # Global adjustment (gamma only for Option A)
            gamma = self.global_net(img_low) # gamma shape: (B) or (B,1)
            
            # Ensure gamma has shape (B, 1, 1, 1) for broadcasting
            gamma = gamma.view(-1, 1, 1, 1) 
            
            # Apply clamping (max(Y', epsilon)) before gamma correction
            img_to_correct = torch.clamp(img_after_local, min=1e-8) # Epsilon is 1e-8 in paper [cite: 1]
            
            # Apply gamma: (max(Y', epsilon)) ^ gamma
            img_final_corrected = img_to_correct ** gamma
            
            # Final clamp for stability if needed, though often handled by loss or subsequent steps
            img_final_corrected = torch.clamp(img_final_corrected, 0.0, 1.0) # Assuming output range 0-1

            return mul, add, img_final_corrected


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    img = torch.Tensor(1, 3, 400, 600)
    net = IAT()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)