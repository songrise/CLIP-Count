from functools import partial
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from models.models_crossvit import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed
import clip
from torchvision import transforms
import einops

class CLIPCount(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=768, encoder_depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.clip, clip_preprocess = clip.load("ViT-B/32")
        self.clip = self.clip.to('cuda')
        self.clip.requires_grad_(False)
        self.preprocess = transforms.Compose([transforms.Resize((224,224)),
                            transforms.Normalize(
                                mean = (0.48145466, 0.4578275, 0.40821073),
                                std= (0.26862954, 0.26130258, 0.27577711)
                                ) 
                            ])
        
        self.img_encoder = CLIPViT(self.clip)
        self.text_encoder = CLIPTextTransformer(self.clip)
        # self.text_encoder = 
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_proj = nn.Conv1d(50,49,1,1) #!HARDCODED Dec 26: for vit-32

        #!HARDCODED Dec 26: for vit-32
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 50, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        #TODO Dec 26: sincos pos embed
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(50**.5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.shot_token = nn.Parameter(torch.zeros(512))

        # Exemplar encoder with CNN
        exemplar_enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #[3,64,64]->[64,32,32]
        )
        exemplar_enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #[64,32,32]->[128,16,16]
        )
        exemplar_enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # [128,16,16]->[256,8,8]
        )
        exemplar_enc4 = nn.Sequential(
            nn.Conv2d(256, decoder_embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
            # [256,8,8]->[512,1,1]
        )

        self.encoder_exemplar = nn.ModuleList([exemplar_enc1, exemplar_enc2, exemplar_enc3, exemplar_enc4])

        #! Dec 24: this is Feature Interaction Module (FIM)
        self.fim_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Density map regresssion module
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  
    
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.shot_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """
        input: x: images, [N, 3, 384, 384]
        """
        # embed patches
        x = self.preprocess(x)
        x = self.img_encoder(x)
        return x

    def forward_decoder(self, x, y_, shot_num=3):
        """
        input:
            x: latent code of query image 
            y_: the exemplar images
        """
        # test text
        y_ = clip.tokenize(y_).to(x.device)
        y_ = self.text_encoder(y_).float()

        # embed tokens
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed



        # Exemplar encoder
      
        
        # apply Transformer blocks
        for blk in self.fim_blocks:
            x = blk(x, y_)
        x = self.decoder_norm(x)
        x = self.decoder_proj(x)
        #! Dec 26: deal with the dimension of the CLIP ViT
        # Density map regression
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)

        x = F.interpolate(
                        self.decode_head0(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head1(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head2(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        # x = F.interpolate(
        #                 self.decode_head3(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        #!HARDCODED Dec 26: directly upsample to 384x384
        x = F.interpolate(
                         self.decode_head3(x), size=384, mode='bilinear', align_corners=False)

        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

    def forward(self, imgs, text, shot_num = 0):
        #! Dec 24: ViT encoder is not trained in finetune stage
        # with torch.no_grad():
        #     latent = self.forward_encoder(imgs)
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, text, shot_num)  # [N, 384, 384]
        return pred


class CLIPViT(nn.Module):
    """
    ViT encoder for CLIP
    """
    def __init__(self, clip_model) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.vit = clip_model.visual
    
    def forward(self, image):
        x = self.vit.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #! Dec 26: temp not use class embedding here
        # x = x + self.vit.positional_embedding.to(x.dtype)
        x = self.vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vit.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x
    
class CLIPTextTransformer(nn.Module):
    """
    Transfromer encoder (text) for CLIP
    """
    def __init__(self, clip_model) -> None:
        super().__init__()
        self.clip_model = clip_model

    def forward(self, text):
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.visual.conv1.weight.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        x = x.unsqueeze(1)  # [batch_size, 1, transformer.width]
        return x


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = CLIPCount(
        patch_size=16, embed_dim=768, encoder_depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = CLIPCount(
        patch_size=16, embed_dim=1024, encoder_depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = CLIPCount(
        patch_size=14, embed_dim=1280, encoder_depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_fim4(**kwargs):
    model = CLIPCount(
        patch_size=16, embed_dim=768, encoder_depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_fim6(**kwargs):
    model = CLIPCount(
        patch_size=16, embed_dim=768, encoder_depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  
mae_vit_base4_patch16 = mae_vit_base_patch16_fim4 # decoder: 4 blocks
mae_vit_base6_patch16 = mae_vit_base_patch16_fim6 # decoder: 6 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  

if __name__ == "__main__":
    clip_count = CLIPCount()