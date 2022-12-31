from functools import partial
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from models.models_crossvit import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed, positional_encoding_1d
import clip
from torchvision import transforms
import einops

class CLIPCount(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=768, encoder_depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_vpt:bool = True, n_vpt:int = 2, use_coop:bool=True, 
                 n_coop:int = 2):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.clip, clip_preprocess = clip.load("ViT-B/16")
        self.clip = self.clip.to('cuda')
        self.clip.requires_grad_(False)
        self.preprocess = transforms.Compose([transforms.Resize((224,224)),
                            transforms.Normalize(
                                mean = (0.48145466, 0.4578275, 0.40821073),
                                std= (0.26862954, 0.26130258, 0.27577711)
                                ) 
                            ])
        self.use_vpt = use_vpt
        self.use_coop = use_coop
        self.n_vpt = n_vpt if use_vpt else 0
        self.n_coop = n_coop if use_coop else 0
        self.img_encoder = CLIPViT(self.clip, use_vpt=self.use_vpt, n_vpt=self.n_vpt)
        self.text_encoder = CLIPTextTransformer(self.clip, learn_context=self.use_coop, n_ctx = self.n_coop)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_linear = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.proj_feat = False  #!HARDCODED Dec 27: 
        self.n_patches = 196
        n_token = self.n_patches
        self.rd_ln = nn.Identity() #layer norm for relation descriptor
        self.rd_pos_embed = None
        if self.proj_feat:
            n_token = n_token + 1 + self.n_vpt
            self.decoder_proj = nn.Conv1d(n_token, self.n_patches,1,1) #!HARDCODED Dec 26: for vit-32
        else:
            self.decoder_proj = nn.Identity()
            #layer norm for relation descriptor text embedding (1) + learned cls token (1) + learned vpt token (n_vpt)
            self.rd_ln = nn.LayerNorm(1+1) 
            self.rd_pos_embed = nn.Parameter(torch.zeros(1, 1+1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            rd_pos_embed = positional_encoding_1d( decoder_embed_dim,1+1)
            self.rd_pos_embed.data.copy_(rd_pos_embed.unsqueeze(0))
        #TODO Dec 29: refactor
        #init weights
        nn.init.xavier_normal_(self.decoder_linear.weight)

        #!HARDCODED Dec 26: for vit-32

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_token, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        #TODO Dec 26: sincos pos embed
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.n_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.shot_token = nn.Parameter(torch.zeros(512))

        # Exemplar encoder with CNN
       

        #! Dec 24: this is Feature Interaction Module (FIM)
        self.decoder_norm_pre = norm_layer(decoder_embed_dim)

        self.fim_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])


        self.decoder_norm = norm_layer(decoder_embed_dim)


        #upsampler
        self.decoder = Decoder(decoder_embed_dim, 384)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss



    

    def forward_encoder(self, x):
        """
        input: x: images, [N, 3, 384, 384]
        """
        # embed patches
        x = self.preprocess(x)
        img_patches, cls_token, x = self.img_encoder(x)
        return img_patches, cls_token, x

    def forward_decoder(self, patch_feature, text, cls_token, img_patches):
        """
        input:
            x: feature map of the patches of the image
            y_: the prompt text
        """
        # encode text

        # embed tokens
        x = self.decoder_linear(img_patches)
        # x_patches = x[:,1+self.n_vpt:,:] #image patches
        x_patches = x
        #TODO Dec 28: handle cls token for use vpt
        # x_tokens = x[:,:1+self.n_vpt,:] # [CLS] token + learned context token
        # x_cls = cls_token / cls_token.norm(dim=-1, keepdim=True)ß
        x_cls = cls_token 
        x_vpt = x[:,1:1+self.n_vpt,:] # learned context token
        # add pos embed
        if self.proj_feat:
            x = x + self.decoder_pos_embed
        else:
            x = x_patches + self.decoder_pos_embed

        #TODO Dec 27: refactor var name
        y_ = clip.tokenize(text).to(x.device)
        y_ = self.text_encoder(y_).float()
        #TODO Dec 28: add negative prompt for a stronger guidance ??
        # y_ = y_ / y_.norm(dim=-1, keepdim=True)

        if not self.proj_feat:
            y_ = torch.concat([y_, x_cls, torch.mul(y_, x_cls)], dim=1) # element-wise multiplication (ZegCLIP)
          
        # apply Transformer blocks (cross-attention)
        x = self.decoder_norm_pre(x)

        for blk in self.fim_blocks:
            x = blk(x, y_) #TODO Dec 28: check Q K V
        x = self.decoder_norm(x)
        
        #! Dec 26: deal with the dimension of the CLIP ViT
        x = self.decoder_proj(x)
        # Density map regression
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        x = self.decoder(x)
        return x

    def forward(self, imgs, text):
        #! Dec 24: ViT encoder is not trained in finetune stage
        # with torch.no_grad():
        #     latent = self.forward_encoder(imgs)
        img_patches, cls_token, patch_feat = self.forward_encoder(imgs)
        pred = self.forward_decoder(patch_feat, text, cls_token, img_patches)  # [N, 384, 384]
        return pred


class CLIPViT(nn.Module):
    """
    ViT encoder for CLIP
    """
    def __init__(self, clip_model, use_vpt:bool, n_vpt:int) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.vit = clip_model.visual
        self.visual_prompt = None
        self.use_vpt = use_vpt
        if use_vpt:
            clip_embed_dim = 768 #!HARDCODED Dec 27: 
            vpt = torch.empty((n_vpt, clip_embed_dim))
            nn.init.normal_(vpt, std=0.5)
            self.visual_prompt = nn.Parameter(vpt)


    
    def forward(self, image):
        x = self.vit.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        img_patches = x
        
        if self.use_vpt:
            vpts = einops.repeat(self.visual_prompt, 'n d -> b n d', b=x.shape[0])
            x = torch.cat([self.vit.class_embedding.to(x.dtype) + \
                            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                            vpts,
                            x], dim=1)  # shape = [*, grid ** 2 + 1 + n_vpt, width]
        else:
            x = torch.cat([self.vit.class_embedding.to(x.dtype) + \
                            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                            x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #! Dec 26: temp not use class embedding here
        # x = x + self.vit.positional_embedding.to(x.dtype)
        x = self.vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vit.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x_patches = x[:, 1:, :]  # optional learnable context token + image patches
        x_cls = x[:, :1, :]  # [CLS] token
        x_cls = self.vit.ln_post(x_cls)
        x_cls = x_cls @ self.vit.proj
        return img_patches, x_cls, x
    
class CLIPTextTransformer(nn.Module):
    """
    Transfromer encoder (text) for CLIP
    """
    def __init__(self, clip_model, learn_context:bool, n_ctx:int = 2) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.learnable_context = None
        self.learn_context = learn_context #global context for all classes
        if learn_context:
            self.n_ctx = n_ctx
            context_vectors = torch.empty(self.n_ctx, self.clip_model.ln_final.weight.shape[0])
            torch.nn.init.normal_(context_vectors, std=.02)
            self.learnable_context = nn.Parameter(context_vectors) # [n_ctx, 512]

    def forward(self, text):
        """
        Input:
            text: tokenized text, shape = [batch_size, n_ctx]
        """
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        if self.learn_context:
            sos_token = x[:, 0, :].unsqueeze(1)  # [batch_size, 1, d_model]
            suffix_tokens = x[:, 1:-self.n_ctx, :] # class tokens + [EOS] token
            ctx = einops.repeat(self.learnable_context, 'n d -> b n d', b=x.shape[0])
            x = torch.cat([sos_token, ctx, suffix_tokens], dim=1)
        

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

# class Decoder(nn.Module):
#     def __init__(self, in_dim:int, target_hw:int) -> None:
#         super().__init__()
#                 # Density map regresssion module
#         self.decode_head0 = nn.Sequential(
#             nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head1 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head3 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1)
#         )  

#     def forward(self, x):
#         #!HARDCODED Dec 26: directly upsample to 384x384
#         x = F.interpolate(
#                          self.decode_head0(x), size=x.shape[-1]*3, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                          self.decode_head1(x), size=x.shape[-1]*3, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                          self.decode_head2(x), size=x.shape[-1]*3, mode='bilinear', align_corners=False)
#         #!HARDCODED Dec 26: directly upsample to 384x384
#         x = F.interpolate(
#                          self.decode_head3(x), size=384, mode='bilinear', align_corners=False)
#         x = F.sigmoid(x)
#         x = einops.rearrange(x, 'n 1 h w -> n h w')
#         return x


class Decoder(nn.Module):
    def __init__(self, in_dim:int, target_hw:int) -> None:
        super().__init__()
        # Density map regresssion module
        self.n_levels = 4
        convs = []
        prev_dim = in_dim # number of feature channels
        
        for i in range(self.n_levels):
            decode_head = nn.Sequential(
                nn.Conv2d(prev_dim, 256, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 256),
                nn.LeakyReLU(inplace=True)
            )
            convs.append(decode_head)
            prev_dim = 256

        self.convs = nn.ModuleList(convs)
        
        self.final_conv = nn.Sequential(
            # nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )
        #initialize weights
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        n_skip = []
        for i in range(self.n_levels):
            if i in n_skip:
                x = self.convs[i](x) + x
            else:
                x = self.convs[i](x)
            # x = self.up_convs[i](x)
            if i < self.n_levels-1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x,size = (384, 384), mode = 'bilinear', align_corners = False)
        x = self.final_conv(x)

        x = F.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
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