import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models_crossvit import CrossAttentionBlock, ConvCrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed, positional_encoding_1d
import clip
from torchvision import transforms
import einops
import functools
import operator
class CLIPCount(nn.Module):
    def __init__(self, fim_depth:int=4, 
                 fim_num_heads:int=8,
                 mlp_ratio:float=4., 
                 norm_layer=nn.LayerNorm,
                 use_vpt:bool = True, 
                 vpt_width:int = 2, 
                 vpt_depth:int = 2,
                 use_coop:bool=True, 
                 coop_width:int = 2, 
                 backbone:str="b16",
                 use_fim:bool = True, 
                 use_mixed_fim:bool=False, 
                 unfreeze_vit:bool=False):
        """
        The CLIP-Count model   
        Param:
            fim_depth: the number of blocks for the patch-text interaction module, only useful for naive ViT.
            fim_num_heads: the number of heads for the patch-text interaction module.
            mlp_ratio: the ratio (mlp width)/(cross attn hidden dim) for the patch-text interaction module.
            norm_layer: the normalization layer for the patch-text interaction module.
            use_vpt: whether to use visual prompt tuning
            vpt_width: how much visual token used per layer,
            vpt_depth: how many layers used for visual prompt tuning (try allocate from the input layer first)
            use_coop: whether use coop for context learning.
            backbone: visual backbone of clip.
            use_fim: whether to use a naive transformer for patch-text interaction
            use_mixed_fim: whether to use a hierarchical transformer for patch-text interaction
            unfreeze_vit: whether to fintune all clip vit parameters.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if backbone == "b16":
            self.clip, clip_preprocess = clip.load("ViT-B/16")
            self.n_patches = 14*14
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512
        elif backbone == "b32":
            self.clip, clip_preprocess = clip.load("ViT-B/32")
            self.n_patches = 7*7
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512

        elif backbone == "l14":
            self.clip, clip_preprocess = clip.load("ViT-L/14")
            self.n_patches = 16*16
            self.clip_hidden_dim = 1024
            self.clip_out_dim = 768



        self.clip = self.clip.to('cuda')
        if unfreeze_vit:
            # deal with some strange behavior of CLIP and pytorch-lightning.
            self.clip = self.clip.float() 
        self.clip.requires_grad_(False)
        self.preprocess = transforms.Compose([transforms.Resize((224,224)),
                            transforms.Normalize(
                                mean = (0.48145466, 0.4578275, 0.40821073),
                                std= (0.26862954, 0.26130258, 0.27577711)
                                ) 
                            ])

        self.use_vpt = use_vpt
        self.use_coop = use_coop
        self.vpt_width = vpt_width if use_vpt else 0
        self.vpt_depth = vpt_depth if use_vpt else 0
        self.coop_width = coop_width if use_coop else 0
        self.img_encoder = CLIPViT(self.clip, self.clip_hidden_dim, use_vpt=self.use_vpt, vpt_width=self.vpt_width,vpt_depth = self.vpt_depth,unfreeze=unfreeze_vit)
        self.text_encoder = CLIPTextTransformer(self.clip, use_coop=self.use_coop, n_ctx = self.coop_width)

        # --------------------------------------------------------------------------
        # Contrastive Learning related
        self.patch_feat_proj = nn.Linear(self.clip_hidden_dim, self.clip_out_dim, bias=True)
        self.patch_feat_proj_contrast = nn.Linear(self.clip_hidden_dim, self.clip_out_dim, bias=True)
        nn.init.xavier_normal_(self.patch_feat_proj.weight)

        n_token = self.n_patches
        # the PE for the patch embeddings \mathcal{E}_p
        self.patch_emb_pos_embed = nn.Parameter(torch.zeros(1, n_token, self.clip_out_dim), requires_grad=False)  # fixed sin-cos embedding
        decoder_pos_embed = positional_encoding_1d(self.clip_out_dim, n_token)
        self.patch_emb_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))

        # --------------------------------------------------------------------------
        # The Hierarchical patch-text interaction module

        self.decoder_ln_pre = norm_layer(self.clip_out_dim)

        self.use_fim = use_fim
        self.use_mixed_fim = use_mixed_fim
        # cannot use mixed_fim and fim at the same time
        assert (not use_fim) or (not use_mixed_fim), "You can not use hierachical transformer and plain transformer at the same time!"
        self.fim_blocks = None
        if use_mixed_fim:
            self.fim_blocks = nn.ModuleList([
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution= 1.),
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution= 2.),
                ])

        elif use_fim:
            self.fim_blocks = nn.ModuleList([
                CrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
                for _ in range(fim_depth)])


        self.decoder_norm = norm_layer(self.clip_out_dim)


        # --------------------------------------------------------------------------
        # CNN-based density decoder
        self.density_decoder = DensityDecoder(self.clip_out_dim, 384, use_hiearachy = use_mixed_fim)
        # --------------------------------------------------------------------------
    

    def forward_visual_encoder(self, x, text_embedding):
        """
        input: x: images, [B, 3, 384, 384]
        text_embedding: [B, 1, 512]
        """
        # embed patches
        x = self.preprocess(x)
        _, cls_token, x = self.img_encoder(x, text_embedding)
        return cls_token, x

    def forward_decoder(self, img_feat_patches, text_embedding, cls_token):
        """

        """

        extra_out = {}
        
        x_cls = cls_token 
        extra_out['x_cls'] = x_cls
        extra_out['text_embedding'] = text_embedding
        # add pos embed

        patch_feat = img_feat_patches[:,1:,:]
        patch_embedding = self.patch_feat_proj(patch_feat)
        extra_out['patch_embedding'] = patch_embedding
        patch_embedding_contrast = self.patch_feat_proj_contrast(patch_feat)
        extra_out['patch_embedding_contrast'] = patch_embedding_contrast
        x = patch_embedding
        x = x + self.patch_emb_pos_embed # [B, 196, 512]

        y_ = text_embedding # [B, 1, 512]


        # apply Transformer blocks (cross-attention)
        if self.use_mixed_fim: 
            xs = []
            for blk in self.fim_blocks:
                x = blk(x, y_)
                xs.append(self.seq_2_2d(x))
        elif self.use_fim:
            for blk in self.fim_blocks:
                x = blk(x, y_)
        else: #add
            x = x + y_
        x = self.decoder_norm(x)
        
        # Density map regression
        x = self.seq_2_2d(x)
        extra_out['pixel_text_matching_map'] = x
        if self.use_mixed_fim:
            pred_density = self.density_decoder.forward_hierarchical(xs)
        else:
            pred_density = self.density_decoder(x)

        return pred_density, extra_out

    def forward(self, imgs, text, return_extra:bool = False, coop_require_grad:bool = False):

        text_token = clip.tokenize(text).to(imgs.device)

        if coop_require_grad:
            text_embedding = self.text_encoder(text_token).float()
        else:
            with torch.no_grad():
                text_embedding = self.text_encoder(text_token).float()

        cls_token, img_feat_patches = self.forward_visual_encoder(imgs, text_embedding)
        pred_density, extra_out = self.forward_decoder(img_feat_patches, text_embedding, cls_token)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out
        return pred_density
    
    def seq_2_2d(self,x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w) 
        return x

class CLIPViT(nn.Module):
    """
    ViT encoder for CLIP
    """
    def __init__(self, 
                 clip_model, 
                 clip_embed_dim:int, 
                 use_vpt:bool, 
                 vpt_width:int, 
                 vpt_depth:int = 8, 
                 unfreeze:bool=False) -> None:
        """
        Param:
            clip_model: pretrained OpenAI CLIP model
            use_vpt: whether to use visual prompt tuning
            vpt_width: number of vpt token per layer
            vpt_depth: number of vpt layers. 1: vpt at the first layer (shallow), >1: deep vpt
            unfreeze: If true, unfreeze the CLIP model
        """
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.vit = clip_model.visual
        if unfreeze:
            for param in self.vit.parameters():
                param.requires_grad = True
        self.use_vpt = use_vpt
        self.visual_prompt = None
        self.vpt_dropout = None
        self.vpt_norm = None
        self.vpt_proj = None
        self.vpt_depth = vpt_depth
        self.vpt_width = vpt_width
        self.visual_prompt = None
        if use_vpt:
            self.vpt_dropout = nn.Dropout(0.1)
            self.vpt_norm = nn.LayerNorm(clip_embed_dim, eps=1e-6)
            self.vpt_proj = nn.Linear(clip_embed_dim, clip_embed_dim)
            nn.init.kaiming_normal_(self.vpt_proj.weight, a=0, mode='fan_out') 

            patch_size = self.vit.conv1.kernel_size
            val = math.sqrt(6. / float(3 * functools.reduce(operator.mul, patch_size, 1) + self.clip_embed_dim))  
            vpt = torch.empty((vpt_depth, vpt_width, clip_embed_dim))
            nn.init.uniform_(vpt, -val, val)
            self.visual_prompt = nn.Parameter(vpt)
            




    
    def forward(self, image, text_embedding):
        """
        input: image: [B, 3, 224, 224]
        text_embedding: [B, 1, 512]
        """
        x = self.vit.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        img_patches = x
        

        x = torch.cat([self.vit.class_embedding.to(x.dtype) + \
                        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                        x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vit.positional_embedding.to(x.dtype)

        if self.use_vpt:
            vpts = einops.repeat(self.visual_prompt[0,...], 'n d -> b n d', b=x.shape[0])
            x = torch.cat([x[:, :1, :],
                            self.vpt_dropout(self.vpt_proj(vpts)),
                            x[:,1:,:]], dim=1)  # shape = [*, grid ** 2 + 1 + n_vpt, width]

        x = self.vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if (not self.use_vpt) or self.vpt_depth == 1 :
            x = self.vit.transformer(x)


        if self.use_vpt and self.vpt_depth > 1:
            x = self.deep_vpt_forward(x,text_embedding)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_cls = x[:, :1, :]  # [CLS] token
        x_cls = self.vit.ln_post(x_cls)
        x_cls = x_cls @ self.vit.proj
        return img_patches, x_cls, x
    

    def deep_vpt_forward(self, embedding_output, text_embdding = None, out_last = False):
        B = embedding_output.shape[1]
        transformer = self.vit.transformer
        assert self.vpt_depth < transformer.layers , "vpt_depth should be smaller than the number of layers in the transformer"
        for i in range(transformer.layers):
            if i == 0:
                hidden_states = transformer.resblocks[i](embedding_output)
            elif i < self.vpt_depth:
                deep_prompt_emb = self.vpt_dropout(self.vpt_proj(self.visual_prompt[i-1,...]).expand(B, -1, -1)).permute(1, 0, 2)
                # B, L, 768

                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.vpt_width):, :, :]
                ), dim=0)

                hidden_states = transformer.resblocks[i](hidden_states)
            elif i == self.vpt_depth:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[(1+self.vpt_width):, :, :]
                ), dim=0)
                hidden_states = transformer.resblocks[i](hidden_states)
            else:
                hidden_states = transformer.resblocks[i](hidden_states)
            
            if i == (transformer.layers-1): #11
                before_last_feats = self.vpt_norm(hidden_states)

        encoded = self.vpt_norm(hidden_states)
        if out_last:
            return before_last_feats, encoded
        else:
            return encoded
    
class CLIPTextTransformer(nn.Module):
    """
    Transfromer encoder (text) for CLIP
    """
    def __init__(self, clip_model, use_coop:bool, n_ctx:int = 2) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.learnable_context = None
        self.use_coop = use_coop #global context for all classes
        if use_coop:
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
        if self.use_coop:
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


class DensityDecoder(nn.Module):
    def __init__(self, in_dim:int, target_hw:int, use_hiearachy:bool = False) -> None:
        super().__init__()
        # Density map regresssion module
        self.n_levels = 4 if use_hiearachy else 2
        self.target_hw = [target_hw, target_hw]
        convs = []
        crt_dim = in_dim # number of feature channels
        for i in range(self.n_levels):
            decode_head = nn.Sequential(
                nn.Conv2d(crt_dim, crt_dim//2, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, crt_dim//2),
                nn.GELU()
            )
            convs.append(decode_head)
            crt_dim = crt_dim//2

        self.convs = nn.ModuleList(convs)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(crt_dim, 1, kernel_size=1, stride=1)
        )
        #initialize weights
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
        self.pyradim_conv = None # the conv to squeeze the fine multimodel features
        if use_hiearachy:
            self.pyradim_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.GroupNorm(8, 256),
                nn.GELU()
            )

    def forward(self, x):

        for i in range(self.n_levels):

            x = self.convs[i](x)
            # x = self.up_convs[i](x)
            if i < self.n_levels-1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x, size = self.target_hw, mode = 'bilinear', align_corners = False)
        x = self.final_conv(x)
        
        x = F.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

    def forward_hierarchical(self, xs):
        """
        xs: [14,14,512], [28,28,512]
        """
        x0, x1= xs[0], xs[1]
        x = x0
        for i in range(self.n_levels):
            if i == 1:
                x = x + self.pyradim_conv(x1)

            x = self.convs[i](x)
            if i < self.n_levels-1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x, size = (384, 384), mode = 'bilinear', align_corners = False)
        x = self.final_conv(x)
        
        x = F.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x





if __name__ == "__main__":
    clip_count = CLIPCount()


    