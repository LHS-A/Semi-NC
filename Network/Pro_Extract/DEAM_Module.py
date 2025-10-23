import torch
import torch.nn as nn
from torch.nn.init import normal_
from mmcv.cnn import ConvModule
from Network.Pro_Extract.Attention.ms_deform_attn import MSDeformAttn
from Network.Pro_Extract.Attention.position_embedding import build_position_encoding, NestedTensor
from Network.Pro_Extract.New_cross_attention import New_CALayer, CA
import torch.nn.functional as F

import numpy as np


def reform(loaded_weights):
    if loaded_weights.ndim == 3:
        reshap = torch.from_numpy(loaded_weights).view(768, 768).t()
    else:
        reshap = torch.from_numpy(loaded_weights).view(768)
    return nn.Parameter(reshap)


def change_totens(loaded_weights):
    return nn.Parameter(torch.from_numpy(loaded_weights))


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DEAM(nn.Module):

    def __init__(self,
                 d_model=240,  
                 nhead=8,
                 d_ffn=960,  
                 dropout=0.1, 
                 act="relu",
                 n_points=4,
                 n_levels=3,
                 n_sa_layers=1, 
                 in_channles=[64, 64, 128, 256, 512],
                 proj_idxs=(2, 3, 4),  
                 activation="relu"

                 ):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.n_levels = n_levels

        self.proj_idxs = proj_idxs
        self.projs = nn.ModuleList()
        for idx in self.proj_idxs:
            self.projs.append(ConvModule(in_channles[idx],
                                         d_model, 
                                         kernel_size=3,
                                         padding=1,
                                         conv_cfg=dict(type="Conv"),
                                         norm_cfg=dict(type='BN'),
                                         act_cfg=dict(type='ReLU')
                                         ))

        CAlayer = New_CALayer(d_model=d_model,
                              d_ffn=d_ffn,
                              dropout=dropout,
                              activation=act,
                              n_levels=n_levels,
                              n_heads=nhead,
                              n_points=n_points)

        self.ca = CA(att_layer=CAlayer,
                     n_layers=n_sa_layers)

       
        self.conv11_1 = nn.Conv2d(d_model, d_ffn, 1, stride=1, padding=0, bias=False) 
        self.norm3 = nn.LayerNorm(d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        ## DW
        self.conv_dw = nn.Conv2d(d_ffn, d_ffn, kernel_size=3, stride=1, padding=1, dilation=1, groups=d_ffn,
                                 bias=False)  
        self.norm4 = nn.LayerNorm(d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout4 = nn.Dropout(dropout)

        self.conv11_2 = nn.Conv2d(d_ffn, d_model, 1, stride=1, padding=0, bias=False)  
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout5 = nn.Dropout(dropout)
     
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        self.position_embedding = build_position_encoding(position_embedding="sine", hidden_dim=d_model)
        self._reset_parameters() 

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward_DW(self, tgt):  
        tgt2 = self.conv11_1(tgt).permute(0,2,3,1)
        tgt2 = self.dropout3(self.activation(self.norm3(tgt2))).permute(0,3,1,2)

        tgt2 = self.conv_dw(tgt2).permute(0,2,3,1)

        tgt2 = self.dropout4(self.activation(self.norm4(tgt2))).permute(0,3,1,2)
        tgt2 = self.conv11_2(tgt2)
        tgt = (tgt + self.dropout5(tgt2)).permute(0,2,3,1)
        tgt = self.norm5(tgt)
        return tgt

    def projection(self, feats):
        pos = []
        masks = []  
        cnn_feats = [] 
        tran_feats = [] 

        for idx, feats in enumerate(feats):
            if idx not in self.proj_idxs:
                cnn_feats.append(feats)
            else:
                n, c, h, w = feats.shape
                mask = torch.zeros((n, h, w)).to(torch.bool).to(feats.device) 
                nested_feats = NestedTensor(feats, mask)
                masks.append(mask)
                pos.append(self.position_embedding(nested_feats).to(nested_feats.tensors.dtype))
                tran_feats.append(feats)

        for idx, proj in enumerate(self.projs):
            tran_feats[idx] = proj(tran_feats[idx]) 

        return cnn_feats, tran_feats, pos, masks

    def forward(self, x):
        # project and prepare for the input
        cnn_feats, trans_feats, pos_embs, masks = self.projection(x)
        # dsa
        features_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        feature_shapes = []
        spatial_shapes = []
        for lvl, (feature, mask, pos_embed) in enumerate(zip(trans_feats, masks, pos_embs)):
            bs, c, h, w = feature.shape
            spatial_shapes.append((h, w))
            feature_shapes.append(feature.shape)

            feature = feature.flatten(2).transpose(1, 2)  ## feature:[bs,h*w,c]
            mask = mask.flatten(1)  ##  mask :[bs,h*w]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            features_flatten.append(feature)
            mask_flatten.append(mask)

        features_flatten = torch.cat(features_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # self att
        feats = self.ca(features_flatten,
                        spatial_shapes,
                        level_start_index,
                        valid_ratios,
                        lvl_pos_embed_flatten,
                        mask_flatten)

        # recover
        out = []
        features = feats.split(spatial_shapes.prod(1).tolist(), dim=1)
        for idx, (feats, ori_shape) in enumerate(
                zip(features, spatial_shapes)):  
            abc = feats.transpose(1, 2).reshape(feature_shapes[idx])
            out.append(abc)

        cnn_feats.extend(out)
        return cnn_feats

