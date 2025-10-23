import torch
import torch.nn as nn
from Network.Pro_Extract.Attention.ms_deform_attn import MSDeformAttn
from Network.Pro_Extract.Attention.External_TF_module import MEAttention
import torch.nn.functional as F
import copy


class New_CALayer(nn.Module):   ##  Transformer Cross Attention
    def __init__(self, d_model=256, d_ffn=960,
                 dropout=0.1, activation="relu",
                 n_levels=1, n_heads=8, n_points=4):
        super().__init__()
        self.deform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.cross_attn = MEAttention(dim=d_model) 
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, x):
        tgt = self.linear1(x)
        tgt = self.activation(tgt)
        tgt = self.dropout3(tgt)
        tgt = self.linear2(tgt)
        tgt = self.dropout4(tgt)
        tgt = self.norm3(tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # deform attention
        tgt2 = self.deform_attn(self.with_pos_embed(tgt, pos),
                              reference_points,
                              tgt,
                              spatial_shapes,
                              level_start_index,
                              padding_mask)
        tgt3 = tgt + self.dropout1(tgt2)
        tgt4 = self.norm1(tgt3)
        tgt5 = self.cross_attn(tgt4, tgt)
        tgt6 = tgt4 + self.dropout1(tgt5)
        tgt7 = self.norm1(tgt6)

        return tgt7



class CA(nn.Module):
    def __init__(self, att_layer, n_layers):
        super().__init__()
        self.layers = _get_clones(att_layer, n_layers)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")