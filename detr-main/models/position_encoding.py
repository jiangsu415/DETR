# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor

# 正余弦位置编码
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # 打印一下x
        x = tensor_list.tensors
        # 输入的特征图分别为Batch Channel W H
        # True表示实际的特征，False表示Padding
        # 对于X来说有些位置是加上padding
        # mask标注一下每一个位置是实际的特征还是padding出来的
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # 行方向的累加
        print(y_embed.shape)
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # 列方向的累加,最后可能是相同的一些值，因为前面mask为False是Padding出来的不需要累加，20表示最后一个位置
        # 在cumsum方法中最后一个值，行方向和列方向一样，表示最大的一个值，做归一化的时候去最大值就可以直接取最后一个值
        print(x_embed.shape)
        if self.normalize:
            eps = 1e-6
            # 行和列转换为一个归一化的结果，最后乘一个2π转换为一个正余弦的格式映射到一个角度中
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

# 指定维度，默认做一个128维的向量
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        print(dim_t.shape)
        # 计算一下是奇数维度还是偶数维度，因为两个维度的三角函数是不一样的,temperature是一个经验值
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

#计算三角函数公式，分别拿到行和列的位置embeding后的结果
        pos_x = x_embed[:, :, :, None] / dim_t
        print(pos_x.shape)
        pos_y = y_embed[:, :, :, None] / dim_t
        print(pos_y.shape)
        # (做一个三角函数)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        #做一个Concat操作，最终的到一个256维的结果，前128是一个Embeding_x后一个是Embeding_y
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# 可学习的位置编码
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
