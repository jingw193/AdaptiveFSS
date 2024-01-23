# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math, copy


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class Prototype_Adaptive_Module(nn.Module):
      """ 
      Protoype adaptive module for query-support feature enhancement.
      produce enhanced features for Few-shot segmentation task.
      Args:
            dim (int): Number of input channels.
            input_resolution (tuple): input_resolution for high and weight
            hidden_ratio(float): the compressed ratio of feature dim
            class_num(int): Number of adaptive class
            momentum(float): the momentum updating ratio for prototype
      """
      def __init__(self, 
                   dim, 
                   input_resolution, 
                   hidden_ratio=16., 
                   drop=0.,
                   class_num=1.,
                   momentum=0.9):
            super().__init__()
            self.dim = dim
            self.dim_low = dim // hidden_ratio
            self.input_resolution = input_resolution
            self.act = nn.ReLU()
            self.linears_down = nn.Linear(dim, self.dim_low)
            self.linears_up = nn.Linear(self.dim_low, dim)

            self.register_buffer("prototype", torch.randn(dim, class_num).requires_grad_(False))
            self.prototype = nn.functional.normalize(self.prototype, dim=0)

            self.drop = drop
            self.proj_drop = nn.Dropout(drop)
            self.momentum = momentum
            self.act_enhance = nn.ReLU6()

      def forward(self, x, s_f, s_y, class_idx=None):
            """ 
            Args:
            x: torch.Tensor
                [B , h * w, dim], query 
            s_f: torch.Tensor
                [B * S , h * w, dim], support 
            s_y: torch.Tensor
                [B, S, H, W],    support mask one-shot
            class_id: list
                len(class_id) = B
            Outputs:
            registered_feas_q: torch.Tensor
                  [B , h * w, dim], injected query features
            registered_feas_sup: torch.Tensor
                  [B * S, h * w, dim], injected support features
            """
            B, N, D = x.shape
            s_f = s_f.reshape(B, -1, N, D)
            S = s_f.size(1)
            s_y = F.interpolate(s_y, (self.input_resolution[0], self.input_resolution[1]), mode='nearest')
            sup_mask_fg = (s_y == 1).float().reshape(B, S, -1).reshape(B, -1).unsqueeze(-1) #[B, S * N, 1]
            semantic_prototype, sign_fore_per_batch = self.extract_semantic_prototype(s_f, sup_mask_fg)

            if class_idx is not None:
                with torch.no_grad():
                    self.train()
                    new_semantic_prototype = self.updata_prototype_bank(semantic_prototype, class_idx, sign_fore_per_batch)
            else:
                self.eval()
                new_semantic_prototype = self.select_prototype_bank(semantic_prototype, self.prototype)

            enhanced_feat_q = self.enhanced_feature(x.unsqueeze(1), new_semantic_prototype, sign_fore_per_batch)
            enhanced_feat_sup = self.enhanced_feature(s_f, new_semantic_prototype, sign_fore_per_batch)

            registered_feas_q = self.proj_drop(self.linears_up(self.act(self.linears_down(enhanced_feat_q.squeeze(1)))))
            registered_feas_sup = self.proj_drop(self.linears_up(self.act(self.linears_down(enhanced_feat_sup.reshape(-1, N, D)))))

            return registered_feas_q, registered_feas_sup


      def extract_semantic_prototype(self, s_f, s_y):
          """
          extract temporary class prototype according to support features and masks
          input:
            s_f: torch.Tensor
                [B, S, N, D], support features
            s_y: torch.Tensor
                [B, S * N, 1], support masks
          output:
            semantic_prototype: torch.Tensor
                [B, D], temporary prototypes
            sign_fore_per_batch: torch.Tensor
                [B], the signal of whether including foreground region in this image
          """
          B, S, N, D = s_f.shape
          num_fore_per_batch = torch.count_nonzero(s_y.reshape(B, -1), dim=1)
          s_y = s_y.repeat(1, 1, D)
          semantic_prototype = s_y * s_f.reshape(B, -1, D)
          semantic_prototype = semantic_prototype.mean(1) * (N * S) / (num_fore_per_batch.unsqueeze(1) + 1e-4)
          one = torch.ones_like(num_fore_per_batch).cuda()
          sign_fore_per_batch =  torch.where(num_fore_per_batch > 0.5, one, num_fore_per_batch)
          return semantic_prototype, sign_fore_per_batch
      
      def updata_prototype_bank(self, semantic_prototype, class_idx, sign_fore_per_batch):
          """
          updata prototype in class prototype bank during traning
          input:
            semantic_prototype: torch.Tensor
                [B, D]
            class_id: list
                len(class_id) = B
            sign_fore_per_batch: torch.Tensor
                [B], the signal of whether including foreground region in this image
          output:
            new_semantic_prototype: torch.Tensor
                [B, D], the updated prototypes for feature enhancement
          """   
          B, D = semantic_prototype.shape
          
          self.prototype = nn.functional.normalize(self.prototype, dim=0)
          semantic_prototype = nn.functional.normalize(semantic_prototype, dim=1)
          new_semantic_prototype_list = []
          for i in range(B):
               semantic_prototype_per = semantic_prototype[i,: ]
               class_idx_per = class_idx[i]
               if sign_fore_per_batch[i] == 1:
                    new_semantic_prototype_per = self.prototype[:, class_idx_per] * self.momentum + (1 - self.momentum) * semantic_prototype_per
                    self.prototype[:, class_idx_per] = new_semantic_prototype_per
               else: 
                    new_semantic_prototype_per = self.prototype[:, class_idx_per]    
               new_semantic_prototype_list.append(new_semantic_prototype_per)
          new_semantic_prototype = torch.stack(new_semantic_prototype_list, dim=0)
          return new_semantic_prototype

      def select_prototype_bank(self, semantic_prototype, prototype_bank):
          """
          select prototypes in class prototype bank during testing
          input:
            semantic_prototype: torch.Tensor
                shape = [B, D]
            prototype_bank: torch.Tensor
                shape = [D, class_num]
          output:
            new_semantic_prototype: torch.Tensor
                [B, D], the prototypes for feature enhancement
          """ 
          B, D = semantic_prototype.shape
          prototype_bank = nn.functional.normalize(prototype_bank, dim=0)
          semantic_prototype = nn.functional.normalize(semantic_prototype, dim=1)
          similar_matrix = semantic_prototype @ prototype_bank  # [B, class_num]
          idx = similar_matrix.argmax(1)
          new_semantic_prototype_list = []
          for i in range(B):
              new_semantic_prototype_per = prototype_bank[:, idx[i]]
              new_semantic_prototype_list.append(new_semantic_prototype_per)
          new_semantic_prototype = torch.stack(new_semantic_prototype_list, dim=0)
          return new_semantic_prototype

      def enhanced_feature(self, feature, new_semantic_prototype, sign_fore_per_batch):
          """ 
          Input:
            feature: torch.Tensor
                [B, S, N, D]
            new_semantic_prototype: torch.Tensor
                [B, D]
          Outputs:
            enhanced_feature: torch.Tensor
                [B, S, N, D]
          """
          B, D = new_semantic_prototype.shape
          feature_sim = nn.functional.normalize(feature, p=2, dim=-1)
          new_semantic_prototype = nn.functional.normalize(new_semantic_prototype, p=2, dim=1)
          similarity_matrix_list = []
          for i in range(B):
              feature_sim_per = feature_sim[i,:, :, :]
              new_semantic_prototype_er = new_semantic_prototype[i, :]
              similarity_matrix_per = feature_sim_per @ new_semantic_prototype_er
              similarity_matrix_list.append(similarity_matrix_per)
          similarity_matrix = torch.stack(similarity_matrix_list, dim=0)

          similarity_matrix = (similarity_matrix * self.dim ** 0.5) * sign_fore_per_batch.unsqueeze(-1).unsqueeze(-1)

          enhanced_feature = self.act_enhance(similarity_matrix).unsqueeze(-1).repeat(1, 1, 1, D) * feature + feature

          return enhanced_feature

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, args, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., adapter_weight=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, whether_adapter=False
                 ):

        super().__init__()
        self.adapter_weight = adapter_weight
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.fold = args.fold
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.whether_adapter = whether_adapter

        if whether_adapter:
            if args.benchmark == 'coco':
                self.class_num = 20
            elif args.benchmark == 'pascal':
                self.class_num = 5
            elif args.benchmark == 'coco2pascal':
                if self.fold in [0,2]:
                    self.class_num = 6
                elif self.fold in [1,3]:
                    self.class_num = 4
            self.adapter_blocks = nn.ModuleList()
            self.gap = 6
            for i in range(depth):
                if i % self.gap == 0:
                    self.adapter_blocks.append(Prototype_Adaptive_Module(dim=dim, 
                                            input_resolution=input_resolution, 
                                            hidden_ratio=args.hidden_ratio,
                                            drop=args.drop_ratio,
                                            class_num=self.class_num,
                                            momentum=args.momentum))
            
        
    def forward_query_and_support(self, query_img, support_img, support_mask, class_idx):
        """
        support_feature [list] 
            len(support_feature) = self.
            
        supoort_mask [tensor]
            shape = (B, H, W)     one-shot
            shape = (B, S, H, W)  few-shot

        adapter_block [nn.Module]
        """
        feats_q = []
        feats_sup = []
        B, N, D = query_img.shape 
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                query_img = blk(query_img)
                support_img = blk(support_img)
                if self.whether_adapter:
                    if i % self.gap == 0:
                        x_adapter_q, x_adapter_sup = self.adapter_blocks[i // self.gap](query_img, support_img, support_mask, class_idx)
                        query_img = query_img + x_adapter_q * self.adapter_weight
                        support_img = support_img + x_adapter_sup * self.adapter_weight

            feats_q.append(query_img)
            feats_sup.append(support_img)
            
        if self.downsample is not None:
            query_img = self.downsample(query_img)
            support_img = self.downsample(support_img)

        return feats_q, feats_sup, query_img, support_img
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops




def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, args,img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, adapter_weight=0.,
                 use_checkpoint=False, feat_ids=[1, 2, 3, 4], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        if args.benchmark == 'coco':
            self.class_num = 20
        elif args.benchmark == 'pascal':
            self.class_num = 5

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_adp = [x.item() for x in torch.linspace(0, 0.4, sum(depths))]  # stochastic depth decay rule


        self.layers = nn.ModuleList()
        self.whether_adapter = [False, False, True, True]
        for i_layer in range(self.num_layers):
            layer = BasicLayer(args, dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               adapter_weight = adapter_weight,
                               whether_adapter=self.whether_adapter[i_layer],
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.feat_ids = feat_ids

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    

    def forward_features_query_and_sup_branch(self, query_img, support_img, mask, stack_id, class_idx):
        """
        stack_id [list]
            stack_id = [2, 4, 22, 24]
        """
        query_img = self.patch_embed(query_img)
        if self.ape:
            query_img = query_img + self.absolute_pos_embed
        query_img = self.pos_drop(query_img)

        support_img= self.patch_embed(support_img)
        if self.ape:
            support_img = support_img + self.absolute_pos_embed
        support_img = self.pos_drop(support_img)
        id_0 = torch.zeros(1).int()
        stack_id = torch.cat((id_0, stack_id), dim=0)
        
        self.feat_maps_sup = []
        self.feat_maps_q = []
        for i, layer in enumerate(self.layers):
            feats_q, feats_sup, query_img, support_img = \
                layer.forward_query_and_support(query_img, support_img, mask, class_idx=class_idx)
            if i + 1 in self.feat_ids:
                self.feat_maps_sup += feats_sup
                self.feat_maps_q += feats_q

        return query_img, self.feat_maps_q, self.feat_maps_sup

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops  


if __name__ == '__main__':
    input = torch.randn(2, 3, 384, 384).cuda()

    net = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    out = net.forward_features(input)
    feat = net.feat_maps
    for x in feat:
        print(x.shape)
