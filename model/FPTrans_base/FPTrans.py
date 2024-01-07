import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dropblock import DropBlock2D
from torch.hub import download_url_to_file
from model.FPTrans_base.losses import PairwiseLoss
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math, copy

# from projects.FPTrans-main.constants import pretrained_weights, model_urls
# from losses import get as get_loss
from model.FPTrans_base.vit import vit_model
from model.FPTrans_base.misc import interpb, interpn

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

class Residual(nn.Module):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = interpb(x, (h * self.up, w * self.up))
        x = x_up + self.layers(x)
        return x
    
class Adapter_block_v6(nn.Module):
      """ 
      adapter block for query-support interaction.
      produce features for segmentation task.
      Args:
            dim (int): Number of input channels.
            input_resolution (tuple): input_resolution for high and weight
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
      """
      def __init__(self, 
                   dim, 
                   input_resolution, 
                   mlp_ratio=4., 
                   drop=0.,
                   drop_path=0.):
            super().__init__()
            self.dim = dim
            self.dim_low = dim // mlp_ratio
            # self.dim_low = 32
            self.input_resolution = input_resolution
            # self.qry_linear = nn.Linear(dim, dim, bias=qkv_bias)
            # self.sup_linear = nn.Linear(dim, dim, bias=qkv_bias)
            # self.qry_v = nn.Linear(dim, dim, bias=qkv_bias)
            # self.linears_down = clones(nn.Linear(dim, dim), 3)
            self.act = nn.ReLU()
            self.linears_down_q = clones(nn.Linear(dim, self.dim_low), 3)
            self.linears_up_q = clones(nn.Linear(self.dim_low, dim), 3)
            self.linears_down_sup = clones(nn.Linear(dim, self.dim_low), 3)
            self.linears_up_sup = clones(nn.Linear(self.dim_low, dim), 3)
            self.registered_mlp = Mlp(in_features=dim, hidden_features=self.dim_low, act_layer=nn.ReLU, drop=drop)
            self.registered_mlp_sup = Mlp(in_features=dim, hidden_features=self.dim_low, act_layer=nn.ReLU, drop=drop)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.proj_reg_down = nn.Linear(dim, self.dim_low)
            self.proj_reg_up = nn.Linear(self.dim_low, dim)
            self.proj_drop = nn.Dropout(drop)
            self.proj_reg_down_sup = nn.Linear(2 * dim, 2 * self.dim_low)
            self.proj_reg_up_sup = nn.Linear(2* self.dim_low, dim)
            # self.sup_downsample_down = nn.Linear(dim*2, self.dim_low * 2)
            # self.sup_downsample_up = nn.Linear(self.dim_low * 2, dim)
            self.loss_align = nn.MSELoss()
            self.aver_pool = nn.AdaptiveAvgPool1d(1)

      def forward(self, query_img, support_img, support_mask):
            """ 
            adapter block for query-support interaction.
            Args:
            query_img: torch.Tensor
                  [B , h * w, dim], query 
            support_img: torch.Tensor
                  [B , h * w, dim],    support_features one-shot   
            s_y: torch.Tensor
                  [B, H, W],    support mask one-shot
            Outputs:
            registered_feas_q: torch.Tensor
                  [B, h * w, dim], injected query features
            registered_feas_sup: torch.Tensor
                  [B, h * w, dim], injected query features
            """
            # origin_x = x
            B, N, D = support_img.shape
            # import ipdb
            # ipdb.set_trace()

            qry, sup, qry_v = [l(x) for l, x in zip(self.linears_down_q, (query_img, support_img, query_img))]
            qry = self.act(qry)
            sup = self.act(sup)
            qry_v = self.act(qry_v)
            qry, sup, qry_v = [l(x) for l, x in zip(self.linears_up_q, (qry, sup, qry_v))]

            sup_q, qry_s, sup_v = [l(x) for l, x in zip(self.linears_down_sup, (support_img, query_img, support_img))]
            sup_q = self.act(sup_q)
            qry_s = self.act(qry_s)
            sup_v = self.act(sup_v)
            sup_q, qry_s, sup_v = [l(x) for l, x in zip(self.linears_up_sup, (sup_q, qry_s, sup_v))]

            # qry, sup, qry_v = [l(x) for l, x in zip(self.linears_up, (qry, sup, qry_v))]
            qry = F.normalize(qry, dim=-1)
            sup = F.normalize(sup, dim=-1)
            qry_v = F.normalize(qry_v, dim=-1)
            sup_v = F.normalize(sup_v, dim=-1)

            # import ipdb
            # ipdb.set_trace()
            # s_y = F.interpolate(s_y, (self.input_resolution[0], self.input_resolution[1]), mode='bilinear', align_corners=True) # [B, S, h, w]
            s_y = F.interpolate(support_mask, (self.input_resolution[0], self.input_resolution[1]), mode='nearest')
            
            
            att_map = qry @ sup.transpose(2, 1) / D ** 0.5 #  [B, N_qry, N]
            
            # aux segmentation head
            sup_mask_fg = (s_y == 1).float().reshape(B, -1).unsqueeze(-1) #[B,N,1]
            sup_mask_bg = (s_y == 0).float().reshape(B, -1).unsqueeze(-1) #[B,N,1]
            att_map_for_aux = att_map
            fg_distance = att_map_for_aux @ sup_mask_fg # [B,  N, 1]
            bg_distance = att_map_for_aux @ sup_mask_bg
            pre = torch.stack((bg_distance, fg_distance), dim=1) # [B, 2, N, 1]
            # pre = torch.mean(pre, dim=2) # [B, 2, N, 1]
            pre = pre.reshape(B, 2, self.input_resolution[0], self.input_resolution[1])

            # att_map_for_registered
            att_map_for_registered = F.softmax(att_map, dim=-2) # [B, N, N1]
            # qry_v = qry_v.unsqueeze(1).repeat(1, S, 1, 1)
            registered_feas = att_map_for_registered.transpose(-1, -2) @ qry_v # [B, N, D]
            registered_feas = registered_feas + query_img

            # registered_feas = torch.mean(registered_feas, dim=1) # [B, N, D]
            
            registered_feas = registered_feas + self.drop_path(self.registered_mlp(self.norm1(registered_feas)))
            # registered_feas = registered_feas + self.drop_path(self.registered_mlp(registered_feas))
            # registered_feas = torch.cat([registered_feas, x], dim=-1)

            registered_feas_index = F.normalize(registered_feas, dim=1)
            x_index = F.normalize(query_img, dim=1)
            registered_feas_index = self.aver_pool(registered_feas_index.permute(0, 2, 1)).reshape(B, D)
            x_index = self.aver_pool(x_index.permute(0, 2, 1)).reshape(B, D)
            registered_feas_index = torch.cat([registered_feas_index, x_index], dim=-1)
            _, index_reg = torch.topk(registered_feas_index, k=D, dim=1)

            registered_feas = torch.cat([registered_feas, query_img], dim=-1)

            feats = []
            for i in range(B):
                index_reg_b = index_reg[i, :]
                feat = registered_feas[i, :, index_reg_b]
                feats.append(feat)

            registered_feas = torch.stack(feats, dim=0)
            registered_feas = self.proj_drop(self.proj_reg_up(self.act(self.proj_reg_down(registered_feas))))

            # for support image
            att_map_sup = qry_s @ sup_q.transpose(2, 1) / D ** 0.5
            att_map_for_sup = F.softmax(att_map_sup, dim=-1) # [B, N, N1]
            registered_feas_sup = att_map_for_sup @ sup_v # [B, N, D]
            registered_feas_sup = registered_feas_sup + support_img
            # registered_feas_sup = registered_feas_sup + self.drop_path(self.registered_mlp_sup(registered_feas_sup))
            registered_feas_sup = registered_feas_sup + self.drop_path(self.registered_mlp_sup(self.norm2(registered_feas_sup)))
            registered_feas_sup = torch.cat([registered_feas_sup, support_img], dim=-1)
            registered_feas_sup = self.proj_drop(self.proj_reg_up_sup(self.act(self.proj_reg_down_sup(registered_feas_sup))))

            return registered_feas, registered_feas_sup, self.loss_align(registered_feas, query_img.detach()), self.loss_align(registered_feas_sup, support_img.detach()), pre      

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class FPTrans_base(nn.Module):
    def __init__(self, args):
        super(FPTrans_base, self).__init__()
        # self.logger = logger
        self.shot = args.nshot
        self.drop_dim = 1  # int, 1 for 1D Dropout, 2 for 2D DropBlock
        self.drop_rate = 0.1 # float, drop rate used in the DropBlock of the purifier
        self.block_size = 4  # int, block size used in the DropBlock of the purifier
        self.drop2d_kwargs = {'drop_prob': self.drop_rate, 'block_size': self.block_size }

        self.bg_num = 5
        self.benchmark = args.benchmark
        # self.adapter_weight = adapter_weight
        # self.adapter_num = [2,5,8,11]
        # self.adapter_num = []
        # self.adapter_block = nn.ModuleList()
        # for i in self.adapter_num:
        #     self.adapter_block.append(Adapter_block_v6(dim=768, 
        #                                                       input_resolution=(30,30), 
        #                                                       mlp_ratio=16,
        #                                                       drop=0.5,
        #                                                       drop_path=0.))

        # import ipdb
        # ipdb.set_trace()
        # Check existence.
        # pretrained = self.get_or_download_pretrained(opt.backbone, opt.tqdm)

        # Main model
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', vit_model('DeiT-B/16-384',
                                       480,
                                       pretrained='',
                                       num_classes=0,
                                       depth=12,
                                       shot=self.shot,
                                       dataset=self.benchmark))
        ]))
        # self.encoder = vit_model('DeiT-B/16-384',
        #                                480,
        #                                pretrained='',
        #                                num_classes=0,
        #                                depth=10,
                                    #    shot=self.shot)
        # embed_dim = vit.vit_factory[opt.backbone]['embed_dim']
        embed_dim = 768
        self.purifier = self.build_upsampler(embed_dim)
        self.__class__.__name__ = f"FPTrans/DeiT-B/16-384"

        # Pretrained model
        self.original_encoder = vit_model('DeiT-B/16-384',
                                            480,
                                            pretrained='backbones/pascal/FPTrans/deit_base_distilled_patch16_384-d0272ac0.pth',
                                            num_classes=0,
                                            shot=self.shot,
                                            original=True,
                                            dataset=self.benchmark)
        for var in self.original_encoder.parameters():
            var.requires_grad = False

        # Define pair-wise loss
        self.pairwise_loss = PairwiseLoss()
        
        # Background sampler
        self.bg_sampler = np.random.RandomState(1289)

        # logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} created")

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        ))
    
    def init_adapter(self, std): 
        pass
        # with torch.no_grad(): 
        #     for name, param in self.adapter_block.named_parameters(): 
        #         init_value = 0 
        #         if 'adapter' in name: 
        #             init_value += torch.normal(0, std, size=param.size()) 
        #             param.copy_(init_value) 

    def train_mode(self):
        self.train()
        # self.encoder.eval()
        # self.original_encoder.eval()

    def forward(self, x, s_x, s_y, y=None, out_shape=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            [B, C, H, W], query image
        s_x: torch.Tensor
            [B, S, C, H, W], support image
        s_y: torch.Tensor
            [B, S, H, W], support mask
        y: torch.Tensor
            [B, 1, H, W], query mask, used for calculating the pair-wise loss
        out_shape: list
            The shape of the output predictions. If not provided, it is default
            to the last two dimensions of `y`. If `y` is also not provided, it is
            default to the [opt.height, opt.width].

        Returns
        -------
        output: dict
            'out': torch.Tensor
                logits that predicted by feature proxies
            'out_prompt': torch.Tensor
                logits that predicted by prompt proxies
            'loss_pair': float
                pair-wise loss
        """
        B, S, C, H, W = s_x.size()
        img_cat = torch.cat((s_x, x.view(B, 1, C, H, W)), dim=1).view(B*(S+1), C, H, W)

        # Calculate class-aware prompts
        with torch.no_grad():
            inp = s_x.view(B * S, C, H, W)
            # Forward
            sup_feat = self.original_encoder.forward_original(inp)['out']
            _, c, h0, w0 = sup_feat.shape
            sup_mask = interpn(s_y.view(B*S, 1, H, W), (h0, w0))                                # [BS, 1, h0, w0]
            sup_mask_fg = (sup_mask == 1).float()                                               # [BS, 1, h0, w0]
            # Calculate fg and bg tokens
            fg_token = (sup_feat * sup_mask_fg).sum((2, 3)) / (sup_mask_fg.sum((2, 3)) + 1e-6)
            fg_token = fg_token.view(B, S, c).mean(1, keepdim=True)  # [B, 1, c]
            bg_token = self.compute_multiple_prototypes(
                self.bg_num,
                sup_feat.view(B, S, c, h0, w0),
                sup_mask == 0,
                self.bg_sampler
            ).transpose(1, 2)    # [B, k, c]

        # Forward
        img_cat = (img_cat, (fg_token, bg_token))
        # all_cat = (img_cat, self.adapter_block, s_y, self.adapter_weight)
        backbone_out = self.encoder(img_cat)

        features = self.purifier(backbone_out['out'])               # [B(S+1), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S+1, c, h, w)                   # [B, S+1, c, h, w]-----
        sup_fts, qry_fts = features.split([S, 1], dim=1)            # [B, S, c, h, w] / [B, 1, c, h, w]
        sup_mask = interpn(s_y.view(B * S, 1, H, W), (h, w))        # [BS, 1, h, w]

        pred = self.classifier(sup_fts, qry_fts, sup_mask)          # [B, 2, h, w]

        # Output
        if not out_shape:
            out_shape = y.shape[-2:] if y is not None else (H, W)
        aux_pre_dict = {}
        for inde, pre_i in enumerate(backbone_out['aux_pre']):
            pre_i = interpb(pre_i, out_shape)  
            key_name = 'aux_pre_' + str(inde)
            aux_pre_dict[key_name] = pre_i
        out = interpb(pred, out_shape)    # [BQ, 2, *, *]
        # sum(backbone_out['distill_loss_q'])/len(backbone_out['distill_loss_q'])
        # sum(backbone_out['distill_loss_sup'])/len(backbone_out['distill_loss_sup'])
        output = dict(out=out,
                    #   distill_loss_q=sum(backbone_out['distill_loss_q'])/len(backbone_out['distill_loss_q']), 
                    #   distill_loss_sup=sum(backbone_out['distill_loss_sup'])/len(backbone_out['distill_loss_sup']), 
                      aux_pre=aux_pre_dict)
        
        if self.training and y is not None:
            # Pairwise loss
            x1 = sup_fts.flatten(3)                 # [B, S, C, N]
            y1 = sup_mask.view(B, S, -1).long()     # [B, S, N]
            x2 = qry_fts.flatten(3)                 # [B, 1, C, N]
            # import ipdb
            # ipdb.set_trace()
            y2 = interpn(y.float(), (h, w)).flatten(2).long()   # [B, 1, N]

            output['loss_pair'] = self.pairwise_loss(x1, y1, x2, y2)

            # Prompt-Proxy prediction
            fg_token = self.purifier(backbone_out['tokens']['fg'])[:, :, 0, 0]        # [B, c]
            bg_token = self.purifier(backbone_out['tokens']['bg'])[:, :, 0, 0]        # [B, c]
            bg_token = bg_token.view(B, self.bg_num, c).transpose(1, 2)     # [B, c, k]
            pred_prompt = self.compute_similarity(fg_token, bg_token, qry_fts.reshape(-1, c, h, w))

            # Up-sampling
            pred_prompt = interpb(pred_prompt, (H, W))
            output['out_prompt'] = pred_prompt

        return output

    def classifier(self, sup_fts, qry_fts, sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, S, c, h, w = sup_fts.shape

        # FG proxies
        sup_fg = (sup_mask == 1).view(-1, 1, h * w)  # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)     # [BS, c]
        # Merge multiple shots
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1)    # [B, c]

        # BG proxies
        bg_proto = self.compute_multiple_prototypes(self.bg_num, sup_fts, sup_mask == 0, self.bg_sampler)

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)
        pred = self.compute_similarity(fg_proto, bg_proto, qry_fts)   # [B, 2, h, w]
        return pred

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    @staticmethod
    def compute_multiple_prototypes(bg_num, sup_fts, sup_bg, sampler):
        """

        Parameters
        ----------
        bg_num: int
            Background partition numbers
        sup_fts: torch.Tensor
            [B, S, c, h, w], float32
        sup_bg: torch.Tensor
            [BS, 1, h, w], bool
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, c, k], where k is the number of background proxies

        """
        B, S, c, h, w = sup_fts.shape
        bg_mask = sup_bg.view(B, S, h, w)    # [B, S, h, w]
        batch_bg_protos = []

        for b in range(B):
            bg_protos = []
            for s in range(S):
                bg_mask_i = bg_mask[b, s]     # [h, w]

                # Check if zero
                with torch.no_grad():
                    if bg_mask_i.sum() < bg_num:
                        bg_mask_i = bg_mask[b, s].clone()    # don't change original mask
                        bg_mask_i.view(-1)[:bg_num] = True

                # Iteratively select farthest points as centers of background local regions
                all_centers = []
                first = True
                pts = torch.stack(torch.where(bg_mask_i), dim=1)     # [N, 2]
                for _ in range(bg_num):
                    if first:
                        i = sampler.choice(pts.shape[0])
                        first = False
                    else:
                        dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                        # choose the farthest point
                        i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                    pt = pts[i]   # center y, x
                    all_centers.append(pt)
            
                # Assign bg labels for bg pixels
                dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                bg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

                # Compute bg prototypes
                bg_feats = sup_fts[b, s].permute(1, 2, 0)[bg_mask_i]    # [N, c]
                for i in range(bg_num):
                    proto = bg_feats[bg_labels == i].mean(0)    # [c]
                    bg_protos.append(proto)

            bg_protos = torch.stack(bg_protos, dim=1)   # [c, k]
            batch_bg_protos.append(bg_protos)
        bg_proto = torch.stack(batch_bg_protos, dim=0)  # [B, c, k]
        return bg_proto

    @staticmethod
    def compute_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        if len(bg_proto.shape) == 3:    # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], dim=1) * dist_scalar        # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = torch.stack(bg_distances, dim=0).max(0)[0]
        else:   # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)               # [B, 2, h, w]

        return pred

    def load_weights(self, ckpt_path, logger, strict=True):
        """

        Parameters
        ----------
        ckpt_path: Path
            path to the checkpoint
        logger
        strict: bool
            strict mode or not

        """
        weights = torch.load(str(ckpt_path), map_location='cpu')
        if "model_state" in weights:
            weights = weights["model_state"]
        if "state_dict" in weights:
            weights = weights["state_dict"]
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        # Update with original_encoder
        weights.update({k: v for k, v in self.state_dict().items() if 'original_encoder' in k})

        self.load_state_dict(weights, strict=strict)        
        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} initialized from {ckpt_path}")

    # @staticmethod
    # def get_or_download_pretrained(backbone, progress):
    #     if backbone not in pretrained_weights:
    #         raise ValueError(f'Not supported backbone {backbone}. '
    #                          f'Available backbones: {list(pretrained_weights.keys())}')

    #     cached_file = Path(pretrained_weights[backbone])
    #     if cached_file.exists():
    #         return cached_file

    #     # Try to download
    #     url = model_urls[backbone]
    #     cached_file.parent.mkdir(parents=True, exist_ok=True)
    #     sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
    #     download_url_to_file(url, str(cached_file), progress=progress)
    #     return cached_file

    def load_state_dict_for_train(self, path):
        # state_dict = torch.load(path)
        state_dict = torch.load(path)['model_state']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_network_dict = self.state_dict()
        # import ipdb
        # ipdb.set_trace()
        for k, _ in self.state_dict().items():
            # if k[:7] == 'encoder':
            #     k_pth = k[:7] + '.backbone' + k[7:]
            #     if k_pth in state_dict.keys():
            #         new_network_dict[k] = state_dict[k_pth]
                #     print('sucesseful load weight', k)
                # else:
                #     print('do not load weight', k)
            # else:
            if k in state_dict.keys():
                new_network_dict[k] = state_dict[k]
                #     print('sucesseful load weight', k)
                # else:
                #     print('do not load weight', k)
            # else:
            #     print('random weight', k)            
        self.load_state_dict(new_network_dict, strict=False)    
        

    def load_state_dict_for_test(self, path):
        state_dict = torch.load(path)
        # state_dict = torch.load(path)['model_state']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_network_dict = self.state_dict()
        # import ipdb
        # ipdb.set_trace()
        for k, _ in self.state_dict().items():
            # if k[:7] == 'encoder':
            #     k_pth = k[:7] + '.backbone' + k[7:]
            #     if k_pth in state_dict.keys():
            #         new_network_dict[k] = state_dict[k_pth]
                #     print('sucesseful load weight', k)
                # else:
                #     print('do not load weight', k)
            # else:
            if k in state_dict.keys():
                new_network_dict[k] = state_dict[k]
                #     print('sucesseful load weight', k)
                # else:
                #     print('do not load weight', k)
            # else:
            #     print('random weight', k)            
        self.load_state_dict(new_network_dict, strict=False)


    def get_params_list(self):
        params = []
        for var in self.parameters():
            if var.requires_grad:
                params.append(var)
        return [{'params': params}]


    