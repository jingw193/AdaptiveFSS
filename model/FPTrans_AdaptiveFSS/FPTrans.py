import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.hub import download_url_to_file
from model.FPTrans_AdaptiveFSS.losses import PairwiseLoss
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math, copy

from model.FPTrans_AdaptiveFSS.vit import vit_model
from model.FPTrans_AdaptiveFSS.misc import interpb, interpn
from model.FPTrans_base.vit import vit_model as vit_model_original

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
            registered_feas: torch.Tensor
                [B * (S + 1), h * w, dim], injected features
            """
            B, N, D = x.shape
            s_f = s_f.reshape(B, -1, N, D)
            S = s_f.size(1)

            s_y = F.interpolate(s_y, (self.input_resolution[0], self.input_resolution[1]), mode='nearest')
            sup_mask_fg = (s_y == 1).float().reshape(B, S, -1).reshape(B, -1).unsqueeze(-1) #[B, S * N, 1]
            samentic_prototype, sign_fore_per_batch = self.extract_samentic_prototype(s_f, sup_mask_fg)

            if class_idx is not None:
                with torch.no_grad():
                    self.train()
                    new_samentic_prototype = self.updata_prototype_bank(samentic_prototype, class_idx, sign_fore_per_batch)
            else:
                self.eval()
                new_samentic_prototype = self.select_prototype_bank(samentic_prototype, self.prototype)

            enhanced_feat_q = self.enhanced_feature(x.unsqueeze(1), new_samentic_prototype, sign_fore_per_batch)
            enhanced_feat_sup = self.enhanced_feature(s_f, new_samentic_prototype, sign_fore_per_batch)

            registered_feas = torch.cat([enhanced_feat_sup.reshape(-1, N, D), enhanced_feat_q.squeeze(1)], dim=0)
            registered_feas = self.proj_drop(self.linears_up(self.act(self.linears_down(registered_feas))))
            return registered_feas


      def extract_samentic_prototype(self, s_f, s_y):
          """
          extract temporary class prototype according to support features and masks
          input:
            s_f: torch.Tensor
                [B, S, N, D], support features
            s_y: torch.Tensor
                [B, S * N, 1], support masks
          output:
            samentic_prototype: torch.Tensor
                [B, D], temporary prototypes
            sign_fore_per_batch: torch.Tensor
                [B], the signal of wether including foreground region in this image
          """
          B, S, N, D = s_f.shape
          num_fore_per_batch = torch.count_nonzero(s_y.reshape(B, -1), dim=1)
          s_y = s_y.repeat(1, 1, D)
          samentic_prototype = s_y * s_f.reshape(B, -1, D)
          samentic_prototype = samentic_prototype.mean(1) * (N * S) / (num_fore_per_batch.unsqueeze(1)+1e-4)
          one = torch.ones_like(num_fore_per_batch).cuda()
          sign_fore_per_batch =  torch.where(num_fore_per_batch > 0.5, one, num_fore_per_batch)
          return samentic_prototype, sign_fore_per_batch
      
      def updata_prototype_bank(self, samentic_prototype, class_idx, sign_fore_per_batch):
          """
          updata prototype in class prototype bank during traning
          input:
            samentic_prototype: torch.Tensor
                [B, D]
            class_id: list
                len(class_id) = B
            sign_fore_per_batch: torch.Tensor
                [B], the signal of wether including foreground region in this image
          output:
            new_samentic_prototype: torch.Tensor
                [B, D], the updated prototypes for feature enhancement
          """  
          B, D = samentic_prototype.shape
          self.prototype = nn.functional.normalize(self.prototype, dim=0)
          samentic_prototype = nn.functional.normalize(samentic_prototype, dim=1)
          new_samentic_prototype_list = []
          for i in range(B):
               samentic_prototype_per = samentic_prototype[i,: ]
               class_idx_per = class_idx[i]
               if sign_fore_per_batch[i] == 1:
                    new_samentic_prototype_per = self.prototype[:, class_idx_per] * self.momentum + (1 - self.momentum) * samentic_prototype_per
                    self.prototype[:, class_idx_per] = new_samentic_prototype_per
               else: 
                    new_samentic_prototype_per = self.prototype[:, class_idx_per]    
               new_samentic_prototype_list.append(new_samentic_prototype_per)
          new_samentic_prototype = torch.stack(new_samentic_prototype_list, dim=0)
          return new_samentic_prototype

      def select_prototype_bank(self, samentic_prototype, prototype_bank):
          """
          select prototypes in class prototype bank during testing
          input:
            samentic_prototype: torch.Tensor
                shape = [B, D]
            prototype_bank: torch.Tensor
                shape = [D, class_num]
          output:
            new_samentic_prototype: torch.Tensor
                [B, D], the prototypes for feature enhancement
          """ 
          B, D = samentic_prototype.shape
          prototype_bank = nn.functional.normalize(prototype_bank, dim=0)
          samentic_prototype = nn.functional.normalize(samentic_prototype, dim=1)
          similar_matrix = samentic_prototype @ prototype_bank  # [B, class_num]
          idx = similar_matrix.argmax(1)

          new_samentic_prototype_list = []
          for i in range(B):
              new_samentic_prototype_per = prototype_bank[:, idx[i]]
              new_samentic_prototype_list.append(new_samentic_prototype_per)
          new_samentic_prototype = torch.stack(new_samentic_prototype_list, dim=0)
          return new_samentic_prototype

      def enhanced_feature(self, feature, new_samentic_prototype, sign_fore_per_batch):
          """ 
            Input:
                feature: torch.Tensor
                    [B, S, N, D]
                new_samentic_prototype: torch.Tensor
                    [B, D]
            Outputs:
                enhanced_feature: torch.Tensor
                    [B, S, N, D]
            """
          B, D = new_samentic_prototype.shape
          feature_sim = nn.functional.normalize(feature, p=2, dim=-1)
          new_samentic_prototype = nn.functional.normalize(new_samentic_prototype, p=2, dim=1)
          similarity_matrix_list = []
          for i in range(B):
              feature_sim_per = feature_sim[i,:, :, :]
              new_samentic_prototype_er = new_samentic_prototype[i, :]
              similarity_matrix_per = feature_sim_per @ new_samentic_prototype_er
              similarity_matrix_list.append(similarity_matrix_per)
          similarity_matrix = torch.stack(similarity_matrix_list, dim=0)

          similarity_matrix = (similarity_matrix * self.dim ** 0.5) *  sign_fore_per_batch.unsqueeze(-1).unsqueeze(-1)

          enhanced_feature = self.act_enhance(similarity_matrix).unsqueeze(-1).repeat(1, 1, 1, D) * feature + feature


          return enhanced_feature

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class FPTrans_AdaptiveFSS(nn.Module):
    def __init__(self, args):
        super(FPTrans_AdaptiveFSS, self).__init__()
        self.shot = args.nshot
        self.drop_dim = 1  # int, 1 for 1D Dropout, 2 for 2D DropBlock
        self.drop_rate = 0.1 # float, drop rate used in the DropBlock of the purifier
        self.block_size = 4  # int, block size used in the DropBlock of the purifier
        self.drop2d_kwargs = {'drop_prob': self.drop_rate, 'block_size': self.block_size }

        self.bg_num = 5
        self.adapter_weight = args.adapter_weight
        self.fold = args.fold
        if args.benchmark == 'coco':
            self.class_num = 20
        elif args.benchmark == 'pascal':
            self.class_num = 5
        elif args.benchmark == 'coco2pascal':
            if self.fold in [0,2]:
                self.class_num = 6
            elif self.fold in [1,3]:
                self.class_num = 4
        self.benchmark = args.benchmark

        self.adapter_num = [6, 7, 8, 9, 10]
        self.adapter_block = nn.ModuleList()
        for i in self.adapter_num:
            self.adapter_block.append(Prototype_Adaptive_Module(dim=768, 
                                                              input_resolution=(30,30), 
                                                              hidden_ratio=args.hidden_ratio,
                                                              drop=args.drop_ratio,
                                                              class_num=self.class_num,
                                                              momentum=args.momentum))


        # Main model
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', vit_model('DeiT-B/16-384',
                                       args.image_size,
                                       pretrained='',
                                       num_classes=0,
                                       shot=self.shot,
                                       benchmark=self.benchmark))
        ]))

        embed_dim = 768
        self.purifier = self.build_upsampler(embed_dim)
        self.__class__.__name__ = f"FPTrans/DeiT-B/16-384"

        # Pretrained model
        self.original_encoder = vit_model_original('DeiT-B/16-384',
                                            args.image_size,
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
        with torch.no_grad(): 
            for name, param in self.adapter_block.named_parameters(): 
                init_value = 0 
                init_value += torch.normal(0, std, size=param.size()) 
                param.copy_(init_value) 

    def train_mode(self):
        self.eval()


    def forward(self, x, s_x, s_y, y=None, out_shape=None, class_idx=None):
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
        class_idx_new = class_idx
        if class_idx is not None:  
            if self.benchmark == 'coco':
                class_idx_new = (class_idx - self.fold) // 4
            elif self.benchmark == 'pascal':
                class_idx_new = class_idx - (self.fold * self.class_num)
            elif self.benchmark == 'coco2pascal':
                class_idx_new = class_idx.clone()
                change_list_1 =  torch.tensor([0, 3, 8, 10, 11, 14]).cuda()
                change_list_2 =  torch.tensor([1, 5, 12, 17]).cuda()
                change_list_3 =  torch.tensor([2, 6, 15, 16, 18, 19]).cuda()
                change_list_4 =  torch.tensor([4, 7, 9, 13]).cuda()
                for i, idx in enumerate(class_idx):
                    if self.fold == 0 :
                        class_idx_new[i] = torch.nonzero(change_list_1==class_idx[i])[0]
                    elif self.fold == 1 :
                        class_idx_new[i] = torch.nonzero(change_list_2==class_idx[i])[0]
                    elif self.fold == 2 :
                        class_idx_new[i] = torch.nonzero(change_list_3==class_idx[i])[0]
                    elif self.fold == 3 :
                        class_idx_new[i] = torch.nonzero(change_list_4==class_idx[i])[0]

        img_cat = (img_cat, (fg_token, bg_token))
        all_cat = (img_cat, self.adapter_block, s_y, self.adapter_weight, class_idx_new)
        backbone_out = self.encoder(all_cat)

        features = self.purifier(backbone_out['out'])               # [B(S+1), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S+1, c, h, w)                   # [B, S+1, c, h, w]-----
        sup_fts, qry_fts = features.split([S, 1], dim=1)            # [B, S, c, h, w] / [B, 1, c, h, w]
        sup_mask = interpn(s_y.view(B * S, 1, H, W), (h, w))        # [BS, 1, h, w]

        pred = self.classifier(sup_fts, qry_fts, sup_mask)          # [B, 2, h, w]

        # Output
        if not out_shape:
            out_shape = y.shape[-2:] if y is not None else (H, W)


        out = interpb(pred, out_shape)    # [BQ, 2, *, *]
        output = dict(out=out,)
        
        if class_idx is not None and y is not None:
            # Pairwise loss
            x1 = sup_fts.flatten(3)                 # [B, S, C, N]
            y1 = sup_mask.view(B, S, -1).long()     # [B, S, N]
            x2 = qry_fts.flatten(3)                 # [B, 1, C, N]

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


    def load_state_dict_for_train(self, path):
        state_dict = torch.load(path)['model_state']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_network_dict = self.state_dict()
        for k, _ in self.state_dict().items():
            if k in state_dict.keys():
                new_network_dict[k] = state_dict[k]          
        self.load_state_dict(new_network_dict, strict=False)    


    def get_params_list(self):
        params = []
        for var in self.parameters():
            if var.requires_grad:
                params.append(var)
        return [{'params': params}]


    