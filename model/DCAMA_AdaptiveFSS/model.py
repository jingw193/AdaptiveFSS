r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import os
from model.DCAMA_AdaptiveFSS.swin_transformer import SwinTransformer, Mlp, DropPath
from model.base.transformer import MultiHeadedAttention, PositionalEncoding
import numpy as np

class DCAMA_AdaptiveFSS(nn.Module):

    def __init__(self, args):
        super(DCAMA_AdaptiveFSS, self).__init__()

        self.backbone = args.backbone
        
        self.use_original_imgsize = False
        self.adapter_weight = args.adapter_weight

        # feature extractor initialization
        if self.backbone == 'swin':
            self.feature_extractor = SwinTransformer(args=args,img_size=384, patch_size=4, window_size=12, embed_dim=128, adapter_weight = args.adapter_weight, 
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)

        self.shots = args.nshot
        
        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.adapter_ids = list(np.array(self.stack_ids) - np.array(self.nlayers))
        self.adapter_ids = [self.adapter_ids[0]]

        
        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids)
        self.fold = args.fold
        self.benchmark = args.benchmark
        if args.benchmark == 'coco':
            self.class_num = 20
        elif args.benchmark == 'pascal':
            self.class_num = 5

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def init_adapter(self, std): 
        with torch.no_grad(): 
            for name, param in self.feature_extractor.named_parameters(): 
                init_value = 0 
                if 'adapter' in name: 
                    init_value += torch.normal(0, std, size=param.size()) 
                    param.copy_(init_value) 
                    
    def forward(self, query_img, support_img, support_mask, class_idx=None):
        """
        Parameters
        ----------
        x: torch.Tensor
                [B, C, H, W], query image
        s_x: torch.Tensor
                [B, C, H, W],    support image   one-shot
                [B, S, C, H, W], support image   few-shot (S)
        s_y: torch.Tensor
                [B, H, W],    support mask    one-shot
                [B, S, H, W], support mask    few-shot (S)
        y: torch.Tensor
                [B, 1, H, W], query mask, used for calculating the pair-wise loss
        Returns
        -------
        output:
                [B, 2, H, W]: torch.Tensor
        """
        B, C, H, W = query_img.size()

        if support_img.shape[1] != 1:
            support_img = support_img.view(B * self.shots, C, H, W)  #----> [BS,C,H,W]
            support_mask_for_adapter = support_mask.view(B * self.shots, H, W).unsqueeze(1)
        else: 
            support_img = support_img.squeeze(1)
            support_mask_for_adapter = support_mask
        self.BxS = support_img.shape[0]
        

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
                for i,idx in enumerate(class_idx):
                    if self.fold == 0 :
                        class_idx_new[i] = torch.nonzero(change_list_1==class_idx[i])[0]
                    elif self.fold == 1 :
                        class_idx_new[i] = torch.nonzero(change_list_2==class_idx[i])[0]
                    elif self.fold == 2 :
                        class_idx_new[i] = torch.nonzero(change_list_3==class_idx[i])[0]
                    elif self.fold == 3 :
                        class_idx_new[i] = torch.nonzero(change_list_4==class_idx[i])[0]
        else:
            class_idx_new = class_idx
 
        query_feats, support_feats = self.extract_feats_q_and_sup(query_img, support_img, support_mask_for_adapter, class_idx_new)

        logit_mask = self.model(query_feats, support_feats, support_mask.squeeze(1).clone())

        out = {}
        out['out'] = logit_mask

        return out

    def forward_5shot_test(self, query_img, support_img, support_mask, class_idx=None):
        """
        Parameters
        ----------
        x: torch.Tensor
                [B, C, H, W], query image
        s_x: torch.Tensor
                [B, S, C, H, W], support image   few-shot (S)
        s_y: torch.Tensor
                [B, S, H, W], support mask    few-shot (S)
        y: torch.Tensor
                [B, 1, H, W], query mask, used for calculating the pair-wise loss
        Returns
        -------
        output:
                [B, 2, H, W]: torch.Tensor
        """
        B, C, H, W = query_img.size()

        support_img = support_img.view(B * self.shots, C, H, W) 
        support_mask_for_adapter = support_mask.view(B * self.shots, H, W).unsqueeze(1)

        self.BxS = support_img.shape[0]
        
        if class_idx is not None:
            if self.benchmark == 'coco':
                class_idx = (class_idx - self.fold) // 4
            elif self.benchmark == 'pascal':
                class_idx = class_idx - (self.fold * self.class_num)


        feats_q = []
        feats_sup = []
        if self.backbone == 'swin':
            _, feat_maps_q, feat_maps_sup = self.feature_extractor.forward_features_query_and_sup_branch(query_img, support_img, support_mask_for_adapter, self.stack_ids, class_idx)
            for feat_q, feat_sup in zip(feat_maps_q, feat_maps_sup):
                bsz, hw, c = feat_q.size()
                h = int(hw ** 0.5)
                
                feat_q = feat_q.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feat_sup = feat_sup.view(self.BxS, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats_q.append(feat_q)
                feats_sup.append(feat_sup) 
        new_support_feats = []  
        for shot in range(self.shots):
            feas_each_shot = []
            for feas in feats_sup:
                _,C,H,W=feas.shape
                feas_each_shot.append(feas.view(B, self.shots, C, H, W)[:,shot,:,:,:])
            new_support_feats.append(feas_each_shot)
        logit_mask = self.model(feats_q, new_support_feats, support_mask.clone(), self.shots)

        out = {}
        out['out'] = logit_mask

        return out

    def extract_feats_q_and_sup(self, query_img, support_img, mask, class_idx=None):
        r""" Extract input image features """
        feats_q = []
        feats_sup = []
        if self.backbone == 'swin':
            _, feat_maps_q, feat_maps_sup = self.feature_extractor.forward_features_query_and_sup_branch(query_img, support_img, mask, self.stack_ids, class_idx)
            for feat_q, feat_sup in zip(feat_maps_q, feat_maps_sup):
                bsz, hw, c = feat_q.size()
                h = int(hw ** 0.5)                
                feat_q = feat_q.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feat_sup = feat_sup.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats_q.append(feat_q)
                feats_sup.append(feat_sup)              
                
        return feats_q, feats_sup
    
    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            out = self(query_img, support_imgs[:, 0, :, :], support_masks[:, 0, :, :].unsqueeze(1))
            logit_mask = out['out']
        else:
            out = self.forward_5shot_test(query_img, support_imgs, support_masks)
            logit_mask = out['out']
 
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        out['out'] = logit_mask.argmax(dim=1)
 
        return out

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()

    def load_state_dict_for_train(self, path):
        state_dict = torch.load(path)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_network_dict = self.state_dict()
        for k, _ in self.state_dict().items():
            if k in state_dict.keys():
                new_network_dict[k] = state_dict[k]        
        self.load_state_dict(new_network_dict, strict=False)    
        
class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids

        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        self.mixer1 = nn.Sequential(nn.Conv2d(outch3+2*in_channels[1]+2*in_channels[0], outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

    def forward(self, query_feats, support_feats, support_mask, nshot=1):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            if nshot == 1:
                support_feat = support_feats[idx]
                mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True).view(support_feat.size()[0], -1)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
                mask = mask.view(bsz, -1)

            # DCAMA blocks forward
            if idx < self.stack_ids[1]:
                coarse_mask = self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask)
            elif idx < self.stack_ids[2]:
                coarse_mask = self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask)
            else:
                coarse_mask = self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask)
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        mix = coarse_masks1 + coarse_masks2
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3
        mix = self.conv5(mix)

        # skip connect 1/8 and 1/4 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)

        # mixer blocks forward
        out = self.mixer1(mix)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        return logit_mask

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)
