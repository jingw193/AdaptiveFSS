import torch 
from torch import nn 
from torch._C import device 
import torch.nn.functional as F 
from torch.nn import BatchNorm2d as BatchNorm 

from model.MSANet_AdaptiveFSS.PAM import Prototype_Adaptive_Module 

import model.MSANet_AdaptiveFSS.resnet as models
import model.MSANet_AdaptiveFSS.vgg as vgg_models
from model.MSANet_AdaptiveFSS.ASPP import ASPP
from model.MSANet_AdaptiveFSS.PPM import PPM
from model.MSANet_AdaptiveFSS.PSPNet import OneModel as PSPNet

from model.MSANet_AdaptiveFSS.feature import extract_feat_res, extract_feat_vgg, extract_feat_res_PAM
from functools import reduce
from operator import add
from model.MSANet_AdaptiveFSS.correlation import Correlation
from math import ceil
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram

class Attention(nn.Module):
    """
    Guided Attention Module (GAM).

    Args:
        in_channels: interval channel depth for both input and output
            feature map.
        drop_rate: dropout rate.
    """

    def __init__(self, in_channels, drop_rate=0.5):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()

        self.cls_type = 'Novel'  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.nshot
        self.vgg = args.vgg
        self.dataset = args.benchmark
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq / 2
        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60

        assert self.layers in [50, 101, 152]

        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)

        if backbone_str == 'vgg':
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.nsimlairy = [1,3,3]
        elif backbone_str == 'resnet50':
            self.feat_ids = list(range(3, 17)) 
            self.extract_feats = extract_feat_res_PAM
            nbottlenecks = [3, 4, 6, 3] 
            self.nsimlairy = [3,6,4]
            dims = [256,512,1024,2048]
            
            self.adapter_weight = args.adapter_weight
            self.fold = args.fold
            # self.adapter_num = [5,8,11]
            self.benchmark = args.benchmark
            if args.benchmark == 'coco':
                self.class_num = 20
            elif args.benchmark == 'pascal':
                self.class_num = 5
            elif args.benchmark == 'coco2pascal':
                if self.fold in [0,2]:
                    self.class_num = 6
                elif self.fold in [1,3]:
                    self.class_num = 4
            
            self.adapter_block = nn.ModuleList()
            self.insert_adapter = [False, False, False, True]
            self.adapter_position = [[],
                                     [],
                                     [],
                                     [0,]]
            stage_rsolution = [ceil(args.image_size/4),
                                    ceil(ceil(args.image_size/4)/2),
                                    ceil(ceil(args.image_size/4)/2),
                                    ceil(ceil(args.image_size/4)/2)]
            for idx, do_adapter in enumerate(self.insert_adapter):
                if do_adapter:
                    not_empty_list = nn.ModuleList()
                    for i in range(nbottlenecks[idx]):
                        if i in self.adapter_position[idx]:
                            adapter = Prototype_Adaptive_Module(dim=dims[idx], 
                                                              input_resolution=(stage_rsolution[idx],stage_rsolution[idx]), 
                                                              hidden_ratio=args.hidden_ratio,
                                                              drop=args.drop_ratio,
                                                              class_num=self.class_num,
                                                              momentum=args.momentum)
                            not_empty_list.append(adapter)
                        else:
                            not_empty_list.append(nn.Identity())
                    self.adapter_block.append(not_empty_list)
                else:
                    empty_list=  nn.ModuleList()
                    self.adapter_block.append(empty_list)
        elif backbone_str == 'resnet101':
            self.feat_ids = list(range(3, 34))      #stage2-4 
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
            self.nsimlairy = [3,23,4]
        else:
            raise Exception('Unavailable backbone: %s' % backbone_str)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks))) #  [0,1,2,0,1,2,3,0,...]把每个stage内的block按照顺序输出
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])   #    [1,1,1,2,2,2,2,33333]跟上面对应，表示stage
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]  #[3,9,13]

        PSPNet_ = PSPNet(args)


        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 256 + 64 + 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.nshot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()
        self.gam=Attention(in_channels=256)
        self.hyper_final = nn.Sequential(
            nn.Conv2d(sum(nbottlenecks[-3:]), 64, kernel_size=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def forward(self, x, s_x, s_y, class_idx = None):

        
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
        
        
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        
        # with torch.no_grad():              #[B,S,C,H,W] 的list
        query_feats, query_backbone_layers, support_feats_full_shot, support_backbone_layers_full_shot,= \
                                self.extract_feats(x, s_x, s_y, self.adapter_block, self.insert_adapter, self.adapter_position,
                                                   [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4], 
                                                   self.feat_ids, self.bottleneck_ids, self.lids, self.shot, class_idx_new, self.adapter_weight)


        if self.vgg:
            query_feat = F.interpolate(query_backbone_layers[2], size=(query_backbone_layers[3].size(2),query_backbone_layers[3].size(3)),\
                 mode='bilinear', align_corners=True)
            query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
        else:
            query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)
        
        query_feat = self.down_query(query_feat)
        
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = []
        corrs = []
        gams = 0
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)

            support_feats = [s_feats_per_shot[:,i,:,:] for s_feats_per_shot in support_feats_full_shot]
            support_backbone_layers = [s_feats_per_shot[:,i,:,:] for s_feats_per_shot in support_backbone_layers_full_shot]
            
            final_supp_list.append(support_backbone_layers[4])
                
            if self.vgg:
                supp_feat = F.interpolate(support_backbone_layers[2], size=(support_backbone_layers[3].size(2),support_backbone_layers[3].size(3)),
                                            mode='bilinear', align_corners=True)
                supp_feat = torch.cat([support_backbone_layers[3], supp_feat], 1)
            else:
                supp_feat = torch.cat([support_backbone_layers[3], support_backbone_layers[2]], 1)
            
            mask_down = F.interpolate(mask, size=(support_backbone_layers[3].size(2), support_backbone_layers[3].size(3)), mode='bilinear', align_corners=True)
            supp_feat = self.down_supp(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask_down)
            supp_pro_list.append(supp_pro)
            supp_feat_list.append(support_backbone_layers[2])

            support_feats_1 = self.mask_feature(support_feats, mask)
            corr = Correlation.multilayer_correlation(query_feats, support_feats_1, self.stack_ids)
            corrs.append(corr)

            gams += self.gam(supp_feat, mask)
        
        gam = gams / self.shot
        corrs_shot = [corrs[0][i] for i in range(len(self.nsimlairy))]
        for ly in range(len(self.nsimlairy)):
            for s in range(1, self.shot):
                corrs_shot[ly] +=(corrs[s][ly])
                
        
        hyper_4 = corrs_shot[0] / self.shot
        hyper_3 = corrs_shot[1] / self.shot
        if self.vgg: 
            hyper_2 = F.interpolate(corr[2], size=(corr[1].size(2),corr[1].size(3)), mode='bilinear', align_corners=True)
        else:
            hyper_2 = corrs_shot[2] / self.shot
        
        hyper_final = torch.cat([hyper_2, hyper_3, hyper_4],1)

        hyper_final = self.hyper_final(hyper_final)

        # K-Shot Reweighting
        que_gram = get_gram_matrix(query_backbone_layers[2])  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_backbone_layers[4]
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)



        # Tile & Cat
        concat_feat = supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, hyper_final, corr_query_mask, gam], 1)  # 256+256+1
        merge_feat = self.init_merge(merge_feat)

        # Base and Meta
        base_out = self.learner_base(query_backbone_layers[4])

        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)  # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:, 0:1, :, :]  # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:, 1:, :, :]  # [bs, 1, 60, 60]
        if class_idx != None and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = class_idx_new[b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)

        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if class_idx != None:
            out = dict()
            out['out'] = final_out
            out['aux'] = meta_out
            return out
        else:
            return final_out.max(1)[1]


    def mask_feature(self, features, support_mask):#bchw
        bs=features[0].shape[0]
        initSize=((features[0].shape[-1])*2,)*2
        support_mask = (support_mask).float()
        support_mask = F.interpolate(support_mask, initSize, mode='bilinear', align_corners=True)
        for idx, feature in enumerate(features):
            feat=[]
            if support_mask.shape[-1]!=feature.shape[-1]:
                support_mask = F.interpolate(support_mask, feature.size()[2:], mode='bilinear', align_corners=True)
            for i in range(bs):
                featI=feature[i].flatten(start_dim=1)#c,hw
                maskI=support_mask[i].flatten(start_dim=1)#hw
                featI = featI * maskI
                maskI=maskI.squeeze()
                meanVal=maskI[maskI>0].mean()
                realSupI=featI[:,maskI>=meanVal]
                if maskI.sum()==0:
                    realSupI=torch.zeros(featI.shape[0],1).cuda()
                feat.append(realSupI)#[b,]ch,w
            features[idx] = feat#nfeatures ,bs,ch,w
        return features
    
    def load_state_dict_for_train(self,path):
        state_dict = torch.load(path)['state_dict']
        self.load_state_dict(state_dict,False)
        
    def init_adapter(self,std):
        with torch.no_grad(): 
            for name, param in self.adapter_block.named_parameters(): 
                init_value = 0 
                if 'adapter' in name: 
                    init_value += torch.normal(0, std, size=param.size()) 
                    param.copy_(init_value) 
                    
    def train_mode(self):
        self.eval()
