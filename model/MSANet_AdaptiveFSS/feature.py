r""" Extracts intermediate features from given backbone network & layer ids """


from collections import Counter
import numpy as np
import torch
def extract_feat_vgg(img, backbone_layers, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feat_ids_1 = [0, 3, 6]
    feats = []
    layers = []
    feat = img
    feat = backbone_layers[0](feat)
    layers.append(feat)
    feat = backbone_layers[1](feat)
    layers.append(feat)
    feat = backbone_layers[2](feat)
    layers.append(feat)

    for layers_34 in [backbone_layers[3], backbone_layers[4]]:
        for lid, module in enumerate(layers_34):
            feat = module(feat)
            if lid in feat_ids_1:
                feats.append(feat.clone())
        layers.append(feat)
    feats.append(feat.clone())
    return feats, layers

    
def extract_feat_res_PAM(img, support_img,support_label, adapter, do_insert_adapter, adapter_position, 
                             backbone_layers, feat_ids, bottleneck_ids, lids, nshot, class_idx , adapter_weight):
    r""" Extract intermediate features from ResNet"""
    q_feats = []
    s_feats = []
    # Layer 0
    B, S, C, H, W = support_img.shape
    support_img = support_img.reshape(B*S, C, H, W)
    image_input = torch.cat([support_img, img], dim=0)
    feat = backbone_layers[0](image_input) #.conv1.forward(img)
    
    layer_nums = np.cumsum(list(Counter(lids).values()))
    layer_nums_iter = iter(layer_nums)
    layer_id = next(layer_nums_iter) #[3,7,13,16]
    _,c,h,w = feat.shape
    q_layers = [feat[B*S:, :, :, :].clone()]

    s_layers = [feat[:B*S, :, :, :].reshape(B, S, c, h, w).clone()]
    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone_layers[lid][bid].conv1.forward(feat)
        feat = backbone_layers[lid][bid].bn1.forward(feat)
        feat = backbone_layers[lid][bid].relu.forward(feat)
        feat = backbone_layers[lid][bid].conv2.forward(feat)
        feat = backbone_layers[lid][bid].bn2.forward(feat)
        feat = backbone_layers[lid][bid].relu.forward(feat)
        feat = backbone_layers[lid][bid].conv3.forward(feat)
        feat = backbone_layers[lid][bid].bn3.forward(feat)
        if bid == 0:
            res = backbone_layers[lid][bid].downsample.forward(res)
        feat += res
        
        # Adapter
        _,c,h,w = feat.shape
        if do_insert_adapter[(lid-1)] == True:
            if bid in adapter_position[lid-1]:
                q_feat_adapter = feat[B*S:, :, :, :].clone()
                s_feat_adapter = feat[:B*S, :, :, :].clone()
                image_token_q = q_feat_adapter.reshape(B,c,-1).permute(0,2,1)
                image_token_sup = s_feat_adapter.reshape(B*S,c,h*w).permute(0,2,1)
                image_token_adapter = adapter[lid-1][bid](image_token_q, image_token_sup, support_label, class_idx)
                q_feat = q_feat_adapter + adapter_weight * image_token_adapter[B*S:,:,:].permute(0,2,1).reshape(B,c,h,w)
                s_feat = s_feat_adapter + adapter_weight * image_token_adapter[:B*S,:,:].permute(0,2,1).reshape(B*S,c,h,w)
                feat = torch.cat([s_feat, q_feat], dim=0)

        if hid + 1 in feat_ids:
            q_feats.append(feat[B*S:, :, :, :].clone())
            s_feats.append(feat[:B*S, :, :, :].reshape(B, S, c, h, w).clone())  #[B,S,C,H,W]的list

        feat = backbone_layers[lid][bid].relu.forward(feat)
        
        if hid + 1 == layer_id :
            if layer_id != layer_nums[-1]:
                layer_id = next(layer_nums_iter)
            q_layers.append(feat[B*S:, :, :, :].clone().detach())     #4 个特征，每个stage最后一个
            s_layers.append(feat[:B*S, :, :, :].reshape(B, S, c, h, w).clone().detach())
    return q_feats, q_layers, s_feats, s_layers
    
def extract_feat_res(img, backbone_layers, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []
    feats
    # Layer 0
    feat = backbone_layers[0](img) #.conv1.forward(img)
    
    layer_nums = np.cumsum(list(Counter(lids).values()))
    layer_nums_iter = iter(layer_nums)
    layer_id = next(layer_nums_iter) #[3,7,13,16]
    q_layers = [feat]
    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone_layers[lid][bid].conv1.forward(feat)
        feat = backbone_layers[lid][bid].bn1.forward(feat)
        feat = backbone_layers[lid][bid].relu.forward(feat)
        feat = backbone_layers[lid][bid].conv2.forward(feat)
        feat = backbone_layers[lid][bid].bn2.forward(feat)
        feat = backbone_layers[lid][bid].relu.forward(feat)
        feat = backbone_layers[lid][bid].conv3.forward(feat)
        feat = backbone_layers[lid][bid].bn3.forward(feat)

        if bid == 0:
            res = backbone_layers[lid][bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone_layers[lid][bid].relu.forward(feat)
        
        if hid + 1 == layer_id :
            if layer_id != layer_nums[-1]:
                layer_id = next(layer_nums_iter)
            q_layers.append(feat.clone())     #4个特征，每个stage最后一个

    return feats, q_layers
    