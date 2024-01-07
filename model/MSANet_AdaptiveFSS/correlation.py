r""" Provides functions that builds/manipulates correlation tensors """
import torch


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5
        corrs = []
        sups=[]
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            queryShape = query_feat.shape#b,c,h,w
            corrI=[]
            realSupI=[]
            for j in range(len(support_feat)):#b
                queryIJ=query_feat[j].flatten(start_dim=1)#c,hw
                queryIJNorm=queryIJ/(queryIJ.norm(dim=0, p=2, keepdim=True) + eps)
                supIJ=support_feat[j]#c,hw
                supIJNorm=supIJ/(supIJ.norm(dim=0, p=2, keepdim=True) + eps)
                corr=(queryIJNorm.permute(1,0)).matmul(supIJNorm)
                corr = corr.clamp(min=0)
                corr=corr.mean(dim=1,keepdim=True)
                corr=(corr.permute(1,0)).unsqueeze(0)#1,1,hw
                corrI.append(corr)#b,1,hw

            corrI=torch.cat(corrI,dim=0)#b,1,h,w
            corrI=corrI.reshape((corrI.shape[0],corrI.shape[1],queryShape[-2],queryShape[-1]))#b,1,h,w

            corrs.append(corrI)#n,b,1,h,w


        corr_l4 = torch.cat(corrs[-stack_ids[0]:],dim=1).contiguous()#b,n,h,w
        corr_l3 = torch.cat(corrs[-stack_ids[1]:-stack_ids[0]],dim=1).contiguous()
        corr_l2 = torch.cat(corrs[-stack_ids[2]:-stack_ids[1]],dim=1).contiguous()

        return [corr_l4, corr_l3, corr_l2] #,[sup_l4,sup_l3,sup_l2]


    @classmethod
    def multilayer_correlation_hsnet(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            # support_feat_norm = torch.norm(support_feat, dim=1, p=2, keepdim=True)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            # query_feat_norm = torch.norm(query_feat, dim=1, p=2, keepdim=True)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)


            corr = torch.bmm(query_feat.transpose(1, 2), support_feat)
            corr = corr.clamp(min=0)
            corr = corr.mean(dim=2,keepdim=True).squeeze(2)
            corr = corr.view(bsz, hb, wb)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]