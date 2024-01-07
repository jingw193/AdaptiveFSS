r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset ,DataLoader
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

from common.utils import fix_randseed
class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold    #
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        fix_randseed(321)
        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx): 
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name) # 
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample 

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('data/splits/pascal/%s/fold%d.txt' % (split, fold_id))  
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            # print(fold_n_metadata)
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds): #[0-3]
                if fold_id == self.fold:  
                    continue
                img_metadata += read_metadata(self.split, fold_id) 
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
 
        return img_metadata 
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise  


class DatasetPASCAL_finetune(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, model_name):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold    
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform
        if 'dcama' in model_name:
            self.splits_finetune = 'splits_DCAMA'
        elif 'fptrans' in model_name:
            self.splits_finetune = 'splits_FPTrans'
        elif 'msanet' in model_name:
            self.splits_finetune = 'splits_MSANet'
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()


    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx): 
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name) 
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample 

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [self.fold * nclass_trn + i for i in range(nclass_trn)]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('data/%s/pascal/%s/fold_%d.txt' % (self.splits_finetune, split, fold_id))  
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            img_metadata = read_metadata(self.split, self.fold)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
 
        return img_metadata  
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise  
 
    
class DatasetPASCAL_finetune_5shot(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize,model_name):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold    
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform
        if 'dcama' in model_name:
            self.splits_finetune = 'splits_DCAMA'
        elif 'fptrans' in model_name:
            self.splits_finetune = 'splits_FPTrans'
        elif 'msanet' in model_name:
            self.splits_finetune = 'splits_MSANet'
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()


    def __len__(self):
        # return len(self.img_metadata) #if self.split == 'trn' else 1000
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx): 
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name) 
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample 

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [self.fold * nclass_trn + i for i in range(nclass_trn)]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            if split =='trn':
                fold_n_metadata = os.path.join('data/%s/pascal/%s/fold_%d_5shot.txt' % (self.splits_finetune, split, fold_id))   
            elif split =='val':
                fold_n_metadata = os.path.join('data/%s/pascal/%s/fold_%d.txt' % (self.splits_finetune, split, fold_id))   
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            # print(fold_n_metadata)
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            img_metadata = read_metadata(self.split, self.fold)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
 
        return img_metadata  #返回的是['2011_001856', 19], ['2011_001926', 19] 这种数据

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        # print(img_metadata_classwise)
        return img_metadata_classwise   # 
    
   

class DatasetPASCAL_COCO2Pascalx(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, model_name):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold    
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform
        if 'dcama' in model_name:
            self.splits_finetune = 'splits_DCAMA'
        elif 'fptrans' in model_name:
            self.splits_finetune = 'splits_FPTrans'

        class_names = {
        "pascal": ["background",
                   "airplane", "bicycle", "bird", "boat", "bottle",#0-4       #对应coco的[1,20,43,2,69]
                   "bus", "car", "cat", "chair", "cow",            #5-9         #对应coco的[21,40,63,14,64]
                   "dining table", "dog", "horse", "motorbike", "person",#10-14   #对应coco的[15,4,24,60,0]   motobike 和motocycle应该是一个东西
                   "potted plant", "sheep", "sofa", "train", "tv"  #15-19            #对应coco的[54,44,35,41,55] sofa和couch应该是一个东西
                   ],
        "coco": ["background",
                 "person", "airplane", "boat", "parking meter", "dog", "elephant", "backpack",# 0-6
                 "suitcase", "sports ball", "skateboard", "wine glass", "spoon", "sandwich", "hot dog", #7-13
                 "chair", "dining table", "mouse", "microwave", "refrigerator", "scissors",#  14-19
                 
                 "bicycle", "bus", "traffic light", "bench", "horse", "bear", "umbrella",# 20-26
                 "frisbee", "kite", "surfboard", "cup", "bowl", "orange", "pizza", #27-33
                 "couch", "toilet", "remote", "oven", "book", "teddy bear", #34-39
                 
                 "car", "train", "fire hydrant", "bird", "sheep", "zebra", "handbag", #40-46
                 "skis", "baseball bat", "tennis racket", "fork", "banana", "broccoli", "donut", #47-53
                 "potted plant", "tv", "keyboard", "toaster", "clock", "hair drier", #54 - 59
                 
                 "motorcycle", "truck", "stop sign", "cat", "cow", "giraffe", "tie", #60 -66 
                 "snowboard", "baseball glove", "bottle", "knife", "apple", "carrot", "cake",# 67-73
                 "bed", "laptop", "cell phone", "sink", "vase", "toothbrush", #74 - 79
                 ]
        }
        self.cls_list = list(range(0,20))
        if fold == 0:
            self.sub_val_list = [1, 4, 9, 11, 12, 15]
        elif fold == 1:
            self.sub_val_list = [2, 6, 13, 18]
        elif fold == 2:
            self.sub_val_list = [3, 7, 16, 17, 19, 20]
        elif fold == 3:
            self.sub_val_list = [5, 8, 10, 14]
            
        # self.remain_fold_list = list(set(self.cls_list) - set(self.sub_val_list))
        
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
    def __len__(self):
        # return len(self.img_metadata) #if self.split == 'trn' else 1000
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx): 
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name) 
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample 

    def build_class_ids(self):
        # nclass_trn = self.nclass // self.nfolds
        class_ids_val = [idxx-1 for idxx in self.sub_val_list]#[self.fold * nclass_trn + i for i in range(nclass_trn)]
        
        class_ids_trn = [idxx-1 for idxx in self.sub_val_list]#[self.fold * nclass_trn + i for i in range(nclass_trn)]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('data/%s/pascal/%s/fold_%d_COCO2Pascal.txt' % (self.splits_finetune, split, fold_id))   
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            # print(fold_n_metadata)
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            img_metadata = read_metadata(self.split, self.fold)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
 
        return img_metadata  #返回的是['2011_001856', 19], ['2011_001926', 19] 这种数据

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        # print(img_metadata_classwise)
        return img_metadata_classwise   # 


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data.distributed import DistributedSampler as Sampler
    import data.transform as tf
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_size=378
    train_transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(img_mean, img_std)])
    split = 'val'
    
    dataset = DatasetPASCAL_COCO2Pascalx(datapath= './dataset', fold=0, model_name='fptrans',transform=train_transform, split=split, shot=2, use_original_imgsize=False)
    # train_sampler = Sampler(dataset) if split == 'trn' else None
    print(dataset.img_train_metadata)
    print(dataset.img_metadata_classwise[1])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=1,
                                pin_memory=True)
    for data in dataloader:
        print(data['support_names'],data['class_id'])
        
    
# 修改思路:
# 如果做Finetune，该如何做，
# 假设 fold=1 ，对应的类别是0-4的话
# 那 query 是 类别0-4内的图像
# 那support呢，也是0-4内和query不同的图像？
# 如果是这个思路，就需要另起一个Dataset_finetune
# 这个 Dataset 内只取 0-4的数据，然后需要划分训练和测试的样本？
# 还是说就拿少量的图去训练，这个我不确定
