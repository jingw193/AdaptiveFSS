r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torch.utils.data.distributed import DistributedSampler as Sampler
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pascal import DatasetPASCAL_finetune, DatasetPASCAL_finetune_5shot , DatasetPASCAL_COCO2Pascalx
from data.coco import  DatasetCOCO_finetune,DatasetCOCO_finetune_5shot

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, model_name):

        cls.datasets = {
            'pascal': DatasetPASCAL_finetune,
            'coco': DatasetCOCO_finetune,
            'coco2pascal': DatasetPASCAL_COCO2Pascalx
        }

        cls.model_name = model_name
        
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])
        

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold,
                                          transform=cls.transform,
                                          split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize,
                                          model_name = cls.model_name)
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        train_sampler = Sampler(dataset) if split == 'trn' else None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, sampler=train_sampler, num_workers=nworker,
                                pin_memory=True)

        return dataloader

class FSSDataset_5shot:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, model_name):

        cls.datasets = {
            'pascal': DatasetPASCAL_finetune_5shot,
            'coco': DatasetCOCO_finetune_5shot,
        }


        cls.model_name = model_name
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        cls.train_transform = transforms.Compose([
                                    transforms.Resize(size=(img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(cls.img_mean, cls.img_std)])
        cls.val_transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(cls.img_mean, cls.img_std)])
        

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0
        if split == 'trn':
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold,
                                            transform=cls.train_transform,
                                            split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize,
                                            model_name = cls.model_name)
        else:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold,
                                            transform=cls.val_transform,
                                            split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize,
                                            model_name = cls.model_name)
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        train_sampler = Sampler(dataset) if split == 'trn' else None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, sampler=train_sampler, num_workers=nworker,
                                pin_memory=True,drop_last= False)

        return dataloader