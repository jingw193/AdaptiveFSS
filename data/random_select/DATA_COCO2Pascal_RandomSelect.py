import os
from random import sample
import random

def random_select(fold,nshot,model):
    object_fold = fold
    nshot = nshot
    root = 'data/splits/pascal/trn'
    test_root = 'data/splits/pascal/val'
    path_metadata =[]
    path_eval_metadata =[]
    for fold in ['fold0.txt','fold1.txt','fold2.txt','fold3.txt']:
        path = os.path.join(root,fold)
        eval_path = os.path.join(test_root,fold)
        with open(path, 'r') as f:
            metadata = f.read().split('\n')[:-1]
        with open(eval_path, 'r') as f:
            eval_metadata = f.read().split('\n')[:-1]
        path_metadata.extend(metadata)
        path_eval_metadata.extend(eval_metadata)
        
    print(len(path_metadata))
    sub_cls =[['01', '04', '09', '11', '12', '15'],
            ['02', '06', '13', '18'],
            ['03', '07', '16', '17', '19', '20'],
            ['05', '08', '10', '14']]
    
    full = ['01','02','03','04','05',
            '06','07','08','09','10',
            '11','12','13','14','15',
            '16','17','18','19','20']
    obj_cls = [sub_cls[int(object_fold[-1])]]
    print(obj_cls)
    for idx, obj in enumerate(obj_cls):
        samples=[]
        for k in obj:
            cls_list =[]
            for key in path_metadata:
                cls = key[-2:]
                if cls == k:
                    cls_list.append(key + '\n')
            random_sample = sample(cls_list, nshot+1)
            samples = samples+random_sample
    for idx, obj in enumerate(obj_cls):
        eval_samples=[]
        for k in obj:
            cls_list =[]
            for key in path_eval_metadata:
                cls = key[-2:]
                if cls == k:
                    cls_list.append(key + '\n')
            eval_samples = eval_samples+ cls_list
    random.shuffle(eval_samples)
    path = 'data/splits_{}/pascal/trn/fold_{}_COCO2Pascal.txt'.format(model,int(object_fold[-1]))
    eval_path = 'data/splits_{}/pascal/val/fold_{}_COCO2Pascal.txt'.format(model,int(object_fold[-1]))
    
    with open(path, 'w') as f:
        metadata = f.writelines(samples)
        
        
if __name__ =='__main__':
    model = ''
    assert model in ['DCAMA','FPTrans','MSANet']
    for i in ['fold1']:
        random_select(fold=i,nshot=1, model= 'FPTrans') 