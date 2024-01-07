import os
from random import sample
def random_select(fold,nshot,model):
    object_fold = fold
    nshot = nshot
    root = 'data/splits/pascal/trn'
    path_metadata =[]
    for fold in [object_fold+'.txt',]:
        path = os.path.join(root,fold)
        with open(path, 'r') as f:
            metadata = f.read().split('\n')[:-1]
        path_metadata.extend(metadata)
    print(len(path_metadata))
    fullobj_cls =[['01','02','03','04','05'],
            ['06','07','08','09','10'],
            ['11','12','13','14','15'],
            ['16','17','18','19','20']]
    obj_cls = [fullobj_cls[int(object_fold[-1])]]
    for idx,obj in enumerate(obj_cls):
        samples=[]
        for k in obj:
            cls_list =[]
            for key in path_metadata:
                cls = key[-2:]
                if cls == k:
                    cls_list.append(key + '\n')
            if nshot==2:
                random_sample = sample(cls_list, nshot)
            else: 
                random_sample = sample(cls_list, nshot+1)
            samples = samples+random_sample
        if nshot == 5:
            path = 'data/splits_{}/pascal/trn/fold_{}_5shot.txt'.format(model,int(object_fold[-1]))
        elif nshot == 2 :
            path = 'data/splits_{}/pascal/trn/fold_{}_2shot.txt'.format(model,int(object_fold[-1]))
        else :
            path = 'data/splits_{}/pascal/trn/fold_{}.txt'.format(model,int(object_fold[-1]))
        with open(path, 'w') as f:
            metadata = f.writelines(samples)
if __name__ =='__main__':
    model = 'DCAMA'
    assert model in ['DCAMA','FPTrans','MSANet']
    for i in ['0','1','2','3']:
        random_select(fold='fold'+i,nshot=19, model= model) #or'DCAMA'