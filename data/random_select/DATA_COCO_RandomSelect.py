import os
import numpy as np
import pickle

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def select_sample(fold,nshot,model):
    import random
    if '1' in fold:
        different_fold = 'fold3'
    else:
        different_fold = 'fold1'
        
    path10 = "data/splits/coco/trn/{}.pkl".format(different_fold)
    target_val={'fold0_val':[0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76],
                'fold1_val':[1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77],
                'fold2_val':[2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78],
                'fold3_val':[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79]}
    result = load_pickle(path10)
    new_dit = {}
    for key,value in result.items():
        if key in target_val['{}_val'.format(fold)]:
            select2 = random.sample(value,nshot+1)
            new_dit[key] = select2
        else:
            new_dit[key] = []
    if nshot ==1 :
        endsign = ''
    else:
        endsign = '_5shot'
    write_pickle(new_dit,'data/splits_{}/coco/trn/{}{}.pkl'.format(model,fold,endsign))
    b=load_pickle('data/splits_{}/coco/trn/{}{}.pkl'.format(model,fold,endsign))
    print(b)
    
if __name__ == '__main__':
    model = 'MSANet'
    assert model in ['DCAMA','FPTrans','MSANet']
    select_sample(fold='fold1', nshot=5, model=model) 