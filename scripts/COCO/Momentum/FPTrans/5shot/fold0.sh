cls=0;
lr=1e-2;
nshot=5;
nepoch=300;
adapter_weight=0.1;
std_init=0.001;
weight_decay=1e-3;
model_name="fptrans_adaptivefss";
description="Momentum";
train_weight_path="backbones/coco/FPTrans/five_shot_DeiT/fold0.pth";
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=20000 \
./train_finetune.py --datapath "dataset/" \
           --benchmark coco \
           --fold $cls \
           --bsz 2 \
           --nworker 4 \
           --std_init $std_init \
           --backbone vit \
           --drop_ratio 0.5 \
           --hidden_ratio 16 \
           --momentum 0.99 \
           --model_name $model_name \
           --adapter_weight $adapter_weight \
           --train_weight_path $train_weight_path \
           --image_size 480 \
           --test_freq 5 \
           --logpath "./logs/AdaptiveFSS/FPTrans/COCO/five_shot" \
           --lr $lr \
           --weight_decay $weight_decay \
           --nshot $nshot \
           --nepoch $nepoch \
           --description $description 
