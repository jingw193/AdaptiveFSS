cls=0;
lr=1e-2;
nshot=1;
nepoch=300;
adapter_weight=0.1;
std_init=0.001;
weight_decay=1e-3;
model_name="fptrans_adaptivefss";
description="Momentum";
train_weight_path="backbones/pascal/FPTrans/one_shot_DeiT/fold0.pth";
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=10000 \
./train_finetune.py --datapath "dataset/" \
           --benchmark pascal \
           --fold $cls \
           --bsz 4 \
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
           --test_freq 10 \
           --logpath "./logs/AdaptiveFSS/FPTrans/Pascal/one_shot" \
           --lr $lr \
           --weight_decay $weight_decay \
           --nshot $nshot \
           --nepoch $nepoch \
           --description $description 
