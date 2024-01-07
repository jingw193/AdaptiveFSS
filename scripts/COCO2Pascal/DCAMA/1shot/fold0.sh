cls=0;
lr=5e-2;
nshot=1;
nepoch=300;
adapter_weight=0.0;
std_init=0.01;
weight_decay=1e-3;
model_name="dcama_adaptivefss";
description="Momentum_COCO2Pascal";
train_weight_path="backbones/coco/DCAMA/swin_fold0.pt";
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=10600 \
./train_finetune.py --datapath "dataset/" \
           --benchmark coco2pascal \
           --fold $cls \
           --bsz 4 \
           --nworker 4 \
           --std_init $std_init \
           --backbone swin \
           --drop_ratio 0.5 \
           --hidden_ratio 16 \
           --momentum 0.99 \
           --model_name $model_name \
           --adapter_weight $adapter_weight \
           --train_weight_path $train_weight_path \
           --image_size 384 \
           --test_freq 10 \
           --logpath "./logs/AdaptiveFSS/DCAMA/COCO2Pascal/test" \
           --lr $lr \
           --weight_decay $weight_decay \
           --nshot $nshot \
           --nepoch $nepoch \
           --description $description 
           #76.46