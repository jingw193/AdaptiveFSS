cls=0;
lr=1e-2;
nshot=5;
nepoch=300;
adapter_weight=0.1;
std_init=0.001;
weight_decay=1e-3;
model_name="msanet_adaptivefss";
description="Momentum";
train_weight_path="backbones/coco/MSANet/five_shot/resnet50_5_0_0.5351.pth";
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=10000 \
./train_finetune.py --datapath "dataset/" \
           --benchmark coco \
           --fold $cls \
           --bsz 1 \
           --nworker 2 \
           --std_init $std_init \
           --backbone resnet50 \
           --drop_ratio 0.5 \
           --hidden_ratio 16 \
           --momentum 0.99 \
           --model_name $model_name \
           --adapter_weight $adapter_weight \
           --train_weight_path $train_weight_path \
           --image_size 641 \
           --test_freq 1 \
           --logpath "./logs/AdaptiveFSS/MSANet/COCO/five_shot" \
           --lr $lr \
           --weight_decay $weight_decay \
           --nshot $nshot \
           --nepoch $nepoch \
           --description $description \