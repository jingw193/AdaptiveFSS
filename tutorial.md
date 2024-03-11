## Prepare Datasets and Environment

This section can refer to the [DCAMA](https://github.com/pawn-sxy/DCAMA). Our data preparation and environment configuration is similar to it.



## Prepare Pre-trained Few-shot Segmentation Models

Downloading the following pre-trained FSS model:

> - [DCAMA](https://github.com/pawn-sxy/DCAMA) trained on PASCAL and COCO with Swin-Transformer.
> - [FPTrans](https://github.com/Jarvis73/FPTrans) trained on PASCAL and COCO with ViT.
> - [MSANet](https://github.com/AIVResearch/MSANet) trained on PASCAL and COCO with ResNet50.

Creating a directory 'backbones' to place the above models. The overall directory structure should be like this:

```
├── backbones/          
	├── coco/           
		├── DCAMA/
	        	├── swin_fold0.pt   
	            	....
	            	└── swin_fold3.pt
	        ├── FPTrans/            
	        	├── one_shot_DeiT/  
	        		├── fold0.pth    
	        		....
	        		└── fold3.pth
	        	└── five_shot_DeiT/      
	            	├── fold0.pth            
	        		....
	        		└── fold3.pth
	    	└── MSANet/            
	        	├── one_shot/          
	        		├── resnet50_0_0.4834.pth          
	        		....
	        		└── resnet50_3_0.4533.pth
	        	└── five_shot/             
		            	├── resnet50_5_0_0.5351.pth              
	        		....
	        		└── resnet50_5_3_0.5093.pth
	├── pascal/            
		├── DCAMA/
	        	├── swin_fold0.pt   
		    	....
		    	└── swin_fold3.pt
	        ├── FPTrans/
	        	├── deit_base_distilled_patch16_384-d0272ac0.pth  # the deit pre-trained ViT-base checkpoint
	        	├── one_shot_DeiT/  
	        		├── fold0.pth    
	        		....
	        		└── fold3.pth
	        	└── five_shot_DeiT/      
		            	├── fold0.pth            
	        		....
	        		└── fold3.pth
	    	└── MSANet/            
	        	├── one_shot/          
	        		├── resnet50_0_0.6925.pth          
	        		....
	        		└── resnet50_3_0.6240.pth
	        	└── five_shot/             
	            		├── resnet50_5_0_0.7306.pth              
	        		....
	        		└── resnet50_5_3_0.6882.pth   
```

You can also down this directory from [Google Drive](https://drive.google.com/drive/folders/19bA4xbQ8ah38ij3m3hmK-wqHDXOwMMiA?usp=drive_link)
You can down the deit_base_distilled_patch16_384-d0272ac0.pth from [here](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth)



## Finetuning the base Few-shot segmentation model

For example, you can use this command to adapt the DCAMA to the novel classes at the PASCAL-5i fold 0 set.

```
sh ./scripts/Pascal/Momentum/DCAMA/1shot/fold0.sh
```

For the five-shot setting, you can run this command.

```
sh ./scripts/Pascal/Momentum/DCAMA/5shot/fold0.sh
```

> Besides, we use the Python scripts in the directory ./data/random_select/ to random select the training samples
