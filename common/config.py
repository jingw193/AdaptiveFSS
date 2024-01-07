r"""config"""
import argparse

def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(description='Adaptive FSS for Few-Shot Segmentation tasks on the various FSS methods')
    # Common
    parser.add_argument('--datapath', type=str, default='./datasets')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco','coco2pascal'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='swin', choices=['resnet50', 'resnet101', 'swin', 'vit'])
    parser.add_argument('--feature_extractor_path', type=str, default='')
    parser.add_argument('--train_weight_path', type=str, default='')
    parser.add_argument('--logpath', type=str, default='./logs')

    # Train
    parser.add_argument('--std_init', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--nepoch', type=int, default=1000)
    parser.add_argument('--adapter_weight',type=float, default=0.1)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--enhancement', default=0, type=int )
    parser.add_argument('--description', type=str, default='adapter only mlp to query feature')
    parser.add_argument('--model_name',type=str, default='fptrans')
    parser.add_argument('--hidden_ratio',type=int, default=16)
    parser.add_argument('--drop_ratio',type=float, default=0.1)
    parser.add_argument('--momentum',type=float, default=0.9)
    parser.add_argument('--image_size',type=int, default=384)
    parser.add_argument('--test_freq',type=int, default=10)
    parser.add_argument('--iter', type=int, default=1000)
    
    # Test
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vispath', type=str, default='./vis')
    parser.add_argument('--use_original_imgsize', action='store_true')

    # Hyperparameters for MSANet
    parser.add_argument('--layers', type=int, default=50)
    parser.add_argument('--vgg', type=bool, default=False)
    parser.add_argument('--aux_weight1', type=float, default=1.0)
    parser.add_argument('--aux_weight2', type=float, default=1.0)
    parser.add_argument('--low_fea', type=str, default='layer2')
    parser.add_argument('--kshot_trans_dim', type=int, default=2)
    parser.add_argument('--merge', type=str, default='final')
    parser.add_argument('--merge_tau', type=float, default=0.9)
    parser.add_argument('--zoom_factor', type=int, default=8)
    parser.add_argument('--ignore_label', type=int, default=255)
    parser.add_argument('--print_freq', type=int, default='10')
    args = parser.parse_args()
    return args