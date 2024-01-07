r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch
from model.init_model import  load_model
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset, FSSDataset_5shot
import setproctitle
from common.vis import Visualizer


def train(args, epoch, model, dataloader, optimizer_adapter, cross_entropy_loss, training):
    r""" Train """
    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        optimizer_adapter.zero_grad()
        # 1. forward pass
        batch = utils.to_cuda(batch)

        if 'fptrans' in args.model_name:
            out = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], batch['query_mask'].unsqueeze(1), class_idx=batch['class_id'])
        elif 'dcama' in args.model_name :
            out = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], class_idx=batch['class_id'])
        elif 'msanet' in args.model_name:
            out = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], class_idx=batch['class_id'])
        else:
            raise ValueError("Cannot find correct parameters for the forward of {}".format(args.model_name))
        logit_mask = out['out']
        pred_mask = logit_mask.argmax(dim=1)
        
        
        # 2. Compute loss & update model parameters
        loss = compute_objective(logit_mask, batch['query_mask'], cross_entropy_loss)
        if 'fptrans' in args.model_name:
            loss_prompt = compute_objective(out['out_prompt'], batch['query_mask'], cross_entropy_loss)
            loss_pair = out['loss_pair']
            loss = loss + 1e-4 * loss_prompt + 1e-4 * loss_pair # these loss weight hyperparameter value is following FPTrans default settings
        elif 'msanet' in args.model_name:
            loss_aux = compute_objective(out['aux'], batch['query_mask'], cross_entropy_loss)
            loss = loss + loss_aux
        if training:
            loss.backward()
            optimizer_adapter.step()
            
        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)

        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        # if not training:
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

def compute_objective(logit_mask, gt_mask, loss_type):
    bsz = logit_mask.size(0)
    logit_mask = logit_mask.view(bsz, 2, -1)
    gt_mask = gt_mask.view(bsz, -1).long()
    return loss_type(logit_mask, gt_mask)

def test(args, model, dataloader, nshot):
    r""" Test """
    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    
    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)
        if 'fptrans' in args.model_name: 
            out = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
            pred_mask = out['out'].argmax(dim=1)
            
        elif 'dcama' in args.model_name:
            out = model.module.predict_mask_nshot(batch, nshot)
            pred_mask=out['out']
            
        elif 'msanet' in args.model_name:
            out = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
            pred_mask=out
        else:
            raise ValueError("Cannot find correct parameters for the testing forward of {}".format(args.model_name))
            
        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou, all_iou = average_meter.compute_iou_finetune()

    return miou, fb_iou, all_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)


    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    model = load_model(args)
    model.load_state_dict_for_train(args.train_weight_path)

    for key, value in model.named_parameters():
        if 'adapter' in key:
            value.requires_grad = True
        else: 
            value.requires_grad = False

    all_number = sum(p.numel() for p in model.parameters())
    train_number = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger.info(all_number)
    Logger.info(train_number)
    
    for key, p in model.named_parameters():
        if p.requires_grad:
            Logger.info(key)

    model.init_adapter(std=args.std_init)
    device = torch.device("cuda", args.local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=False)

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer_adapter = optim.SGD([{"params": [p for name, p in model.named_parameters() if p.requires_grad], "lr": args.lr ,
                            "momentum": 0.9, "weight_decay": args.weight_decay, "nesterov": True}])

    Evaluator.initialize()


    # Dataset initialization
    if args.nshot == 1 :
            FSSDataset.initialize(img_size=args.image_size, datapath=args.datapath, use_original_imgsize=False, model_name= args.model_name)
            dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=args.nshot)
            if args.local_rank == 0:
                dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', shot=args.nshot)
    elif args.nshot == 5 :
            FSSDataset_5shot.initialize(img_size=args.image_size, datapath=args.datapath, use_original_imgsize=False, model_name= args.model_name)
            if 'dcama' in args.model_name:
                dataloader_trn = FSSDataset_5shot.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=1)
                if args.local_rank == 0:
                    dataloader_val = FSSDataset_5shot.build_dataloader(args.benchmark, 1, args.nworker, args.fold, 'test', shot=args.nshot)
            else:
                dataloader_trn = FSSDataset_5shot.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=args.nshot)
                if args.local_rank == 0:
                    dataloader_val = FSSDataset_5shot.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', shot=args.nshot)
    
    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')

    for epoch in range(args.nepoch):
        setproctitle.setproctitle("Adapter_FSS:{}/{}".format(epoch, args.nepoch))
        dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, dataloader_trn, optimizer_adapter, cross_entropy_loss, training=True)
        if args.local_rank == 0 and (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                model.eval()
                val_miou, val_fb_iou, all_iou = test(args, model, dataloader_val, args.nshot)
            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_all_miou(model, epoch, val_miou, all_iou)
    if args.local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
