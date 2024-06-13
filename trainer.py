import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch
from monai.utils.enums import MetricReduction
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import FBetaScore
from monai.metrics import MeanIoU
from loss.loss import eval_metric

hd_acc = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data) 
            loss = loss_func(logits, target)
    
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    run_acc_hd=AverageMeter()
    start_time = time.time()
    
    hd_avg_list=[]
    diceme_avg_list = []
    acc_avg_list = []
    iou_avg_list = []
    spe_avg_list = []
    pre_avg_list = []
    recall_avg_list = []
    fs_avg_list = []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)                   
                else:  
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)                          
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            
            #dice
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            dice, not_nans = acc_func.aggregate()
            dice = dice.cuda(args.rank)
            #hd
            hd_acc.reset()
            hd_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc_hd, not_nans_hd = hd_acc.aggregate()
            acc_hd=acc_hd.cuda(args.rank)
            

#            eval_metric
            diceme,acc,iou,spe,pre,recall,fs=eval_metric(y_pred=val_output_convert[0], y=val_labels_convert[0])

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [dice, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())
                if(str(acc_hd.item())!='inf'):
                    run_acc_hd.update(acc_hd.cpu().numpy(), n=not_nans_hd.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                
                avg_acc_hd=np.mean(acc_hd.item())                                
                diceme_avg_list.append(diceme)
                if (str(acc)!='nan'):
                    acc_avg_list.append(acc)
                if (str(iou)!='nan'):
                    iou_avg_list.append(iou)
                if (str(spe)!='nan'):
                    spe_avg_list.append(spe)
                if (str(pre)!='nan'):
                    pre_avg_list.append(pre)
                if (str(recall)!='nan'):
                    recall_avg_list.append(recall)
                if (str(fs)!='nan'):
                    fs_avg_list.append(fs)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",avg_acc,"diceme",diceme,"hd",avg_acc_hd,"iou",iou,"acc",acc,"pre",pre,"spe",spe,"recall",recall,"fs",fs,
                    "time {:.2f}s".format(time.time() - start_time),)
            start_time = time.time()
    return np.mean(avg_acc),np.mean(run_acc_hd.avg),np.mean(iou_avg_list),np.mean(acc_avg_list),np.mean(pre_avg_list),np.mean(spe_avg_list),np.mean(recall_avg_list),np.mean(fs_avg_list)


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),)
            
            open('Log.txt', 'a').write("Final training {}/{} loss: {:.4f} time {:.2f}s ".format(epoch, args.max_epochs - 1,train_loss,time.time() - epoch_time))
            
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            dice,hd,iou,acc,pre,spe,recall,fs = val_epoch(model,val_loader,epoch=epoch,acc_func=acc_func,model_inferer=model_inferer,args=args,post_label=post_label,post_pred=post_pred,)
            
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "dice",dice,"hd",hd,"iou",iou,"acc",acc,"pre",pre,"spe",spe,"recall",recall,"fs",fs,
                    "time {:.2f}s".format(time.time() - epoch_time),)
                
                open('Log.txt', 'a').write("Final validation {}/{} dice {} hd {} iou {} acc {} pre {} spe {} recall {} fs {} time {:.2f}s \n".format(epoch, args.max_epochs - 1,dice,hd,iou,acc,pre,spe,recall,fs,time.time() - epoch_time))
                
                if writer is not None:
                    writer.add_scalar("val_acc", dice, epoch)
                if dice > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, dice))
                    val_acc_max = dice
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return val_acc_max
