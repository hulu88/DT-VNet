import torch
import numpy as np

def eval_metric(y_pred, y):
    y=y.unsqueeze(0)
    y_pred=y_pred.unsqueeze(0)
    y = y.float()  
    y_pred = y_pred.float()  

    batch_size, n_class = y_pred.shape[:2]  
    y_pred = y_pred.view(batch_size, n_class, -1) 
    y = y.view(batch_size, n_class, -1) 
    tp = ((y_pred + y) == 2).float()
    tn = ((y_pred + y) == 0).float()

    tp = tp.sum(dim=[2])
    tn = tn.sum(dim=[2])
    p = y.sum(dim=[2])
    n = y.shape[-1] - p
    fn = p - tp
    fp = n - tn

    
    dice0=(2*tp[0][0])/(2*tp[0][0]+fn[0][0]+fp[0][0])
    dice1=(2*tp[0][1])/(2*tp[0][1]+fn[0][1]+fp[0][1])
    dice=(dice0+dice1)/2 
    
    acc0=(tp[0][0]+tn[0][0])/(tp[0][0]+tn[0][0]+fp[0][0]+fn[0][0])
    acc1=(tp[0][1]+tn[0][1])/(tp[0][1]+tn[0][1]+fp[0][1]+fn[0][1])
    acc_1=(acc0+acc1)/2 
    
    iou0=tp[0][0]/(tp[0][0]+fp[0][0]+fn[0][0])
    iou1=tp[0][1]/(tp[0][1]+fp[0][1]+fn[0][1])
    iou=(iou0+iou1)/2 
    
    spe=tn[0][1]/(tn[0][1]+fp[0][1])  
    
    pre0=tp[0][0]/(tp[0][0]+fp[0][0])
    pre1=tp[0][1]/(tp[0][1]+fp[0][1])
    pre_1=(pre0+pre1)/2 
    
    recall0=tp[0][0]/(tp[0][0]+fn[0][0])
    recall1=tp[0][1]/(tp[0][1]+fn[0][1])
    recall_1=(recall0+recall1)/2 
       
    f1_score1=2*recall_1*pre_1/(pre_1+recall_1) 
    return np.mean(dice),np.mean(acc_1),np.mean(iou),np.mean(spe),np.mean(pre_1),np.mean(recall_1),np.mean(f1_score1)