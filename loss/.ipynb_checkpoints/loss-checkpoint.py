import torch
import numpy as np

def eval_metric(y_pred, y):
    y=y.unsqueeze(0)
    y_pred=y_pred.unsqueeze(0)
    y = y.float()  ##[1,2,141,141,67]
#     print('y'),print(y.shape)
    y_pred = y_pred.float()  #[1,2,141,141,67]
#     print('y_pred'),print(y_pred.shape)

    # get confusion matrix related metric
    batch_size, n_class = y_pred.shape[:2]  #[1,2]
    # convert to [BNS], where S is the number of pixels for one sample.
    # As for classification tasks, S equals to 1.
    y_pred = y_pred.view(batch_size, n_class, -1) #[1,2,1332027]
#     print('y_pred'),print(y_pred.shape)
    y = y.view(batch_size, n_class, -1) #[1,2,1332027]
#     print('y'),print(y.shape)
    tp = ((y_pred + y) == 2).float()
#     print('tp'),print(tp)
    tn = ((y_pred + y) == 0).float()
#     print('tn'),print(tn)

    tp = tp.sum(dim=[2])
#     print('tp'),print(tp)
    tn = tn.sum(dim=[2])
#     print('tn'),print(tn)
    p = y.sum(dim=[2])
#     print('p'),print(p)
    n = y.shape[-1] - p
#     print('n'),print(n)
    fn = p - tp
#     print('fn'),print(fn)
    fp = n - tn
#     print('fp'),print(fp)

    
    dice0=(2*tp[0][0])/(2*tp[0][0]+fn[0][0]+fp[0][0])
    dice1=(2*tp[0][1])/(2*tp[0][1]+fn[0][1]+fp[0][1])
    dice=(dice0+dice1)/2 #0.3954
    
    #acc_2=(np.mean(tp)+np.mean(tn))/(np.mean(tp)+np.mean(tn)+np.mean(fp)+np.mean(fn)) #0.63365
    acc0=(tp[0][0]+tn[0][0])/(tp[0][0]+tn[0][0]+fp[0][0]+fn[0][0])
    acc1=(tp[0][1]+tn[0][1])/(tp[0][1]+tn[0][1]+fp[0][1]+fn[0][1])
    acc_1=(acc0+acc1)/2 #0.63365
    
    iou0=tp[0][0]/(tp[0][0]+fp[0][0]+fn[0][0])
    iou1=tp[0][1]/(tp[0][1]+fp[0][1]+fn[0][1])
    iou=(iou0+iou1)/2 #0.3203
    
#     spe0=tn[0][0]/(tn[0][0]+fp[0][0])
    spe=tn[0][1]/(tn[0][1]+fp[0][1])
#     spe_1=(spe0+spe1)/2 #0.4052
#     spe_2=np.mean(tn)/(np.mean(tn)+np.mean(fp)) #0.63365    
    
    pre0=tp[0][0]/(tp[0][0]+fp[0][0])
    pre1=tp[0][1]/(tp[0][1]+fp[0][1])
    pre_1=(pre0+pre1)/2 #0.4929
#     pre_2=np.mean(tp)/(np.mean(tp)+np.mean(fp)) #0.63365
    
    recall0=tp[0][0]/(tp[0][0]+fn[0][0])
    recall1=tp[0][1]/(tp[0][1]+fn[0][1])
    recall_1=(recall0+recall1)/2 #0.4052
#     recall_2=np.mean(tp)/(np.mean(tp)+np.mean(fn)) #0.63365
    
    
#     f1_score=2*recall*pre/(pre+recall)
    f1_score1=2*recall_1*pre_1/(pre_1+recall_1) #0.4448  这个选择的与monai中不同，我觉得这个更适合
#     f1_score2=2*np.mean(tp)/(2*np.mean(tp)+np.mean(fn)+np.mean(fp)) #0.63365
#     f1_score3=2*recall_2*pre_2/(pre_2+recall_2) #0.63365
    return np.mean(dice),np.mean(acc_1),np.mean(iou),np.mean(spe),np.mean(pre_1),np.mean(recall_1),np.mean(f1_score1)
#     return f1_score