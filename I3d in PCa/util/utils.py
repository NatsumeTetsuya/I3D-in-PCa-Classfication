# -*- coding = utf-8 -*-
# @Time : 2022/3/18 16:31
# @Author : Tetsuya Chen
# @File : utils.py
# @software : PyCharm
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.modules.module import Module
from torch.utils.data.dataset import Dataset


def split_attention_res(model):
    all_params = model.parameters()
    attention_params = []
    res_params = []
    for name, param in model.named_params:
        if any(name.startwith('attetion') or name=='fc'):
            attention_params += param
        else:
            res_params += param
        # params_id = map(id, attention_params)
        # # 取回剩余分特殊处置参数的id
        # other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    return attention_params, res_params

def vote(df_folder):
    precision_ = []
    pred_ = []
    prediction = []
    for i in df_folder:
        precision = i['mean_label']
        pred = i['mean_conf']
        label = i['properties']
        precision_.append(np.array(precision))
        pred_.append(pred)
    precision_ = np.array(precision_)
    pred_ = np.array(pred_)
    label = np.array(label)

    for j in range(len(precision_[1])):
        max_label = np.argmax(np.bincount(precision_[:, j]))
        prediction.append(max_label)
    acc_total = sum(prediction == label) / len(label)
    preds = pred_.sum(axis=0)/pred_.shape[0]
    predict = np.where(preds>0.5, 1 ,0)
    i['vote_conf'] = preds
    i['vote'] = predict
    clo_n = ['num', 'properties', 'path', 'vote_conf', 'vote']
    df = pd.DataFrame(i, columns=clo_n)
    acc_total_ = sum(predict == label) / len(label)
    print(acc_total)
    print(acc_total_)
    return df

def df_mean_pred(df, k_fold):
    fc_ = []
    for i in range(k_fold):
        fc_.append(list(df[f'{i}pred1']))
    fc_ = np.array(fc_)
    fc_ = np.mean(fc_, axis=0)
    return fc_

class CutMixCrossEntropyLoss(Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)




if __name__ == '__main__':
    '''
    csv_dir1 = "/home/chenpeizhe/project/guokeda/ptc/exp/fn_nofn_final_/resnest20d/21/x[1]_s[0]_bs64_LR0.0004_wd0.05_EPOCH100_mixFalse_poolingconcat_fc2layers_feature2048_supernewaug_earlystop_LOSS[1.5,1]/test ratio[1] shift[0]_last.csv"
    csv_dir2 = "/home/chenpeizhe/project/guokeda/ptc/exp/fn_nofn_final_/resnet18/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.05_EPOCH200_mixFalse_changefcTrue_drop0.1_FLOODFalse_midaug_earlystop/test ratio[1] shift[0]_bestvalloss.csv"
    csv_dir3 = "/home/chenpeizhe/project/guokeda/ptc/exp/fn_nofn_final_/inception_v4/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.05_EPOCH300_mixFalse_cutmix0.5_poolingconcat_fc2layers_feature1536_drop0.1_superaug_/test ratio[1] shift[0]_bestvalloss.csv"

    csv_dir1 = "/home/chenpeizhe/project/guokeda/ptc/exp/fn_nofn_final_/resnest20d/21/x[1]_s[0]_bs64_LR0.0004_wd0.05_EPOCH100_mixFalse_poolingconcat_fc2layers_feature2048_supernewaug_earlystop_LOSS[1.5,1]/testExternal/last.csv"
    csv_dir2 = "/home/chenpeizhe/project/guokeda/ptc/exp/fn_nofn_final_/resnet18/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.05_EPOCH200_mixFalse_changefcTrue_drop0.1_FLOODFalse_midaug_earlystop/testExternal/bestvalloss.csv"
    csv_dir3 = "/home/chenpeizhe/project/guokeda/ptc/exp/fn_nofn_final_/inception_v4/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.05_EPOCH300_mixFalse_cutmix0.5_poolingconcat_fc2layers_feature1536_drop0.1_superaug_/testExternal/bestvalloss.csv"
    '''
    csv_dir1 = "/home/chenpeizhe/project/guokeda/ptc/exp_fn/resnet18/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.5_EPOCH200_mixFalse_cutmix0.5_poolingavgpool_fcmlp_feature512_drop0.5_superaug_soft/test ratio[1] shift[0]_bestvalloss.csv"
    csv_dir2 = "/home/chenpeizhe/project/guokeda/ptc/exp_fn/inception_v4/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.5_EPOCH200_mixFalse_cutmix0.5_poolingconcat_fc2layers_feature1536_drop0.5_superaug_soft/test ratio[1] shift[0]_bestvalloss.csv"
    csv_dir3 = "/home/chenpeizhe/project/guokeda/ptc/exp_fn/vgg/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.5_EPOCH200_mixFalse_cutmix0.5_poolingavgpool_fcmlp_feature512_drop0.5_superaug_soft/test ratio[1] shift[0]_bestvalloss.csv"
    csv_dir4 = "/home/chenpeizhe/project/guokeda/ptc/exp_fn/resnet50/pretrain/x[1]_s[0]_bs64_LR0.0002_wd0.5_EPOCH200_mixFalse_cutmix0.5_poolingavgpool_fcmlp_feature2048_drop0.5_superaug_soft/test ratio[1] shift[0]_bestvalloss.csv"

    df1 = pd.read_csv(csv_dir1)
    df2 = pd.read_csv(csv_dir2)
    df3 = pd.read_csv(csv_dir3)
    df4 = pd.read_csv(csv_dir4)

    # df1['pred_all'] = df_mean_pred(df1,8)
    # df2['pred_all'] = df_mean_pred(df2, 8)
    # df3['pred_all'] = df_mean_pred(df3, 8)
    # df4['pred_all'] = df_mean_pred(df4, 8)
    # df1.to_csv(csv_dir1, index=0)
    # df2.to_csv(csv_dir2, index=0)
    # df3.to_csv(csv_dir3, index=0)
    # df4.to_csv(csv_dir4, index=0)

    vote_result = vote([df1,df2,df3,df4])

    vote_dir = '/home/chenpeizhe/project/guokeda/ptc/exp_fn/vote/resnet18+inceptionv4+vgg11bn+resnet50_mean/'

    os.makedirs(vote_dir, exist_ok=True)
    vote_result.to_csv(vote_dir+'vote.csv', index=0)


