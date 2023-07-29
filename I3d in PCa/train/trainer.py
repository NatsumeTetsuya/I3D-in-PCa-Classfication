# -*- coding = utf-8 -*-
# @Time : 2022/1/17 16:46
# @Author : Tetsuya Chen
# @File : trainer.py
# @software : PyCharm

import copy
import os
import time

import numpy as np
import pandas as pd
import scipy.special
import torch
import ttach as tta
# from pytorchtools import EarlyStopping
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch_utils as tu
from tqdm.autonotebook import tqdm
from sklearn.metrics import *
from scipy.special import softmax
import warnings

from dataset.dataset import TrainDataset, TestDataset, stack_train_dataset, stack_test_dataset
from util import ImageModel,split_attention_res, CutMix
from train.config import CONFIG
from data_analyze.data_analyze import Find_Optimal_Cutoff
from data_process.split_trantest import sample_df
import data_process.utils as utils
sigmoid = lambda x: scipy.special.expit(x)


warnings.filterwarnings("ignore")


def train_model(CONFIG, train_loader, trainset, scaler, optimizer, scheduler, model_conv, verbose=True):
    model_conv.train()
    avg_loss = 0.
    sum_acc = 0
    sum_train = 0
    optimizer.zero_grad()

    if verbose:
        bar = tqdm(total=len(train_loader))
    mixup_fn = tu.Mixup(prob=CONFIG.MIXUP, switch_prob=0.3, onehot=True, label_smoothing=0.05, num_classes=CONFIG.CLASSES)

    for idx, (imgs, labels) in enumerate(train_loader):
        if CONFIG.sigmoid:
            imgs_train, labels_train = imgs.float().cuda(), labels.float().cuda()
        else:
            imgs_train, labels_train = imgs.float().cuda(), labels.cuda()

        with torch.cuda.amp.autocast():
            if CONFIG.MIXUP:
                imgs_train, labels_train = mixup_fn(imgs_train, labels_train)
            if CONFIG.resnest50d:
                if CONFIG.tu_model:
                    output_train, _ = model_conv(imgs_train)
                else:
                    output_train = model_conv(imgs_train)
            if CONFIG.random_frame:
                random_frame = np.random.randint(0, imgs_train.size(1))
                output_train = model_conv({'imgs':imgs_train, 'frame':random_frame})

            else:
                output_train = model_conv(imgs_train)

            criterion = CONFIG.criterion.cuda()
            if CONFIG.sigmoid:
                loss = criterion(output_train.squeeze(1), labels_train)
            else:
                loss = criterion(output_train, labels_train)
            if not CONFIG.MIXUP and not CONFIG.CUTMIX:
                sum_acc += sum((output_train.argmax(dim=-1) == labels_train).float())
            sum_train += len(labels_train)
        if CONFIG.FLOOD:
            flood = (loss - CONFIG.FLOOD).abs() + CONFIG.FLOOD
            scaler.scale(flood).backward()
        else:
            scaler.scale(loss).backward()
        if ((idx+1)%CONFIG.ACCUMULATE==0): # Gradient Accumulate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        avg_loss += loss.item() / len(train_loader)
        if verbose:
            bar.update(1)
    if verbose:
        bar.close()
    metric_train = 0
    if not CONFIG.MIXUP and not CONFIG.CUTMIX and not CONFIG.sigmoid:
        metric_train = (sum_acc / sum_train).item()

    return avg_loss, metric_train


def test_model(CONFIG, model_conv, valset, val_loader, layer_output=False, features_out=False):
    avg_val_loss = 0.
    model_conv.eval()
    y_true_val = np.zeros(len(valset))
    if CONFIG.sigmoid:
        y_pred_val = np.zeros((len(valset)))
    else:
        y_pred_val = np.zeros((len(valset), CONFIG.CLASSES))
    #print(model_conv)
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            if CONFIG.sigmoid:
                imgs_valid, labels_valid = imgs.float().cuda(), labels.float().cuda()
            else:
                imgs_valid, labels_valid = imgs.float().cuda(), labels.cuda()
                # print(imgs_valid)

            if CONFIG.resnest50d:
                if CONFIG.tu_model:
                    output_test, _ = model_conv(imgs_valid)

                else:

                    output_test = model_conv(imgs_valid)
            if CONFIG.random_frame:
                # random_frame = np.random.randint(0, imgs_valid.size(1))
                # output_test = model_conv({'imgs': imgs_valid, 'frame': random_frame})

                output_test = model_conv({'imgs':imgs_valid, 'frame':imgs_valid.size(1)//2})

            else:
                output_test = model_conv(imgs_valid)




            criterion_test = CONFIG.criterion.cuda()


            if CONFIG.sigmoid:
                avg_val_loss += (criterion_test((output_test).squeeze(1), labels_valid).item() / len(val_loader))
                a = labels_valid.detach().cpu().numpy().astype(np.int)
                b = output_test.squeeze(1).detach().cpu().numpy()

            else:
                avg_val_loss += (criterion_test(output_test, labels_valid).item() / len(val_loader))
                a = labels_valid.detach().cpu().numpy().astype(np.int)
                b = softmax(output_test.detach().cpu().numpy(), axis=1)


            y_true_val[idx * CONFIG.BATCHSIZE:idx * CONFIG.BATCHSIZE + b.shape[0]] = a
            y_pred_val[idx * CONFIG.BATCHSIZE:idx * CONFIG.BATCHSIZE + b.shape[0]] = b

            if layer_output:
                feature_output1 = model_conv.featuremap1.cpu().numpy()
                if type(features_out) == bool:
                    features_out = feature_output1
                else:
                    features_out = np.concatenate((features_out, feature_output1), 0)
    if CONFIG.sigmoid:
        fpr, tpr, thresholds = roc_curve(y_true_val, y_pred_val)
        optimal_threshold, optimal_point = Find_Optimal_Cutoff(tpr, fpr, thresholds)
        y_pred_val = np.select([y_pred_val >= optimal_threshold, y_pred_val < optimal_threshold], [1.0, 0])
        metric_val = sum(y_pred_val == y_true_val) / len(y_true_val)

    else:
        metric_val = sum(np.argmax(y_pred_val, axis=1) == y_true_val) / len(y_true_val)

    if layer_output:
        return avg_val_loss, metric_val, y_pred_val, y_true_val, features_out
    else:
        return avg_val_loss, metric_val, y_pred_val, y_true_val

def train_once(CONFIG, log, fold, train_loader, trainset, scaler, optimizer, scheduler, model_conv,  valset, val_loader):
    best_avg_loss = 100.0
    best_acc = 0.0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    count = 0
    # for parameter in model_conv.parameters():
    #     print(parameter)
    #     log.write(f'{parameter} \n')

    avg_val_loss, avg_val_acc, _, _ = test_model(CONFIG, model_conv, valset, val_loader)
    print('pretrain val loss %.4f precision %.4f'%(avg_val_loss, avg_val_acc))
    log.write(f'pertrain val loss {avg_val_loss}, precision {avg_val_acc}')

    ### training
    for epoch in range(CONFIG.EPOCH):

        print('lr:', optimizer.param_groups[0]['lr'])
        np.random.seed(CONFIG.SEED+fold*99+epoch*999)
        start_time = time.time()
        avg_loss, metric_train = train_model(CONFIG, train_loader, trainset, scaler, optimizer, scheduler, model_conv)
        avg_val_loss, avg_val_acc, _, _ = test_model(CONFIG, model_conv, valset, val_loader)
        #scheduler.step(avg_val_loss)
        #scheduler.step()
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t train_loss={:.4f} \t train_precision={:.4f} \t val_loss={:.4f} \t val_precision={:.4f} \t time={:.2f}s'.format(
            epoch + 1, CONFIG.EPOCH, avg_loss, metric_train, avg_val_loss, avg_val_acc, elapsed_time))
        log.write('Epoch {}/{} \t train_loss={:.4f} \t train_precision={:.4f} \t val_loss={:.4f} \t val_precision={:.4f} \t time={:.2f}s \n'.format(
            epoch + 1, CONFIG.EPOCH, avg_loss, metric_train, avg_val_loss, avg_val_acc, elapsed_time))
        if count == CONFIG.COUNT:
            torch.save(model_conv.state_dict(),
                       CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP) + '/model-last' + str(fold) + '.pth')
            print('early stopping!')
            break

        if avg_val_loss < best_avg_loss:
            count = 0
            best_avg_loss = avg_val_loss
            torch.save(model_conv.state_dict(),
                       CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP) + '/model-bestvalloss' + str(fold) + '.pth')
            print('model saved!')

        else:
            count += 1

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            #muti-gpu
            #torch.save(model_conv.module.state_dict(), CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP) + '/model-best' + str(fold) + '.pth')
            #single-gpu
            torch.save(model_conv.state_dict(),
                       CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP) + '/model-best' + str(fold) + '.pth')
            print('model saved!')


        train_loss.append(avg_loss)
        train_acc.append(metric_train)
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

        print('=================================')

    print('best loss:', best_avg_loss)
    print('best precision:', best_acc)
    log.write(f'best loss:{best_avg_loss}     best precision:{best_acc}\n')
    utils.plt_training(train_loss, train_acc, val_loss, val_acc, fold, CONFIG)
    # muti-gpu
    # torch.save(model_conv.module.state_dict(),
    #            CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP) + '/model-last' + str(fold) + '.pth')
    #single-gpu
    torch.save(model_conv.state_dict(),
               CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP) + '/model-last' + str(fold) + '.pth')

    print('model_last saved!')
    return best_avg_loss, best_acc

def train_all(CONFIG):

    if not os.path.exists(CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP)):
        os.makedirs(CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP), exist_ok=True)
    log = open(CONFIG.MODEL_SAVE_DIR + str(CONFIG.EXP) + '/log.txt', 'w')
    cv_losses = []
    cv_metrics = []



    for fold in range(CONFIG.FOLD):
        print('\n ********** Fold %d **********\n' % fold)
        log.write(('\n ********** Fold %d **********\n' % fold))


        ####################### Model ########################

        # model_conv = tu.ImageModel(name='resnest50d', pretrained=True, classes=CONFIG.CLASSES)
        # model_conv = copy.deepcopy(model)
        model = ImageModel(copy.deepcopy(CONFIG.MODEL),
                           pretrain_path=CONFIG.pretrained_model_path,
                           model_backbone=CONFIG.backbone,
                            test_model_path=CONFIG.test_model_path,
                           pooling=CONFIG.pooling,
                           fc=CONFIG.fc,
                           no_load_fc=CONFIG.NO_LOAD_FC,
                           dropout=CONFIG.DROP_OUT,
                           num_feature=CONFIG.num_feature,
                           classes=CONFIG.CLASSES,)
        model_conv = copy.deepcopy(model)
        model_conv.cuda()
        print(model_conv)
        #model_conv = torch.nn.DataParallel(model_conv)
        ###################### Dataset #######################


        if os.path.exists(
                CONFIG.MODEL_SAVE_DIR + '{}/model-best{}.pth'.format(CONFIG.EXP, fold)):
            if CONFIG.CONTINUE_TRAIN and not os.path.exists(CONFIG.MODEL_SAVE_DIR + '{}/model-last{}.pth'.format(CONFIG.EXP, fold)):
                model_conv.load_state_dict(
                    torch.load(CONFIG.MODEL_SAVE_DIR + '{}/model-best{}.pth'.format(CONFIG.EXP, fold)))
                print('continue training fold {}.'.format(fold))
            else:
                print('fold {} has been trained.'.format(fold))
                continue
        if CONFIG.test_model_path:
            model_conv.load_state_dict(torch.load(CONFIG.test_model_path))

        df_train = pd.read_csv(os.path.join(CONFIG.DATA_DIR, f'train_{fold}.csv'))
        df_valid = pd.read_csv(os.path.join(CONFIG.DATA_DIR, f'valid_{fold}.csv'))

        # df_train = sample_df(df_train, 10)
        # df_valid = sample_df(df_valid,5)

        if CONFIG.AUTOAUG:
            # trainset = NoduleDataset_autoaug(df_train)
            # valset = NoduleDataset_test_autoaug(df_valid)
            trainset = TrainDataset(df_train)
            valset = TestDataset(df_valid)
        elif CONFIG.STACK:
            trainset = stack_train_dataset(df_train)
            valset = stack_test_dataset(df_valid)
        else:
            trainset = TrainDataset(df_train)
            valset = TestDataset(df_valid)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG.BATCHSIZE, num_workers=0, shuffle=True,
                                                   drop_last=False)
        if CONFIG.CUTMIX:
            train_loader = torch.utils.data.DataLoader(CutMix(trainset, num_class=2, beta=1.0, prob=CONFIG.CUTMIX, num_mix=2), batch_size=CONFIG.BATCHSIZE, num_workers=0,
                                                       shuffle=True,
                                                       drop_last=False)


        val_loader = torch.utils.data.DataLoader(valset, batch_size=CONFIG.BATCHSIZE, shuffle=False, num_workers=0)



        ###################### Optim ########################
        if CONFIG.attention == True and CONFIG.pretrained_model_path:
            optimizer = AdamW([ {"params": split_attention_res(model_conv)[0], "lr": CONFIG.LR, "weight_decay": CONFIG.WEIGHT_DECAY},
                                {"params": split_attention_res(model_conv)[1], "lr": 0.1*CONFIG.LR}])
            # optimizer = tu.RangerLars(
            #     {"params": split_attention_res(model_conv)[0], "lr": CONFIG.LR, "weight_decay": CONFIG.WEIGHT_DECAY},
            #     {"params": split_attention_res(model_conv)[1], "lr": 0.1 * CONFIG.LR})
            # optimizer = SGD({"params": split_attention_res(model_conv)[0], "lr": CONFIG.LR, "weight_decay": CONFIG.WEIGHT_DECAY},
            #     {"params": split_attention_res(model_conv)[1], "lr": 0.1 * CONFIG.LR})
        else:
            optimizer = AdamW(model_conv.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)

            #optimizer = SGD(model_conv.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)
            #optimizer = tu.RangerLars(model_conv.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)
        T = len(train_loader) // CONFIG.ACCUMULATE * CONFIG.EPOCH  # cycle
        #T = CONFIG.EPOCH
        print(f'scheduler T = {T}')

        scheduler = CosineAnnealingLR(optimizer, T_max=T, eta_min=CONFIG.LR / CONFIG.DECAY_SCALE)

        # scheduler = tu.CosineAnnealingWarmupRestarts(optimizer = optimizer, first_cycle_steps = 10, max_lr = CONFIG.LR,
        #                                              min_lr = CONFIG.LR*0.1, warmup_steps = 5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.5, patience = 5,
        #                                                        min_lr = CONFIG.LR * 0.1, verbose = True)
        val_loss, val_acc = train_once(CONFIG, log = log ,fold = fold, train_loader = train_loader, trainset = trainset,
                                       scaler = CONFIG.SCALER, optimizer = optimizer, scheduler = scheduler,
                                       model_conv = model_conv, valset = valset, val_loader = val_loader)

        cv_losses.append(val_loss)
        cv_metrics.append(val_acc)
        torch.cuda.empty_cache()

    cv_loss = sum(cv_losses) / CONFIG.FOLD
    cv_acc = sum(cv_metrics) / CONFIG.FOLD
    print('CV loss:%.6f  CV precision:%.6f' % (cv_loss, cv_acc))
    log.write('CV loss:%.6f  CV precision:%.6f\n\n' % (cv_loss, cv_acc))

def record_model(CONFIG, model, model_path, saveFileName, df_test, test_dataset, test_loader):
    predictions_folder = []
    acc_folder = []

    for fold in range(CONFIG.FOLD):
        print(torch.load(model_path + '{}.pth'.format(fold)).keys())
        #print(torch.load(model_path + '{}.pth'.format(fold)))
        model.load_state_dict(torch.load(model_path + '{}.pth'.format(fold)))
        #model = torch.load(model_path + '{}.pth'.format(fold))

        predictions = []
        if CONFIG.TTA:
            transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 180]),
                # tta.Scale(scales=[0.9,1,1.1]),
                tta.Multiply(factors=[0.9,1,1.1])


            ])
            model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
        avg_test_loss, metric_test, y_pred_test, y_true_test = test_model(CONFIG, model, test_dataset, test_loader)


        predictions.extend(y_pred_test.argmax(axis=-1).tolist())

        predictions_folder.append(predictions)
        print(f'fold{fold} acc = {metric_test}')
        acc_folder.append(metric_test)
        df_test[f'fc{fold}'] = predictions
        df_test[f'{fold}pred0'] = y_pred_test[:, 0]
        df_test[f'{fold}pred1'] = y_pred_test[:, 1]
        df_test[f'{fold}acc'] = metric_test

    prediction = []
    predictions_folder = np.array(predictions_folder)

    for i in range(len(predictions_folder[0])):
        max_label = np.argmax(np.bincount(predictions_folder[:, i]))
        prediction.append(max_label)

    acc_total = sum(prediction == y_true_test) / len(y_true_test)
    df_test[f'fc_all'] = prediction
    df_test[f'acc_all'] = acc_total

    print(df_test)
    df_test.to_csv(saveFileName, index=0)

def test_all(CONFIG, valid_test = False):

    test_path = os.path.join(CONFIG.DATA_DIR, 'yiwu.csv')
                            # 'test.csv')

    df_test = pd.read_csv(test_path)

    if CONFIG.AUTOAUG:
        #test_dataset = NoduleDataset_test_autoaug(df_test)
        test_dataset = TestDataset(df_test)
    elif CONFIG.STACK:
        test_dataset = stack_test_dataset(df_test)
    else:
        test_dataset = TestDataset(df_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=CONFIG.BATCHSIZE,
        shuffle=False,
        num_workers=8
    )
    print(len(test_loader))
    ## predict
    # model = res_model(176)
    # model = CONFIG.MODEL
    # create model and load weights from checkpoint
    model_conv = ImageModel(copy.deepcopy(CONFIG.MODEL),
                            pretrain_path=CONFIG.pretrained_model_path,
                            model_backbone=CONFIG.backbone,
                            test_model_path=CONFIG.test_model_path,
                            pooling=CONFIG.pooling,
                            fc=CONFIG.fc,
                            no_load_fc=CONFIG.NO_LOAD_FC,
                            dropout=CONFIG.DROP_OUT,
                            num_feature=CONFIG.num_feature,
                            classes=CONFIG.CLASSES,)
    #model_conv = finetune_models(copy.deepcopy(CONFIG.MODEL), CONFIG.CLASSES, CONFIG.DROP_OUT, no_load_fc=True, change_fc=True, freeze=False, model_path=False, pretrain_model=False)
    model = model_conv.cuda()
    print(model)
    #-------------------------------test for model_best----------------------------
    print('---------------------test for model_best-----------------------')
    saveFileName = CONFIG.MODEL_SAVE_DIR + f'{CONFIG.EXP}/yiwu_best.csv'
    model_path = CONFIG.MODEL_SAVE_DIR + '{}/model-best'.format(CONFIG.EXP)
    record_model(CONFIG, model, model_path, saveFileName, df_test, test_dataset, test_loader)

    print('---------------------test for model_bestvalloss-----------------------')
    saveFileName = CONFIG.MODEL_SAVE_DIR + f'{CONFIG.EXP}/yiwu_bestvalloss.csv'
    model_path = CONFIG.MODEL_SAVE_DIR + '{}/model-bestvalloss'.format(CONFIG.EXP)
    record_model(CONFIG, model, model_path, saveFileName, df_test, test_dataset, test_loader)

    print('---------------------test for model_last-----------------------')
    saveFileName = CONFIG.MODEL_SAVE_DIR + f'{CONFIG.EXP}/yiwu_last.csv'
    model_path = CONFIG.MODEL_SAVE_DIR + '{}/model-last'.format(CONFIG.EXP)
    record_model(CONFIG, model, model_path, saveFileName, df_test, test_dataset, test_loader)

    #----------------------------------------------------------------------------
    #-------------------------------valid for model_best--------------------------
    if valid_test:
        print('--------------------------------valid for model_best-------------------------------------------')

        predictions_folder = []
        acc_folder = []
        model_path = CONFIG.MODEL_SAVE_DIR + '{}/model-best'.format(CONFIG.EXP)
        for fold in range(CONFIG.FOLD):

            saveFileName_ = CONFIG.MODEL_SAVE_DIR + f'{CONFIG.EXP}/valid{fold}/best.csv'
            valid_path = os.path.join(CONFIG.DATA_DIR, f'valid_{fold}.csv')
            df_valid = pd.read_csv(valid_path)
            valid_dataset = TestDataset(df_valid)
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=CONFIG.BATCHSIZE,
                shuffle=False,
                num_workers=8
            )

            model.load_state_dict(torch.load(model_path + '{}.pth'.format(fold)))
            predictions = []
            avg_test_loss, metric_test, y_pred_test, y_true_test = test_model(CONFIG, model, valid_dataset, valid_loader)

            predictions.extend(y_pred_test.argmax(axis=-1).tolist())

            predictions_folder.append(predictions)
            print(f'fold{fold} acc = {metric_test}')
            acc_folder.append(metric_test)
            df_valid[f'fc{fold}'] = predictions
            df_valid[f'{fold}pred0'] = y_pred_test[:, 0]
            df_valid[f'{fold}pred1'] = y_pred_test[:, 1]
            df_valid[f'{fold}acc'] = metric_test

            df_valid.to_csv(saveFileName_, index=0)

        print('--------------------------------valid for model_bestvalloss-------------------------------------------')

        predictions_folder = []
        acc_folder = []
        model_path = CONFIG.MODEL_SAVE_DIR + '{}/model-bestvalloss'.format(CONFIG.EXP)
        for fold in range(CONFIG.FOLD):
            saveFileName_ = CONFIG.MODEL_SAVE_DIR + f'{CONFIG.EXP}/valid{fold}/bestvalloss.csv'
            valid_path = os.path.join(CONFIG.DATA_DIR, f'valid_{fold}.csv')
            df_valid = pd.read_csv(valid_path)
            valid_dataset = TestDataset(df_valid)
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=CONFIG.BATCHSIZE,
                shuffle=False,
                num_workers=8
            )

            model.load_state_dict(torch.load(model_path + '{}.pth'.format(fold)))
            predictions = []
            avg_test_loss, metric_test, y_pred_test, y_true_test = test_model(CONFIG, model, valid_dataset,
                                                                              valid_loader)

            predictions.extend(y_pred_test.argmax(axis=-1).tolist())

            predictions_folder.append(predictions)
            print(f'fold{fold} acc = {metric_test}')
            acc_folder.append(metric_test)
            df_valid[f'fc{fold}'] = predictions
            df_valid[f'{fold}pred0'] = y_pred_test[:, 0]
            df_valid[f'{fold}pred1'] = y_pred_test[:, 1]
            df_valid[f'{fold}acc'] = metric_test

            df_valid.to_csv(saveFileName_, index=0)

        print('--------------------------------valid for model_last-------------------------------------------')

        predictions_folder = []
        acc_folder = []
        model_path = CONFIG.MODEL_SAVE_DIR + '{}/model-last'.format(CONFIG.EXP)
        for fold in range(CONFIG.FOLD):
            saveFileName_ = CONFIG.MODEL_SAVE_DIR + f'{CONFIG.EXP}/valid{fold}/last.csv'
            valid_path = os.path.join(CONFIG.DATA_DIR, f'valid_{fold}.csv')
            df_valid = pd.read_csv(valid_path)
            valid_dataset = TestDataset(df_valid)
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=CONFIG.BATCHSIZE,
                shuffle=False,
                num_workers=8
            )

            model.load_state_dict(torch.load(model_path + '{}.pth'.format(fold)))
            predictions = []
            avg_test_loss, metric_test, y_pred_test, y_true_test = test_model(CONFIG, model, valid_dataset,
                                                                              valid_loader)

            predictions.extend(y_pred_test.argmax(axis=-1).tolist())

            predictions_folder.append(predictions)
            print(f'fold{fold} acc = {metric_test}')
            acc_folder.append(metric_test)
            df_valid[f'fc{fold}'] = predictions
            df_valid[f'{fold}pred0'] = y_pred_test[:, 0]
            df_valid[f'{fold}pred1'] = y_pred_test[:, 1]
            df_valid[f'{fold}acc'] = metric_test
            df_valid.to_csv(saveFileName_, index=0)

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.__version__)
    print(torch.cuda.is_available())

    torch.cuda.device(CONFIG.device)

    #train_all(CONFIG)
    test_all(CONFIG, valid_test=True)

