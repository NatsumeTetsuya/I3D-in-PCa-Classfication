# -*- coding = utf-8 -*-
# @Time : 2023/5/12 12:52
# @Author : Tetsuya Chen
# @File : heatmaps.py
# @software : PyCharm
# -*- coding = utf-8 -*-
# @Time : 2023/5/12 12:48
# @Author : Tetsuya Chen
# @File : heatmaps1.py
# @software : PyCharm
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from mmaction.models import build_backbone
from mmaction.apis import inference_recognizer, init_recognizer
from mmcv.parallel import MMDataParallel
from mmcv import Config
from mmaction.datasets import build_dataset, build_dataloader


def hook(module, input, output):
    global features
    features = output


def get_heatmap(frame, model, features, class_names):
    print(frame)
    with torch.no_grad():
        output = model(frame)

    # 提取中间层输出
    features = features.squeeze(0).cpu().numpy()
    print(features.shape)

    # 将中间层输出与输入图像序列相乘，得到热力图
    heatmap = np.zeros((features.shape[1], features.shape[2]), dtype=np.float32)
    for i in range(features.shape[0]):
        heatmap += features[i] * cv2.resize(frame[0, i].cpu().numpy().transpose(1, 2, 0),
                                            (features.shape[2], features.shape[1]))
    # 调整图像大小和预处理
    # back_bone = model.backbone
    # # 获取卷积层输出
    # input = back_bone.layer3(back_bone.layer2(back_bone.layer1((back_bone.pool2(back_bone.maxpool(back_bone.conv1(frame)))))))
    # conv_layer = back_bone.layer4
    #
    # fea_out = conv_layer(input)
    # activation = torch.mean(fea_out, dim=2).squeeze()
    # # 获取预测结果
    # output = model.cls_head(fea_out)
    #
    # prediction = torch.argmax(output, dim=1).item()
    # idx = torch.max()
    # prediction_label = class_names[prediction]
    # #print(list(back_bone.layer4[-1].conv3.conv.parameters())[0][0])
    # # 提取热图
    # weights = torch.Tensor(list(back_bone.layer4[-1].conv3.conv.parameters())[0].cpu()).cuda()
    # print(weights.shape)
    # #weights = weights.unsqueeze(0).unsqueeze(0).expand(1,7,weights.shape[0])
    #
    # cam = torch.matmul(activation.permute(1, 2, 0),weights)
    # #print(cam)
    # #torch.sum(, dim=0)
    #
    #
    # cam = np.maximum(cam.detach().cpu().numpy(), 0)
    # #print(cam)
    # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    #
    # cam = cv2.resize(cam, (frame.shape[-1], frame.shape[-2]))
    # frame_ = frame.permute(0,2,1,3,4)
    #
    # # 可视化热图
    # raw_img = frame_[0,0].permute(1,2,0).cpu().numpy().copy() * std + mean
    # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    #
    # superimposed_img = cv2.addWeighted(frame_[0,0].permute(1,2,0).cpu().numpy(), 0.75, heatmap, 0.25, 0, raw_img, dtype=cv2.CV_32S)
    # superimposed_img = torch.from_numpy(superimposed_img).permute(2,0,1)
    #
    # return heatmap, prediction_label#superimposed_img

if __name__ == '__main__':
    class_names = ['benign', 'malignant']
    cfg = Config.fromfile('./config.py')
    checkpoint = "/home/chenpeizhe/project/guokeda/dongyang/classfication/MMaction/results/epoch_5.pth"
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    data_loader = build_dataloader(
            dataset,
            videos_per_gpu=1,    # batchsize设置为1
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
    model = init_recognizer(cfg, checkpoint, device='cuda:0')
    model.eval()
    features = None

    def hook(module, input, output):
        global features
        features = output

    model.backbone.layer3[-1].register_forward_hook(hook)

    # 将输入的图像序列传递给模型
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data['imgs'] = data['imgs'].cuda()
            output = model(return_loss=False, **data)
            video_data = data['imgs'].cuda()[0][0]

            # 提取中间层输出
            features = features.squeeze(0).cpu().numpy()
            #features = np.mean(features, axis = 1)

            # 将中间层输出与输入图像序列相乘，得到热力图
            heatmap = np.zeros((features.shape[1], features.shape[2]), dtype=np.float32)
            for j in range(features.shape[0]):
                k = features[j] * cv2.resize(video_data[j].cpu().numpy().transpose(1, 2, 0),
                                                    (features.shape[2], features.shape[3]))
                heatmap +=1
                # 可视化热力图
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            result = cv2.resize(cv2.cvtColor(video_data[ 0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR),
                                (heatmap.shape[1], heatmap.shape[0]))
            result = cv2.addWeighted(result, 0.5, heatmap, 0.5, 0)

            cv2.imwrite(f'heatmap_{i + 1}.jpg', result)
    # for i, frame in enumerate(data_loader):
    #
    #     heatmap_frame, prediction = get_heatmap(frame, model, features,class_names)
    #     # 显示带有热图的帧和预测结果
    #     #plt.imshow(heatmap_frame.permute(1, 2, 0))
    #     plt.imshow(heatmap_frame)
    #     plt.title('Prediction: %s' % prediction)
    #     plt.show()
    #     # plt.imshow(heatmap_frame[:, :, ::-1])
    #     # plt.title('Prediction: %s' % prediction)
    #     # plt.show()
    #
    #     # # 显示带有热图的原始图像
    #     # plt.imshow(superimposed_img[:, :, ::-1])
    #     # plt.title('Prediction: %s' % prediction)
    #     # plt.show()