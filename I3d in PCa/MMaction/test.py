# -*- coding = utf-8 -*-
# @Time : 2022/11/5 23:32
# @Author : Tetsuya Chen
# @File : test.py
# @software : PyCharm
from mmaction.datasets import build_dataset   # 调用build_dataset构建数据集
from mmaction.models import build_model       # 调用build_model构建模型
from mmaction.apis import train_model         # 调用train_model训练模型，传入配置文件，数据，模型
from mmaction.apis import single_gpu_test
from mmaction.datasets import build_dataloader
from mmcv.parallel import MMDataParallel
from mmcv import Config
from mmaction.apis import inference_recognizer, init_recognizer
import torch
import numpy as np

cfg = Config.fromfile('./config.py')

# 构建测试数据集
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,    # batchsize设置为1
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

checkpoint = "model_pth/epoch_5.pth"
model = init_recognizer(cfg, checkpoint, device='cuda:0')
model = MMDataParallel(model, device_ids=[0])    # 初始化模型
outputs = single_gpu_test(model, data_loader)    # 得到所有模型的分类输出
print(np.array(outputs))
np.save('out/test_out.npy',np.array(outputs))

# 在测试集上评价训练完成的识别模型
eval_config = cfg.evaluation
eval_config.pop('interval')
eval_res = dataset.evaluate(outputs, **eval_config)   # 比较输出值与真实值，计算准确率
for name, val in eval_res.items():
    print(f'{name}: {val:.04f}')
