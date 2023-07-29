# -*- coding = utf-8 -*-
# @Time : 2022/11/4 16:34
# @Author : Tetsuya Chen
# @File : train.py
# @software : PyCharm
import os.path as osp

from mmaction.datasets import build_dataset   # 调用build_dataset构建数据集
from mmaction.models import build_model       # 调用build_model构建模型
from mmaction.apis import train_model         # 调用train_model训练模型，传入配置文件，数据，模型

import mmcv
from mmcv import Config

cfg = Config.fromfile('config.py')
# 构建数据集
datasets = [build_dataset(cfg.data.train)]

# 构建动作识别模型（基于预训练模型，把分类数改为2）
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# 创建工作目录并训练模型
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_model(model, datasets, cfg, distributed=False, validate=True)

