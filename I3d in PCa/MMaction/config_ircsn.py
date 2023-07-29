# -*- coding = utf-8 -*-
# @Time : 2023/1/4 13:18
# @Author : Tetsuya Chen
# @File : config_ircsn.py
# @software : PyCharm
# model settings
model = dict(
    # type='Recognizer3D',
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'  # noqa: E501
    ),
    # cls_head = dict(
    #     type='I3DHead',
    #     num_classes=2,
    #     in_channels=2048,  # 通道维
    #     spatial_type='avg',
    #     dropout_ratio=0.5,
    #     init_std=0.01),
    #            train_cfg = dict(
    #     blending=dict(type='MixupBlending', num_classes=2, alpha=.2)),
    #                        test_cfg = dict(average_clips='prob')
)
# dataset settings
# 数据集设置
dataset_type = 'RawframeDataset'  # 训练，验证，测试的数据集类型
data_root = '/home/chenpeizhe/Dataset/dongyang_plus/clips_preprocess256/'  # 训练集的根目录
data_root_val = '/home/chenpeizhe/Dataset/dongyang_plus/clips_preprocess256/'  # 验证集，测试集的根目录
ann_file_train = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds/train_0_annotation.txt"  # 训练集的标注文件
ann_file_val = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds/valid_0_annotation.txt"  # 验证集的标注文件
ann_file_test = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds/test_annotation.txt"  # 测试集的标注文件
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224,224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=3,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(  # 训练期间做验证的设置
    interval=5,  # 执行验证的间隔
    metrics=['mean_class_accuracy'],  # 验证方法 'top_k_accuracy',mean_class_accuracy
    save_best='mean_class_accuracy')  # 设置 `top_k_accuracy` 作为指示器，用于存储最好的模型权重文件

# optimizer
optimizer = dict(
    type='Adam', lr=0.0001,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict()
# learning policy

lr_config = dict(  # 用于注册学习率调整钩子的设置
    policy='CosineAnnealing',  # 调整器策略, 支持 CosineAnnealing，Cyclic等方法。更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    min_lr_ratio=0.05,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1
)  # 学习率衰减步长
total_epochs = 200

log_config = dict(  # 注册日志钩子的设置
    interval=20,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        #dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])

# 运行设置
dist_params = dict(backend='nccl')  # 建立分布式训练的设置，其中端口号也可以设置
log_level = 'INFO'  # 日志等级
work_dir = './work_dirs/ircsn/32_2_1/Adam_strong_mixup'  # 记录当前实验日志和模型权重文件的文件夹
load_from = "/home/chenpeizhe/project/guokeda/dongyang/classfication/MMaction/pretrain/vmz_ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-e63ee1bd.pth"# 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
resume_from = None#"/home/chenpeizhe/project/guokeda/dongyang/classfication/MMaction/work_dirs/i3d_nonlocal/8_2_1/Adam/latest.pth"  # 加载给定路径的模型权重文件作为断点续连的模型, 训练将从该时间点保存的周期点继续进行
workflow = [('train', 1)]  # runner 的执行流. [('train', 1)] 代表只有一个执行流，并且这个名为 train 的执行流只执行一次

omnisource = True  #是否使用到omnisource训练
gpu_ids = [0]
seed = 0
find_unused_parameters = True