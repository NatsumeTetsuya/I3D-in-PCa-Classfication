# -*- coding = utf-8 -*-
# @Time : 2023/1/3 23:44
# @Author : Tetsuya Chen
# @File : config_timetrans.py
# @software : PyCharm
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=2, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# 数据集设置
dataset_type = 'RawframeDataset'  # 训练，验证，测试的数据集类型
data_root = '/home/chenpeizhe/Dataset/dongyang_plus/clips_preprocess256/'  # 训练集的根目录
data_root_val = '/home/chenpeizhe/Dataset/dongyang_plus/clips_preprocess256/'  # 验证集，测试集的根目录
ann_file_train = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds/train_0_annotation.txt"  # 训练集的标注文件
ann_file_val = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds/valid_0_annotation.txt"  # 验证集的标注文件
ann_file_test = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds/test_annotation.txt"  # 测试集的标注文件

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=4),
    dict(type='RawFrameDecode'),
    #dict(type='RandomRescale', scale_range=(256, 320)),
    #dict(type='RandomCrop', size=224),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type = 'Resize', scale = (224, 224), keep_ratio = False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale = (224, 224), keep_ratio = False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale = (224, 224), keep_ratio = False),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=4,
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(  # 训练期间做验证的设置
    interval=1,  # 执行验证的间隔
    metrics=['mean_class_accuracy'],  # 验证方法 'top_k_accuracy',mean_class_accuracy
    save_best='mean_class_accuracy')  # 设置 `top_k_accuracy` 作为指示器，用于存储最好的模型权重文件

# optimizer
optimizer = dict(
    type='Adam',
    lr=0.0008,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    )  # this lr is used for 8 gpus  nesterov=True
optimizer_config = dict()#grad_clip=dict(max_norm=40, norm_type=2)
# learning policy
lr_config = dict(  # 用于注册学习率调整钩子的设置
    policy='CosineAnnealing',  # 调整器策略, 支持 CosineAnnealing，Cyclic等方法。更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    min_lr_ratio=0.05,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1
)  # 学习率衰减步长

total_epochs = 200  # 训练模型的总周期数
checkpoint_config = dict(  # 模型权重钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
    interval=5)  # 模型权重文件保存间隔


log_config = dict(  # 注册日志钩子的设置
    interval=20,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        #dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])

# 运行设置
dist_params = dict(backend='nccl')  # 建立分布式训练的设置，其中端口号也可以设置
log_level = 'INFO'  # 日志等级
work_dir = './work_dirs/time_trans/8_32_1/Adam_strong'  # 记录当前实验日志和模型权重文件的文件夹
load_from = "/home/chenpeizhe/project/guokeda/dongyang/classfication/MMaction/pretrain/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth"# 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
resume_from = None#"/home/chenpeizhe/project/guokeda/dongyang/classfication/MMaction/work_dirs/i3d_nonlocal/8_2_1/Adam/latest.pth"  # 加载给定路径的模型权重文件作为断点续连的模型, 训练将从该时间点保存的周期点继续进行
workflow = [('train', 1)]  # runner 的执行流. [('train', 1)] 代表只有一个执行流，并且这个名为 train 的执行流只执行一次

omnisource = True  #是否使用到omnisource训练
gpu_ids = [0]
seed = 0
