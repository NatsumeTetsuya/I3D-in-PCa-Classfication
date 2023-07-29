# -*- coding = utf-8 -*-
# @Time : 2022/11/4 15:51
# @Author : Tetsuya Chen
# @File : config.py
# @software : PyCharm
# 模型设置
model = dict(
    type='Recognizer3D',
    # 膨胀的3D ResNet-50作为主干网络
    backbone=dict(
        non_local_cfg=dict(
                    sub_sample=True,
                    use_scale=False,
                    norm_cfg=dict(type='BN3d', requires_grad=True),
                    mode='embedded_gaussian'),
        type='ResNet3d',
        pretrained2d=True,  # 使用 2D ResNet-50的预训练参数
        pretrained="/home/chenpeizhe/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth",  # 从torchvision中拿取ResNet-50的预训练参数
        depth=50,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,

        # infalte = 1,表示在对应的层使用膨胀策略，将2D卷积变为3D卷积，指定为0就不使用膨胀
        # ResNet-50有4组残差模块每组残差模块中分别有3，4，6，3个残差模块，1和0就表示指定的残差模块是否膨胀。
        inflate = ((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
          zero_init_residual = True),  # 分类时设置为True

        # I3D的分类头
        cls_head = dict(
        type='I3DHead',
        num_classes=2,
        in_channels=2048,  # 通道维
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
        train_cfg=dict(
            blending=dict(type='MixupBlending', num_classes=2, alpha=.2)),
        test_cfg=dict(average_clips='prob')

)


# 数据集设置
dataset_type = 'RawframeDataset'  # 训练，验证，测试的数据集类型
data_root = '/home/chenpeizhe/Dataset/dongyang_plus/clips_preprocess256/'  # 训练集的根目录
data_root_val = '/home/chenpeizhe/Dataset/dongyang_plus/clips_preprocess256/'  # 验证集，测试集的根目录
ann_file_train = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds_re/train_0_annotation_1.txt"  # 训练集的标注文件
ann_file_val = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds_re/valid_0_annotation.txt"  # 验证集的标注文件
#ann_file_test = "/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds_re/test.txt"  # 测试集的标注文件
#ann_file_test = "/home/chenpeizhe/Dataset/dongyang_plus/outsides/label_files/yiwu_annotation.txt"  # 测试集的标注文件
ann_file_test = "/home/chenpeizhe/Dataset/dongyang_plus/cam.txt"

img_norm_cfg = dict(  # 图像正则化参数设置
    mean=[123.675, 116.28, 103.53],  # 图像正则化平均值
    std=[58.395, 57.12, 57.375],  # 图像正则化方差
    to_bgr=False)  # 是否将通道数从 RGB 转为 BGR

train_pipeline = [
    dict(type = 'SampleFrames', clip_len = 32, frame_interval = 2, num_clips = 1),  # 32帧，每隔两帧抽取一帧，覆盖64帧，抽取1个片段
    dict(type = 'RawFrameDecode'),             # 解码，成为32个h×W数组
    #dict(type = 'Resize', scale = (-1, 256)),  # 裁剪
    #dict(type = 'RandomResizedCrop', area_range = (0.6, 1.0), aspect_ratio_range=(1.0, 1.0)),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=224,
    #     scales=(1, 0.875, 0.75),
    #     random_crop=False,
    #     max_wh_scale_gap=1,
    #     num_fixed_crops=13),
    #dict(type='Imgaug', transforms='default'),
    #dict(type = 'Imgaug', transforms=[dict(type= 'Rotate', rotate=(-20, 20))]),
    dict(type = 'Resize', scale = (224, 224), keep_ratio = False),
    dict(type = 'Flip', flip_ratio = 0.5),     # 翻转
    #dict(type='Fuse'),
    dict(type = 'Normalize', **img_norm_cfg),  # 像素归一化
    dict(type = 'FormatShape', input_format = 'NCTHW'),               # 维度排序
    dict(type = 'Collect', keys = ['imgs', 'label'], meta_keys = []),
    dict(type = 'ToTensor', keys = ['imgs', 'label'])                # 转化为totensor格式
]

val_pipeline = [
    dict(type = 'SampleFrames', clip_len = 32, frame_interval = 2, num_clips = 1, test_mode=True),  # 32帧，每隔两帧抽取一帧，覆盖64帧，抽取1个片段
    dict(type = 'RawFrameDecode'),             # 解码，成为32个h×W数组
    #dict(type = 'Resize', scale = (-1, 256)),  # 裁剪
    #dict(type = 'RandomResizedCrop'),
    dict(type = 'Resize', scale = (224, 224), keep_ratio = False),
    #dict(type = 'Flip', flip_ratio = 0.5),     # 翻转
    dict(type = 'Normalize', **img_norm_cfg),  # 像素归一化
    dict(type = 'FormatShape', input_format = 'NCTHW'),               # 维度排序
    dict(type = 'Collect', keys = ['imgs', 'label'], meta_keys = []),
    dict(type = 'ToTensor', keys = ['imgs'])                # 转化为totensor格式
]
test_pipeline = [
    dict(type = 'SampleFrames', clip_len = 32, frame_interval = 2, num_clips = 1, test_mode=True),  # 32帧，每隔两帧抽取一帧，覆盖64帧，抽取1个片段
    dict(type = 'RawFrameDecode'),             # 解码，成为32个h×W数组
    #dict(type = 'Resize', scale = (-1, 256)),  # 裁剪
    #dict(type = 'RandomResizedCrop'),
    dict(type = 'Resize', scale = (224, 224), keep_ratio = False),
    #dict(type = 'Flip', flip_ratio = 0.5),     # 翻转
    dict(type = 'Normalize', **img_norm_cfg),  # 像素归一化
    dict(type = 'FormatShape', input_format = 'NCTHW'),               # 维度排序
    dict(type = 'Collect', keys = ['imgs', 'label'], meta_keys = []),
    dict(type = 'ToTensor', keys = ['imgs'])                # 转化为totensor格式
]

data = dict(  # 数据的配置
    videos_per_gpu=4,  # 单个 GPU 的批大小
    workers_per_gpu=1,  # 单个 GPU 的 dataloader 的进程
    train_dataloader=dict(  # 训练过程 dataloader 的额外设置
        drop_last=True),  # 在训练过程中是否丢弃最后一个批次
    val_dataloader=dict(  # 验证过程 dataloader 的额外设置
        videos_per_gpu=1),  # 单个 GPU 的批大小
    test_dataloader=dict(  # 测试过程 dataloader 的额外设置
        videos_per_gpu=2),  # 单个 GPU 的批大小
    train=dict(  # 训练数据集的设置
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(  # 验证数据集的设置
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(  # 测试数据集的设置
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# 优化器设置
optimizer = dict(
    # 构建优化器的设置，支持：
    # (1) 所有 PyTorch 原生的优化器，这些优化器的参数和 PyTorch 对应的一致；
    # (2) 自定义的优化器，这些优化器在 `constructor` 的基础上构建。
    # 更多细节可参考 "tutorials/5_new_modules.md" 部分
    type='Adam',  # 优化器类型, 参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13
    lr=0.00008,  # 学习率, 参数的细节使用可参考 PyTorch 的对应文档
    #momentum=0.9,  # 动量大小
    weight_decay=0.001)  #0.001 SGD 优化器权重衰减
optimizer_config = dict(  # 用于构建优化器钩子的设置
    )  # 使用梯度裁剪 grad_clip=dict(max_norm=4000, norm_type=2)
# 学习策略设置
lr_config = dict(  # 用于注册学习率调整钩子的设置
    policy='CosineAnnealing',  # 调整器策略, 支持 CosineAnnealing，Cyclic等方法。更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    min_lr_ratio=0.05,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1
)  # 学习率衰减步长
total_epochs = 100 # 训练模型的总周期数
checkpoint_config = dict(  # 模型权重钩子设置，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
    interval=5)  # 模型权重文件保存间隔
evaluation = dict(  # 训练期间做验证的设置
    interval=5,  # 执行验证的间隔
    metrics=['mean_class_accuracy'],  # 验证方法 'top_k_accuracy',mean_class_accuracy
    save_best='mean_class_accuracy')  # 设置 `top_k_accuracy` 作为指示器，用于存储最好的模型权重文件
log_config = dict(  # 注册日志钩子的设置
    interval=20,  # 打印日志间隔
    hooks=[  # 训练期间执行的钩子
        dict(type='TextLoggerHook'),  # 记录训练过程信息的日志
        #dict(type='TensorboardLoggerHook'),  # 同时支持 Tensorboard 日志
    ])

# 运行设置
dist_params = dict(backend='nccl')  # 建立分布式训练的设置，其中端口号也可以设置
log_level = 'INFO'  # 日志等级
work_dir = './work_dirs/i3d_emd/32_2_1/Adam_mid_mixup_re'  # 记录当前实验日志和模型权重文件的文件夹
load_from = "/home/chenpeizhe/project/guokeda/dongyang/classfication/MMaction/work_dirs/i3d_emd/32_2_1/Adam_mid_mixup_re/epoch_25.pth" # 从给定路径加载模型作为预训练模型. 这个选项不会用于断点恢复训练
resume_from ="/home/chenpeizhe/project/guokeda/dongyang/classfication/MMaction/work_dirs/i3d_emd/32_2_1/Adam_mid_mixup_re/epoch_25.pth" # 加载给定路径的模型权重文件作为断点续连的模型, 训练将从该时间点保存的周期点继续进行
workflow = [('train', 1)]  # runner 的执行流. [('train', 1)] 代表只有一个执行流，并且这个名为 train 的执行流只执行一次

omnisource = True  #是否使用到omnisource训练
gpu_ids = [0]
seed = 100