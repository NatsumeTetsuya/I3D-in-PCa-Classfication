# -*- coding = utf-8 -*-
# @Time : 2022/5/28 0:39
# @Author : Tetsuya Chen
# @File : config.py
# @software : PyCharm
import torch
import torch.nn as nn
import torch_utils as tu
import torchvision
from model.resnet50 import *
from util.utils import CutMixCrossEntropyLoss
from vit_pytorch import SimpleViT, ViT, MAE
from vit_pytorch.vit_for_small_dataset import ViT as sd_ViT
from vit_pytorch.mobile_vit import MobileViT


mbvit_xs = MobileViT(
    image_size = (256, 256),
    dims = [96, 120, 144],
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    num_classes = 2
)


s_vit = SimpleViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

vit = ViT(
    channels=30,
    image_size = 256,
    patch_size = 16,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

sd_vit = sd_ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

mae = MAE(
    encoder = vit,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)
class CONFIG():
    #--------------------------------DATA-----------------------------------------------
    device = 0
    DATA_DIR = ""

    #------------------------------TRAINING PARAMS-------------------------------------
    CLASSES = 2
    FOLD = 1
    BATCHSIZE = 64
    ACCUMULATE = 1
    LR = 1.5e-4
    WEIGHT_DECAY = 0.05
    EPOCH = 200
    DECAY_SCALE = 40.0
    SCALER = torch.cuda.amp.GradScaler()
    SEED = 113
    MIXUP = False
    CUTMIX = False
    DROP_OUT = 0.2
    COUNT = 200
    AUTOAUG = False
    STACK = False
    DUAL_STACK = False
    random_frame = False


    # --------------------------------TEST PARAMS-----------------------------------------
    TTA = False

    #-------------------------------NET WORKS-------------------------------------------
    INIT = False
    sigmoid = False
    NO_LOAD_FC = False
    CHANGE_FC = True
    CONTINUE_TRAIN = True
    attention = False
    resnest50d = False
    tu_model = False
    backbone = -1
    pooling = 'concat'
    fc = 'dropout'
    num_feature = 512#1536

    MODEL = resnet18(add_shape=False)
    if sigmoid:
        MODEL = torchvision.models.resnet18(num_classes=1, pretrained=False)
    #MODEL = resnet50(add_shape=False)
    #MODEL = timm.create_model('mobilenetv2_050')
    #MODEL = timm.create_model('resnest14d')
    #MODEL = resnest18d_21()

    #MODEL = resnest20d_21(pretrained=True)
    #MODEL = resnest50(pretrained=True)

    #MODEL = ResNet50()
    #MODEL = VAN(num_classes=2, drop_path_rate=0.1, mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2])
    #MODEL = van_small(pretrained=True, num_classes=2)
    #MODEL = tu.ImageModel(name='resnest50d', pretrained=True, classes=CLASSES, num_feature = 2048, fc='2layers')
    #MODEL = cnn(32)
    #MODEL = s_vit
    #MODEL = vit
    MODEL_SAVE_DIR = "model/out.pth"
    #MODEL = torchvision.models.resnet50(pretrained=True, classes=CLASSES)

    #----------------------------------------------
    #OPTIMIZER = AdamW()
    #OPTIMIZER = tu.RangerLars(MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pretrained_model_path = False

    test_model_path = None
    criterion = nn.CrossEntropyLoss()


    if MIXUP:
        criterion = tu.SoftTargetCrossEntropy()
    if CUTMIX:
        criterion = CutMixCrossEntropyLoss(True)


    criterion_test = nn.CrossEntropyLoss()#weight=torch.FloatTensor([1., 1.]))


    if sigmoid:
        criterion = nn.BCEWithLogitsLoss()
        criterion_test = nn.BCEWithLogitsLoss()
        CLASSES = 1
    #criterion_test = tu.LabelSmoothingCrossEntropy()
    FLOOD = False
    #FLOOD = 0.28

    EXP = None
