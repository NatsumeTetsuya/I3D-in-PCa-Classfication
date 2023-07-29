# -*- coding = utf-8 -*-
# @Time : 2022/3/11 0:47
# @Author : Tetsuya Chen
# @File : model_finetune.py
# @software : PyCharm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch_utils.models.layers import FastGlobalConcatPool2d, FastGlobalAvgPool2d, GeM_cw, MultiSampleDropoutFC, SEBlock
def finetune_models(model, classes, drop_out, no_load_fc=True, change_fc=True, freeze = False, model_path = False, pretrain_model = False):

    if freeze:
        for param in model.parameters:
            param.requires_grad = False
    if change_fc:
        try :
            in_fc = model.fc.in_features

            print(f'in_fc = {in_fc}')
            if in_fc == 512:
                model.fc = nn.Sequential(nn.Linear(in_fc, 256),
                                         #nn.GELU(),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(256, 128),
                                         #nn.GELU(),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(128, classes)
                                         )
            elif in_fc == 1024:
                model.fc = nn.Sequential(nn.Linear(in_fc, 4096),
                                         # nn.GELU(),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(4096, 1024),
                                         # nn.GELU(),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(1024, 512),
                                         # nn.GELU(),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(512, classes)
                                         )
            elif in_fc == 2048:
                model.fc = nn.Sequential(nn.Linear(in_fc, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(1024, 512),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(512, 256),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(256, classes)
                                         )
            elif in_fc == 4096:
                model.fc = nn.Sequential(nn.Linear(in_fc, 2048),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(2048, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(1024, 256),
                                         nn.ReLU(),
                                         nn.Dropout(p=drop_out),
                                         nn.Linear(256, classes)
                                         )
        except:
            print('this model has no fc')

    if model_path or pretrain_model:
        model_dict = model.state_dict()
        if model_path:
            pretrained_dict = torch.load(model_path)
        else:
            pretrained_dict = pretrain_model
        if no_load_fc:
            _ = pretrained_dict.pop('fc.weight')
            _ = pretrained_dict.pop('fc.bias')
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.fc.named_parameters():
                param.requires_grad = True
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    #print(model)

    return model

def load_pretrain(model, model_path=False, pretrain_model=False, no_load_fc=True, freeze=False):
    if model_path or pretrain_model:
        model_dict = model.state_dict()

        if model_path:

            pretrained_dict = torch.load(model_path)
        else:
            pretrained_dict = pretrain_model
        if no_load_fc:
            try:
                _ = pretrained_dict.pop('fc.weight')
                _ = pretrained_dict.pop('fc.bias')
                # _ = pretrained_dict.pop('classifier.weight')
                # _ = pretrained_dict.pop('classifier.bias')
            except:
                pass
            try:
                _ = pretrained_dict.pop('classifier.weight')
                _ = pretrained_dict.pop('classifier.bias')
            except:
                pass
            try:
                _ = pretrained_dict.pop('last_linear.weight')
                _ = pretrained_dict.pop('last_linear.bias')
            except:
                pass
            try:
                _ = pretrained_dict.pop('roi_heads.box_predictor.cls_score.weight')
                _ = pretrained_dict.pop('roi_heads.box_predictor.cls_score.bias')
                _ = pretrained_dict.pop('roi_heads.box_predictor.bbox_pred.weight')
                _ = pretrained_dict.pop('roi_heads.box_predictor.bbox_pred.bias')
            except:
                pass
        print(pretrained_dict.keys())
        #pretrained_dict = torch.nn.remove_prefix('model.')
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        print(model_dict.keys())
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.fc.named_parameters():
                param.requires_grad = True
    return model

class ImageModel(nn.Module):

    def __init__(self,
                 model,
                 pretrain_path,
                 model_backbone=0,
                 test_model_path=False,
                 pooling='concat',
                 fc='multi-dropout',
                 no_load_fc=True,
                 dropout=0,
                 num_feature=2048,
                 classes=1,
                 in_channel=3):
        super(ImageModel, self).__init__()
        self.model = model
        self.model_backbone = model_backbone
        self.pool = pooling

        if pretrain_path:
            self.model = load_pretrain(self.model, model_path=pretrain_path, pretrain_model=False, no_load_fc=no_load_fc, freeze=False)
        if model_backbone == 1:
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            if pooling == 'concat':
                self.pooling = FastGlobalConcatPool2d()
                num_feature *= 2
            elif pooling == 'gem':
                self.pooling = GeM_cw(num_feature)
            elif pooling == 'avgpool':
                self.pooling = nn.AdaptiveAvgPool2d((1,1))
            else:
                self.pooling = FastGlobalAvgPool2d()

            if fc == 'multi-dropout':
                self.fc = nn.Sequential(
                            MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes, dropout=dropout))

            if fc == 'attention':
                self.fc = nn.Sequential(
                            SEBlock(num_feature),
                            MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes, dropout=dropout))

            elif fc == 'dropout':
                self.fc = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(num_feature, classes, bias=True))

            elif fc == '2layers':
                self.fc = nn.Sequential(
                            nn.Linear(num_feature, 512, bias=False),
                            nn.BatchNorm1d(512),
                            nn.SiLU(inplace=True),
                            nn.Dropout(dropout),
                            nn.Linear(512, classes, bias=True))

            elif fc == 'mlp':
                self.fc = nn.Sequential(nn.Linear(num_feature, 256),
                                         # nn.GELU(),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.Linear(256, 128),
                                         # nn.GELU(),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.Linear(128, classes)
                                         )

            else:
                self.fc = nn.Linear(in_features=num_feature, out_features=classes, bias=True)
            #self.fc = initialize_weights(self.fc)

        elif model_backbone == 0:
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            if pooling == 'concat':
                self.model.pooling = FastGlobalConcatPool2d()
                num_feature *= 2
            elif pooling == 'gem':
                self.model.pooling = GeM_cw(num_feature)
            elif pooling == 'avgpool':
                self.model.pooling = nn.AdaptiveAvgPool2d((1,1))
            else:
                self.model.pooling = FastGlobalAvgPool2d()

            if fc == 'multi-dropout':
                self.model.fc = nn.Sequential(
                    MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes, dropout=dropout))

            if fc == 'attention':
                self.model.fc = nn.Sequential(
                    SEBlock(num_feature),
                    MultiSampleDropoutFC(in_ch=num_feature, out_ch=classes, dropout=dropout))

            elif fc == 'dropout':
                self.model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_feature, classes, bias=True))

            elif fc == '2layers':
                self.model.fc = nn.Sequential(
                    nn.Linear(num_feature, 512, bias=False),
                    nn.BatchNorm1d(512),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(512, classes, bias=True))

            elif fc == 'mlp':
                self.model.fc = nn.Sequential(nn.Linear(num_feature, 256),
                                         # nn.GELU(),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.Linear(256, 128),
                                         # nn.GELU(),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.Linear(128, classes)
                                         )

            else:
                self.model.fc = nn.Linear(in_features=num_feature, out_features=classes, bias=True)
        else:

            pass



        if test_model_path:
            self.model = load_pretrain(self.model, model_path=test_model_path, pretrain_model=False, no_load_fc=False, freeze=False)

    @autocast()
    def forward(self, x):
        if self.model_backbone == 1:
            feature_map = self.model(x)
            embedding = self.pooling(feature_map)
            if self.pool=='avgpool':
                embedding = embedding.view(embedding.size(0), -1)
            logits = self.fc(embedding)
        else:
            logits = self.model(x)
        return logits

def double_chain(model1, model2, classes, freeze = False, model_path1 = False, model_path2 = False):
    if model_path1:
        model_dict1 = model1.state_dict()
        pretrained_dict1 = torch.load(model_path1)
        pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in model_dict1}
        print(pretrained_dict1.keys())
        model_dict1.update(pretrained_dict1)
        model1.load_state_dict(model_dict1)
        if freeze:
            for name, param in model1.named_parameters():
                param.requires_grad = False
            for name, param in model1.fc.named_parameters():
                param.requires_grad = True
    if model_path2:
        model_dict2 = model2.state_dict()
        pretrained_dict2 = torch.load(model_path2)
        pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in model_dict2}
        print(pretrained_dict2.keys())
        model_dict2.update(pretrained_dict2)
        model2.load_state_dict(model_dict2)
        if freeze:
            for name, param in model2.named_parameters():
                param.requires_grad = False
            for name, param in model2.fc.named_parameters():
                param.requires_grad = True
    in_fc1 = model1.fc.in_features
    in_fc2 = model2.fc.in_features
    model1.fc = nn.Sequential(
                             )
    model2.fc = nn.Sequential(
                              )
    model_fc = nn.Sequential(nn.Linear(in_fc1 + in_fc2, 1024),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(1024, 512),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(512, 256),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(256, classes))
    return model1, model2, model_fc

def initialize_weights(model):
    for m in model.modules():

        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
                # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 1, 0.1)
            # m.weight.data.normal_(0,0.01)
            m.bias.data.zero_()
    return model