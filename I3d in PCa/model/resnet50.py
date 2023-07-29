import torch
import torch.nn as nn
import numpy as np
# from torchvision.models.utils import load_state_dict_from_url
from ptc.model.attention import cbam_block, SpatialAttention
from ptc.model.van import Block as LKA

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)              #nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=3, attention=False, attention_dim=0, add_shape=False, zero_init_residual=False):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.attention = attention
        self.add_shape = add_shape

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        #self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if self.add_shape:
            self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
            #self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.features_list = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3,
        #                       self.layer4]
        # self.classfier_list = [self.avgpool, self.fc]
        # self.features = nn.Sequential(*self.features_list)
        # self.classfier = nn.Sequential(*self.classfier_list)

        if self.attention:
            self.attention0 = self.attention(attention_dim)
            self.attention1 = self.attention(attention_dim)
            self.attention2 = self.attention(attention_dim*2)
            self.attention3 = self.attention(attention_dim*4)
            self.attention4 = self.attention(attention_dim*8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.attention:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            #x = self.attention0(x)
            x = self.layer1(x)
            #x = self.attention1(x)
            x = self.layer2(x)
            #x = self.attention2(x)
            x = self.layer3(x)
            #x = self.attention3(x)
            x = self.layer4(x)
            x = self.attention4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            self.featuremap1 = x.detach()

            x = self.fc(x)
        else:
            x = self.conv1(x)
            #self.featuremap1 = x.detach()
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            #self.featuremap1 = x.detach()
            x = x.view(x.size(0), -1)

                # len_n = y[:, 0].unsqueeze(-1)
                # ratio = y[:, 1].unsqueeze(-1)
                # x = x * ratio + len_n

            x = self.fc(x)
        return x

class ResNet18_pca(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=3, pca_size=24, dropout = 0, attention=False, attention_dim=0, zero_init_residual=False):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet18_pca, self).__init__()
        self.attention = attention
        self.dropout = dropout

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        #self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion, 256),
                                # nn.GELU(),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(256, 128),
                                # nn.GELU(),
                                )
        self.fc2 = nn.Sequential(nn.Linear(128+pca_size, 64),
                                # nn.GELU(),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(64, num_classes),
                                )

        # self.features_list = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3,
        #                       self.layer4]
        # self.classfier_list = [self.avgpool, self.fc]
        # self.features = nn.Sequential(*self.features_list)
        # self.classfier = nn.Sequential(*self.classfier_list)

        if self.attention:
            self.attention0 = self.attention(attention_dim)
            self.attention1 = self.attention(attention_dim)
            self.attention2 = self.attention(attention_dim*2)
            self.attention3 = self.attention(attention_dim*4)
            self.attention4 = self.attention(attention_dim*8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x, y = x[0], x[1]
        if self.attention:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.attention0(x)
            x = self.layer1(x)
            x = self.attention1(x)
            x = self.layer2(x)
            x = self.attention2(x)
            x = self.layer3(x)
            x = self.attention3(x)
            x = self.layer4(x)
            x = self.attention4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            self.featuremap1 = x.detach()

            x = self.fc1(x)
            cat = torch.cat([x, y], dim=1)
            out = self.fc2(cat)
            return out
        else:

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # x = self.attention0(x)
            x = self.layer1(x)
            # x = self.attention1(x)
            x = self.layer2(x)
            # x = self.attention2(x)
            x = self.layer3(x)
            # x = self.attention3(x)
            x = self.layer4(x)
            # x = self.attention4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            self.featuremap1 = x.detach()

            x = self.fc1(x)
            cat = torch.cat([x, y], dim=1)
            out = self.fc2(cat)
            return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=3, attention=False, attention_dim=0, add_shape=False, zero_init_residual=False):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.attention = attention
        self.add_shape = add_shape

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        if self.add_shape:
            self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
            #self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        if self.attention:
            self.attention0 = self.attention(dim=64)
            self.attention1 = self.attention(dim=256)
            self.attention2 = self.attention(dim=512)
            self.attention3 = self.attention(dim=1024)
            self.attention4 = self.attention(dim=2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        if self.attention:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.attention0(x)
            x = self.layer1(x)
            x = self.attention1(x)
            x = self.layer2(x)
            x = self.attention2(x)
            x = self.layer3(x)
            x = self.attention3(x)
            x = self.layer4(x)
            x = self.attention4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            #self.featuremap1 = x.detach()
            if self.add_shape:
                x = torch.cat((x.unsqueeze(-1), x.unsqueeze(-1)), -1)
                y = y.unsqueeze(1)
                x = x * y
                x = x.view(x.size(0), -1)

                # len_n = y[:, 0].unsqueeze(-1)
                # ratio = y[:, 1].unsqueeze(-1)
                # x = x * ratio + len_n
            x = self.fc(x)
        else:
            x = self.conv1(x)
            self.featuremap1 = x.detach()
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            self.featuremap2 = x.detach()
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            self.featuremap0 = x.detach()
            if self.add_shape:
                x = torch.cat((x.unsqueeze(-1), x.unsqueeze(-1)), -1)
                y = y.unsqueeze(1)
                x = x * y
                x = x.view(x.size(0), -1)

                # len_n = y[:, 0].unsqueeze(-1)
                # ratio = y[:, 1].unsqueeze(-1)
                # x = x * ratio + len_n
            x = self.fc(x)
        return x

class Dual_ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channel1=30, in_channel2=1, dual_fc=False, attention1=False, attention2=False, attention_dim=0, zero_init_residual=False):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes1 = 64
        self.inplanes2 = 64
        super(Dual_ResNet18, self).__init__()
        self.dual_fc = dual_fc
        self.attention1 = attention1
        self.attention2 = attention2


        # 600,600,3 -> 300,300,64
        self.conv1_1 = nn.Conv2d(in_channel1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = nn.Conv2d(in_channel2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        #self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1_1 = self._make_layer1(block, 64, layers[0])
        self.layer1_2 = self._make_layer2(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2_1 = self._make_layer1(block, 128, layers[1], stride=2)
        self.layer2_2 = self._make_layer2(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3_1 = self._make_layer1(block, 256, layers[2], stride=2)
        self.layer3_2 = self._make_layer2(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4_1 = self._make_layer1(block, 512, layers[3], stride=2)
        self.layer4_2 = self._make_layer2(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(1024 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)


        # self.features_list = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3,
        #                       self.layer4]
        # self.classfier_list = [self.avgpool, self.fc]
        # self.features = nn.Sequential(*self.features_list)
        # self.classfier = nn.Sequential(*self.classfier_list)

        if self.attention1:
            self.attention0 = self.attention(attention_dim)
            self.attention1 = self.attention(attention_dim)
            self.attention2 = self.attention(attention_dim*2)
            self.attention3 = self.attention(attention_dim*4)
            self.attention4 = self.attention(attention_dim*8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer1(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes1, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes1, planes, stride, downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes1, planes))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes2, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample))
        self.inplanes2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes2, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.attention1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            #x = self.attention0(x)
            x = self.layer1(x)
            #x = self.attention1(x)
            x = self.layer2(x)
            #x = self.attention2(x)
            x = self.layer3(x)
            #x = self.attention3(x)
            x = self.layer4(x)
            x = self.attention4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            self.featuremap1 = x.detach()

            x = self.fc(x)
        else:
            random_frame = torch.randint((0,x.size(1)))
            y = x[:, x.size(1)//2, :, :].unsqueeze(1)
            x = self.conv1_1(x)
            y = self.conv1_2(y)

            x = self.bn1_1(x)
            y = self.bn1_2(y)

            x = self.relu(x)
            y = self.relu(y)

            x = self.maxpool(x)
            y = self.maxpool(y)

            x = self.layer1_1(x)
            y = self.layer1_2(y)
            x = x + y

            x = self.layer2_1(x)
            y = self.layer2_2(y)
            x = x + y

            x = self.layer3_1(x)
            y = self.layer3_2(y)
            x = x + y
            x = self.layer4_1(x)
            y = self.layer4_2(y)
            z = x + y

            x = self.avgpool(x)
            y = self.avgpool(y)
            z = self.avgpool(z)

            #self.featuremap1 = x.detach()
            #第一种concatx,y
            if self.dual_fc:
                x = torch.concat([x, y], 1)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)

            #第二种直接对z
            else:
                x = z.view(z.size(0), -1)
                x = self.fc2(x)
        return x

class Dual_ResNet18_concat(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channel1=30, in_channel2=1, dropout=0.2, dual_fc=False, attention1=False, attention2=False, attention_dim=0, zero_init_residual=False):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes1 = 64
        self.inplanes2 = 64
        super(Dual_ResNet18_concat, self).__init__()
        self.dual_fc = dual_fc
        self.attention1 = attention1
        self.attention2 = attention2


        # 600,600,3 -> 300,300,64
        self.conv1_1 = nn.Conv2d(in_channel1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = nn.Conv2d(in_channel2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        #self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1_1 = self._make_layer1(block, 64, layers[0])
        self.layer1_2 = self._make_layer2(block, 64, layers[0])
        self.conv1x1_1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # 150,150,256 -> 75,75,512
        self.layer2_1 = self._make_layer1(block, 128, layers[1], stride=2)
        self.layer2_2 = self._make_layer2(block, 128, layers[1], stride=2)
        self.conv1x1_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3_1 = self._make_layer1(block, 256, layers[2], stride=2)
        self.layer3_2 = self._make_layer2(block, 256, layers[2], stride=2)
        self.conv1x1_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.layer4被用在classifier模型中
        self.layer4_1 = self._make_layer1(block, 512, layers[3], stride=2)
        self.layer4_2 = self._make_layer2(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024 * block.expansion, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes))


        # self.features_list = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3,
        #                       self.layer4]
        # self.classfier_list = [self.avgpool, self.fc]
        # self.features = nn.Sequential(*self.features_list)
        # self.classfier = nn.Sequential(*self.classfier_list)

        if self.attention1:
            self.attention0 = self.attention(attention_dim)
            self.attention1 = self.attention(attention_dim)
            self.attention2 = self.attention(attention_dim*2)
            self.attention3 = self.attention(attention_dim*4)
            self.attention4 = self.attention(attention_dim*8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer1(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes1, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes1, planes, stride, downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes1, planes))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes2, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample))
        self.inplanes2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes2, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.attention1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            #x = self.attention0(x)
            x = self.layer1(x)
            #x = self.attention1(x)
            x = self.layer2(x)
            #x = self.attention2(x)
            x = self.layer3(x)
            #x = self.attention3(x)
            x = self.layer4(x)
            x = self.attention4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            self.featuremap1 = x.detach()

            x = self.fc(x)
        else:
            y = x[:, x.size(1)//2, :, :].unsqueeze(1)
            x = self.conv1_1(x)
            y = self.conv1_2(y)

            x = self.bn1_1(x)
            y = self.bn1_2(y)

            x = self.relu(x)
            y = self.relu(y)

            x = self.maxpool(x)
            y = self.maxpool(y)

            x = self.layer1_1(x)
            y = self.layer1_2(y)
            x = torch.concat([x,y], 1)
            x = self.conv1x1_1(x)

            x = self.layer2_1(x)
            y = self.layer2_2(y)
            x = torch.concat([x, y], 1)
            x = self.conv1x1_2(x)

            x = self.layer3_1(x)
            y = self.layer3_2(y)
            x = torch.concat([x, y], 1)
            x = self.conv1x1_3(x)

            x = self.layer4_1(x)
            y = self.layer4_2(y)
            x = torch.concat([x, y], 1)
            x = self.avgpool(x)

            #self.featuremap1 = x.detach()
            #第一种concatx,y

            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

class Dual_ResNet18_concat_rf(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channel1=30, in_channel2=1, dropout=0.2, dual_fc=False, attention1=False, attention2=False, attention_dim=64, zero_init_residual=False):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        super(Dual_ResNet18_concat_rf, self).__init__()
        self.inplanes1 = 64
        self.inplanes2 = 64

        self.dual_fc = dual_fc
        self.attention_dim = attention_dim
        self.attention1 = attention1
        self.attention2 = attention2


        # 600,600,3 -> 300,300,64
        self.conv1_1 = nn.Conv2d(in_channel1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = nn.Conv2d(in_channel2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        #self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1_1 = self._make_layer1(block, 64, layers[0])
        self.layer1_2 = self._make_layer2(block, 64, layers[0])
        self.conv1x1_1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # 150,150,256 -> 75,75,512
        self.layer2_1 = self._make_layer1(block, 128, layers[1], stride=2)
        self.layer2_2 = self._make_layer2(block, 128, layers[1], stride=2)
        self.conv1x1_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3_1 = self._make_layer1(block, 256, layers[2], stride=2)
        self.layer3_2 = self._make_layer2(block, 256, layers[2], stride=2)
        self.conv1x1_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.layer4被用在classifier模型中
        self.layer4_1 = self._make_layer1(block, 512, layers[3], stride=2)
        self.layer4_2 = self._make_layer2(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024 * block.expansion, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes))


        # self.features_list = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3,
        #                       self.layer4]
        # self.classfier_list = [self.avgpool, self.fc]
        # self.features = nn.Sequential(*self.features_list)
        # self.classfier = nn.Sequential(*self.classfier_list)

        if self.attention1:
            self.attention1_0 = self.attention1(attention_dim)       #64
            self.attention1_1 = self.attention1(attention_dim)       #64
            self.attention1_2 = self.attention1(attention_dim*2)     #128
            self.attention1_3 = self.attention1(attention_dim*4)     #256
            self.attention1_4 = self.attention1(attention_dim*8)     #512
        if self.attention2:
            self.attention2_0 = self.attention2(attention_dim)
            self.attention2_1 = self.attention2(attention_dim)
            self.attention2_2 = self.attention2(attention_dim*2)
            self.attention2_3 = self.attention2(attention_dim*4)
            self.attention2_4 = self.attention2(attention_dim*8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer1(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes1, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes1, planes, stride, downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes1, planes))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes2, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample))
        self.inplanes2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes2, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.attention1:
            x, frame = x['imgs'], int(x['frame'])
            y = x[:, frame, :, :].unsqueeze(1)
            x = self.conv1_1(x)
            y = self.conv1_2(y)

            x = self.bn1_1(x)
            y = self.bn1_2(y)
            x = self.attention1_0(x)
            y = self.attention2_0(y)

            x = self.relu(x)
            y = self.relu(y)

            x = self.maxpool(x)
            y = self.maxpool(y)

            x = self.layer1_1(x)
            y = self.layer1_2(y)

            x = self.attention1_1(x)
            y = self.attention2_1(y)

            x = torch.concat([x, y], 1)
            x = self.conv1x1_1(x)

            x = self.layer2_1(x)
            y = self.layer2_2(y)
            x = self.attention1_2(x)
            y = self.attention2_2(y)
            x = torch.concat([x, y], 1)
            x = self.conv1x1_2(x)

            x = self.layer3_1(x)
            y = self.layer3_2(y)
            x = self.attention1_3(x)
            y = self.attention2_3(y)
            x = torch.concat([x, y], 1)
            x = self.conv1x1_3(x)

            x = self.layer4_1(x)
            y = self.layer4_2(y)
            x = self.attention1_4(x)
            y = self.attention2_4(y)
            x = torch.concat([x, y], 1)
            x = self.avgpool(x)

            # self.featuremap1 = x.detach()
            # 第一种concatx,y

            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x, frame = x['imgs'], int(x['frame'])
            y = x[:, frame, :, :].unsqueeze(1)
            x = self.conv1_1(x)
            y = self.conv1_2(y)

            x = self.bn1_1(x)
            y = self.bn1_2(y)

            x = self.relu(x)
            y = self.relu(y)

            x = self.maxpool(x)
            y = self.maxpool(y)

            x = self.layer1_1(x)
            y = self.layer1_2(y)
            x = torch.concat([x,y], 1)
            x = self.conv1x1_1(x)

            x = self.layer2_1(x)
            y = self.layer2_2(y)
            x = torch.concat([x, y], 1)
            x = self.conv1x1_2(x)

            x = self.layer3_1(x)
            y = self.layer3_2(y)
            x = torch.concat([x, y], 1)
            x = self.conv1x1_3(x)

            x = self.layer4_1(x)
            y = self.layer4_2(y)
            x = torch.concat([x, y], 1)
            x = self.avgpool(x)

            #self.featuremap1 = x.detach()
            #第一种concatx,y

            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

class resResNet18_add_rf(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channel1=30, in_channel2=1, dropout=0.2, dual_fc=False, attention1=False, attention2=False, attention_dim=0, zero_init_residual=False):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        super(resResNet18_add_rf, self).__init__()
        self.inplanes1 = 64
        self.inplanes2 = 64
        self.inplanes3 = 64

        self.dual_fc = dual_fc
        self.attention1 = attention1
        self.attention2 = attention2


        # 600,600,3 -> 300,300,64
        #self.conv1_0 = nn.Conv2d(in_channel1, 32, kernel_size=3, stride=1, padding=0, bias=False)
        #self.conv1_1 = nn.Conv2d(32,          64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_1 = nn.Conv2d(in_channel1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_2 = nn.Conv2d(in_channel1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        #self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1_1 = self._make_layer1(block, 64, layers[0][0])
        self.layer1_2 = self._make_layer2(block, 64, layers[1][0])
        self.layer1_3 = self._make_layer3(block, 64, layers[2][0])

        # 150,150,256 -> 75,75,512
        self.layer2_1 = self._make_layer1(block, 128, layers[0][1], stride=2)
        self.layer2_2 = self._make_layer2(block, 128, layers[1][1], stride=2)
        self.layer2_3 = self._make_layer3(block, 128, layers[2][1], stride=2)

        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3_1 = self._make_layer1(block, 256, layers[0][2], stride=2)
        self.layer3_2 = self._make_layer2(block, 256, layers[1][2], stride=2)
        self.layer3_3 = self._make_layer3(block, 256, layers[2][2], stride=2)

        # self.layer4被用在classifier模型中
        self.layer4_1 = self._make_layer1(block, 512, layers[0][3], stride=2)
        self.layer4_2 = self._make_layer2(block, 512, layers[1][3], stride=2)
        self.layer4_3 = self._make_layer3(block, 512, layers[2][3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes))


        # self.features_list = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3,
        #                       self.layer4]
        # self.classfier_list = [self.avgpool, self.fc]
        # self.features = nn.Sequential(*self.features_list)
        # self.classfier = nn.Sequential(*self.classfier_list)

        if self.attention1:
            self.attention0 = self.attention(attention_dim)
            self.attention1 = self.attention(attention_dim)
            self.attention2 = self.attention(attention_dim*2)
            self.attention3 = self.attention(attention_dim*4)
            self.attention4 = self.attention(attention_dim*8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer1(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes1, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes1, planes, stride, downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes1, planes))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes2, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample))
        self.inplanes2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes2, planes))
        return nn.Sequential(*layers)

    def _make_layer3(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes3, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.attention1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            #x = self.attention0(x)
            x = self.layer1(x)
            #x = self.attention1(x)
            x = self.layer2(x)
            #x = self.attention2(x)
            x = self.layer3(x)
            #x = self.attention3(x)
            x = self.layer4(x)
            x = self.attention4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            self.featuremap1 = x.detach()

            x = self.fc(x)
        else:
            x, frame = x['imgs'], int(x['frame'])
            z = x[:, frame, :, :].unsqueeze(1)
            y = x - z.repeat(1, x.size(1), 1, 1)

            x = self.conv1_1(x)
            y = self.conv1_2(y)

            x = self.bn1_1(x)
            y = self.bn1_2(y)

            x = self.relu(x)
            y = self.relu(y)

            x = self.maxpool(x)
            y = self.maxpool(y)

            x = self.layer1_1(x)
            y = self.layer1_2(y)
            x = x + y


            x = self.layer2_1(x)
            y = self.layer2_2(y)
            x = x + y

            x = self.layer3_1(x)
            y = self.layer3_2(y)
            x = x + y

            x = self.layer4_1(x)
            y = self.layer4_2(y)
            x = x + y
            x = self.avgpool(x)

            #self.featuremap1 = x.detach()
            #第一种concatx,y

            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

def resnet50_attention():
    model = ResNet50(Bottleneck,[3,4,6,3], attention=cbam_block)
    return model

def resnet50(add_shape):
    model = ResNet50(Bottleneck,[3,4,6,3], num_classes=2 ,attention=False, add_shape=add_shape)
    return model

def resnet18_LKA(add_shape):
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=2 ,attention=LKA, attention_dim=64, add_shape=add_shape)
    return model

def resnet18_cbam(add_shape):
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=2 ,attention=cbam_block, attention_dim=64, add_shape=add_shape)
    return model

def resnet18(add_shape):
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=2 ,attention=False, add_shape=add_shape)
    return model

def resnet18_pca(pca_size=25, dropout=0.5, attention = False):
    model = ResNet18_pca(BasicBlock, [2, 2, 2, 2], num_classes=2, pca_size=pca_size, dropout = dropout, attention=attention, attention_dim=64)
    return model

def resnet18_2211_attention():
    model = ResNet18(BasicBlock, [2, 2, 1, 1], attention=cbam_block)
    return model

def resnet18_2211():
    model = ResNet18(BasicBlock, [2, 2, 1, 1], attention=False)
    return model
def resnet18_2111_attention():
    model = ResNet18(BasicBlock, [2, 1, 1, 1], attention=False)
    return model
def resnet18_2111():
    model = ResNet18(BasicBlock, [2, 1, 1, 1], attention=False)
    return model


def resnet18_3322():
    model = ResNet18(BasicBlock, [3, 3, 2, 2], attention=False)
    return model

def stack_resnet18(in_channels):
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=2, in_channels=in_channels)
    return model

def stack_resnet50(in_channels):
    model = ResNet50(Bottleneck, [3,4,6,3], num_classes=2, in_channels=in_channels)
    return model

def stack_dual_resnet18(in_channel1, inchannel2, dual_fc):
    model = Dual_ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=2, in_channel1=in_channel1, in_channel2=inchannel2, dual_fc=dual_fc)
    return model

def stack_dual_resnet18_concat(in_channel1, inchannel2):
    model = Dual_ResNet18_concat(BasicBlock, [2, 2, 2, 2], num_classes=2, in_channel1=in_channel1, in_channel2=inchannel2,
                          )
    return model
def stack_dual_resnet18_concat_rf(in_channel1, inchannel2):
    model = Dual_ResNet18_concat_rf(BasicBlock, [2, 2, 2, 2], num_classes=2, in_channel1=in_channel1, in_channel2=inchannel2, dropout=0.5
                          )
    return model

def stack_dual_resnet18_concat_rf_att(in_channel1, inchannel2):
    model = Dual_ResNet18_concat_rf(BasicBlock, [2, 2, 2, 2], num_classes=2, in_channel1=in_channel1, in_channel2=inchannel2, dropout=0.5, attention1=LKA, attention2=cbam_block,
                          )
    return model

def stack_resResnet_add_rf(in_channel1, inchannel2):
    model = resResNet18_add_rf(BasicBlock, [[2,2,1,1],[2,2,1,1],[2,2,1,1]], num_classes=2, in_channel1=in_channel1,
                                    in_channel2=inchannel2, dropout=0.5
                                    )
    return model
def stack_resResnet18_add_rf(in_channel1, inchannel2):
    model = resResNet18_add_rf(BasicBlock, [[2,2,2,2],[2,2,2,2],[2,2,2,2]], num_classes=2, in_channel1=in_channel1,
                                    in_channel2=inchannel2, dropout=0.5
                                    )
    return model

