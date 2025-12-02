import torch
import torch.nn as nn
import torch.nn.functional as F


class Srnet(nn.Module):
    def __init__(self):
        super(Srnet, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)

        # Fully Connected layer
        self.fc = nn.Linear(512 * 1 * 1, 2)

    def forward(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        # print(actv.size())
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        # print(actv.size())
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        # print(res.size())
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        # print(res.size())
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        # print(res.size())
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        # print(res.size())
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        # print(res.size())
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # print(res.size())
        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)
        # print(res.size())
        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # print(res.size())
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # print(res.size())
        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)
        # print(bn.size())
        avgp = torch.mean(bn, dim=(2, 3), keepdim=True)
        flatten = avgp.view(avgp.size(0), -1)
        fc = self.fc(flatten)
        out = F.softmax(fc, dim=1)
        return out


class VIFINet1(nn.Module):
    def __init__(self):
        super(VIFINet1, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)

    def forward(self, inputs):
        # input [b, 16, 352, 288]
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        # print('Layer 1', actv.size())
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        # print('Layer 2', actv.size())
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        # print('Layer 3', res.size())
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        # print('Layer 4', res.size())
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        # print('Layer 5', res.size())
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        # print('Layer 6', res.size())
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        # print('Layer 7', res.size())

        return res


class VIFINet2(nn.Module):
    def __init__(self):
        super(VIFINet2, self).__init__()
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.avg_pool_8 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.avg_pool_9 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.global_avg_pool9 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc9_1 = nn.Linear(in_features=64, out_features=round(64 / 16), bias=False)
        self.fc9_2 = nn.Linear(in_features=round(64 / 16), out_features=64, bias=False)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.avg_pool_10 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.global_avg_pool10 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc10_1 = nn.Linear(in_features=128, out_features=round(128 / 16), bias=False)
        self.fc10_2 = nn.Linear(in_features=round(128 / 16), out_features=128, bias=False)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.avg_pool_11 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.global_avg_pool11 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc11_1 = nn.Linear(in_features=256, out_features=round(256 / 16), bias=False)
        self.fc11_2 = nn.Linear(in_features=round(256 / 16), out_features=256, bias=False)
        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)
        # Layer 13
        self.fc13 = nn.Linear(512 * 1 * 1, 2)

    def forward(self, x):
        # Layer 8
        residual = x
        x = self.layer82(x)
        x = self.bn82(x)
        x = F.relu(x)
        x = self.layer83(x)
        x = self.bn83(x)
        residual = self.layer81(residual)
        residual = self.bn81(residual)
        x = self.avg_pool_8(x)
        x = x + residual
        # print(x.size())
        # Layer 9
        residual = x
        x = self.layer92(x)
        x = self.bn92(x)
        x = F.relu(x)
        x = self.layer93(x)
        x = self.bn93(x)
        residual = self.layer91(residual)
        residual = self.bn91(residual)
        org_x = x
        x = self.global_avg_pool9(x)
        x = x.view(x.size(0), -1)
        x = self.fc9_1(x)
        x = F.relu(x)
        x = self.fc9_2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x * org_x
        x = self.avg_pool_9(x)
        x = x + residual
        # print(x.size())
        # Layer 10
        residual = x
        x = self.layer102(x)
        x = self.bn102(x)
        x = F.relu(x)
        x = self.layer103(x)
        x = self.bn103(x)
        residual = self.layer101(residual)
        residual = self.bn101(residual)
        org_x = x
        x = self.global_avg_pool10(x)
        x = x.view(x.size(0), -1)
        x = self.fc10_1(x)
        x = F.relu(x)
        x = self.fc10_2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x * org_x
        x = self.avg_pool_10(x)
        x = x + residual
        # print(x.size())
        # Layer 11
        residual = x
        x = self.layer112(x)
        x = self.bn112(x)
        x = F.relu(x)
        x = self.layer113(x)
        x = self.bn113(x)
        residual = self.layer111(residual)
        residual = self.bn111(residual)
        org_x = x
        x = self.global_avg_pool11(x)
        x = x.view(x.size(0), -1)
        x = self.fc11_1(x)
        x = F.relu(x)
        x = self.fc11_2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x * org_x
        x = self.avg_pool_11(x)
        x = x + residual
        # print(x.size())
        # Layer 12
        x = self.layer121(x)
        x = self.bn121(x)
        x = F.relu(x)
        x = self.layer122(x)
        x = self.bn122(x)
        # print(x.size())
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = x.view(x.size(0), -1)
        x = self.fc13(x)
        # print(x.size())
        x = F.softmax(x, dim=1)

        return x


class VIFINet(nn.Module):
    def __init__(self):
        super(VIFINet, self).__init__()
        self.vifi_net1 = VIFINet1()
        self.vifi_net2 = VIFINet2()

    def init_vifi_net1(self, chkpt):
        '''
        初始化模型参数
        :param model 存储模型
        '''
        ckpt = torch.load(chkpt)
        parameters_dict = ckpt['model_state_dict']
        for name, parameters in self.vifi_net1.named_parameters():
            parameters.data = parameters_dict[name]

    def init_vifi_net2(self, chkpt):
        ckpt = torch.load(chkpt)
        parameters_dict = ckpt['model_state_dict']
        for name, parameters in self.vifi_net2.named_parameters():
            if name in parameters_dict.keys():
                parameters.data = parameters_dict[name]

    def forward(self, x):
        x = self.vifi_net1(x)
        x = self.vifi_net2(x)
        return x




