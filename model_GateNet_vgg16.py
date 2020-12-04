import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from multi_scale_module import FoldConv_aspp


class GateNet(nn.Module):
    def __init__(self):
        super(GateNet, self).__init__()
        ################################vgg16#######################################
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv1 = nn.Sequential(*feats[:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])
        ################################Gate#######################################
        self.attention_feature5 = nn.Sequential(nn.Conv2d(64+32, 2, kernel_size=3, padding=1))
        self.attention_feature4 = nn.Sequential(nn.Conv2d(128+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(256+128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                nn.Conv2d(128, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(512+256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                                nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(512+512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
                                                 nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                 nn.Conv2d(128, 2, kernel_size=3, padding=1))
        ###############################Transition Layer########################################
        self.dem1 = nn.Sequential(FoldConv_aspp(in_channel=512,
                      out_channel=512,
                      out_size=384 // 16,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      win_size=2,
                      win_padding=0,

        ), nn.BatchNorm2d(512), nn.PReLU())
        self.dem2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.dem3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.dem4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem5 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        ################################Parallel branch#######################################
        self.dem1_1 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.dem2_1 = nn.Sequential(nn.Conv2d(256, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.dem3_1 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.dem4_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.dem5_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        self.out_res = nn.Sequential(nn.Conv2d(32+32+32+32+32+1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                     nn.Conv2d(64, 1, kernel_size=3, padding=1))
        #######################################################################


        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True


    def forward(self, x):
        input = x
        B,_,_,_ = input.size()
        ################################Encoder block#######################################
        E1 = self.conv1(x)
        E2 = self.conv2(E1)
        E3 = self.conv3(E2)
        E4 = self.conv4(E3)
        E5 = self.conv5(E4)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Gated FPN#######################################
        G5 = self.attention_feature1(torch.cat((E5,T5),1))
        G5 = F.adaptive_avg_pool2d(F.sigmoid(G5),1)
        D5 = self.output1(G5[:, 0,:,:].unsqueeze(1).repeat(1,512,1,1)*T5)

        G4 = self.attention_feature2(torch.cat((E4,F.upsample(D5, size=E4.size()[2:], mode='bilinear')),1))
        G4 = F.adaptive_avg_pool2d(F.sigmoid(G4),1)
        D4 = self.output2(F.upsample(D5, size=E4.size()[2:], mode='bilinear')+G4[:, 0,:,:].unsqueeze(1).repeat(1,256,1,1)*T4)

        G3 = self.attention_feature3(torch.cat((E3,F.upsample(D4, size=E3.size()[2:], mode='bilinear')),1))
        G3 = F.adaptive_avg_pool2d(F.sigmoid(G3),1)
        D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear')+G3[:, 0,:,:].unsqueeze(1).repeat(1,128,1,1)*T3)

        G2 = self.attention_feature4(torch.cat((E2,F.upsample(D3, size=E2.size()[2:], mode='bilinear')),1))
        G2 = F.adaptive_avg_pool2d(F.sigmoid(G2),1)
        D2 = self.output4(F.upsample(D3, size=E2.size()[2:], mode='bilinear')+G2[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T2)

        G1 = self.attention_feature5(torch.cat((E1,F.upsample(D2, size=E1.size()[2:], mode='bilinear')),1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1),1)
        D1 = self.output5(F.upsample(D2, size=E1.size()[2:], mode='bilinear')+G1[:, 0,:,:].unsqueeze(1).repeat(1,32,1,1)*T1)


        ################################Gated Parallel&Dual branch residual fuse#######################################
        R5 = self.dem1_1(T5)
        R4 = self.dem2_1(T4)
        R3 = self.dem3_1(T3)
        R2 = self.dem4_1(T2)
        R1 = self.dem5_1(T1)
        output_res = self.out_res(torch.cat((D1,F.upsample(G5[:, 1,:,:].unsqueeze(1).repeat(1,32,1,1)*R5,size=E1.size()[2:], mode='bilinear'),F.upsample(G4[:, 1,:,:].unsqueeze(1).repeat(1,32,1,1)*R4,size=E1.size()[2:], mode='bilinear'),F.upsample(G3[:, 1,:,:].unsqueeze(1).repeat(1,32,1,1)*R3,size=E1.size()[2:], mode='bilinear'),F.upsample(G2[:, 1,:,:].unsqueeze(1).repeat(1,32,1,1)*R2,size=E1.size()[2:], mode='bilinear'),F.upsample(G1[:, 1,:,:].unsqueeze(1).repeat(1,32,1,1)*R1,size=E1.size()[2:], mode='bilinear')),1))
        output_res = F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        output_fpn = F.upsample(D1, size=input.size()[2:], mode='bilinear')
        pre_sal = output_fpn+output_res
        #######################################################################
        if self.training:
            return output_fpn, pre_sal
        return F.sigmoid(pre_sal)

if __name__ == "__main__":
    model = GateNet()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    output = model(input)
