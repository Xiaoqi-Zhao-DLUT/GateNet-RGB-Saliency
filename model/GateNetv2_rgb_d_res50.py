#!/usr/bin/python3
#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from model.FoldConv import FoldConv_aspp,FoldConv_denseaspp
import torch
import timm
from thop import clever_format
from thop import profile


class GateNetv2_two_stream(nn.Module):

    def __init__(self):
        super(GateNetv2_two_stream, self).__init__()
        self.bkbone_rgb = timm.create_model('resnet50d', features_only=True, pretrained=True)
        self.bkbone_d = timm.create_model('resnet50d', features_only=True, pretrained=True)


        self.E5_rgbd = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=1), nn.BatchNorm2d(2048), nn.ReLU(inplace=True))
        self.E4_rgbd = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
        self.E3_rgbd = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.E2_rgbd = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.E1_rgbd = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.attention_feature_G1_cm = nn.Sequential(nn.Conv2d(64 + 64, 2, kernel_size=3, padding=1))
        self.attention_feature_G2_cm = nn.Sequential(nn.Conv2d(256 + 256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature_G3_cm = nn.Sequential(nn.Conv2d(512 + 512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature_G4_cm = nn.Sequential(nn.Conv2d(1024 + 1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature_G5_cm = nn.Sequential(nn.Conv2d(2048 + 2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        ################################Gate#######################################
        self.attention_feature5 = nn.Sequential(nn.Conv2d(64 + 64, 2, kernel_size=3, padding=1))
        self.attention_feature4 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        ###############################Transition Layer########################################
        self.dem1 = FoldConv_aspp(in_channel=2048,
                                  out_channel=64,
                                  out_size=384 // 16,
                                  kernel_size=3,
                                  stride=1,
                                  padding=2,
                                  dilation=2,
                                  win_size=2,
                                  win_padding=0,
                                  )
        self.dem1_bn = nn.BatchNorm2d(64)
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))

        self.T54321_5 = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.T54321_4 = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.T54321_3 = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.T54321_2 = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.T54321_1 = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output5 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        ################################Parallel branch#######################################
        self.out_res = nn.Sequential(nn.Conv2d(64 + 64 + 64 + 64 + 64 + 1, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 1, kernel_size=3, padding=1))

    def forward(self, x,depth):
        input = x
        depth = torch.cat([depth,depth,depth],1)
        B, _, _, _ = input.size()
        E1_rgb, E2_rgb, E3_rgb, E4_rgb, E5_rgb = self.bkbone_rgb(x)
        E1_d, E2_d, E3_d, E4_d, E5_d = self.bkbone_d(depth)
        ################################Transition Layer#######################################
        # print(E5.shape)
        G5_cm = self.attention_feature_G5_cm(torch.cat((E5_rgb, E5_d), 1))
        G5_cm = F.adaptive_avg_pool2d(F.sigmoid(G5_cm), 1)
        G4_cm = self.attention_feature_G4_cm(torch.cat((E4_rgb, E4_d), 1))
        G4_cm = F.adaptive_avg_pool2d(F.sigmoid(G4_cm), 1)
        G3_cm = self.attention_feature_G3_cm(torch.cat((E3_rgb, E3_d), 1))
        G3_cm = F.adaptive_avg_pool2d(F.sigmoid(G3_cm), 1)
        G2_cm = self.attention_feature_G2_cm(torch.cat((E2_rgb, E2_d), 1))
        G2_cm = F.adaptive_avg_pool2d(F.sigmoid(G2_cm), 1)
        G1_cm = self.attention_feature_G1_cm(torch.cat((E1_rgb, E1_d), 1))
        G1_cm = F.adaptive_avg_pool2d(F.sigmoid(G1_cm), 1)

        E5 = self.E5_rgbd(G5_cm[:, 0, :, :].unsqueeze(1).repeat(1, 2048, 1, 1) * E5_rgb + G5_cm[:, 1, :, :].unsqueeze(1).repeat(1, 2048, 1, 1) * E5_d)
        E4 = self.E4_rgbd(G4_cm[:, 0, :, :].unsqueeze(1).repeat(1, 1024, 1, 1) * E4_rgb + G4_cm[:, 1, :, :].unsqueeze(1).repeat(1, 1024, 1, 1) * E4_d)
        E3 = self.E3_rgbd(G3_cm[:, 0, :, :].unsqueeze(1).repeat(1, 512, 1, 1) * E3_rgb + G3_cm[:, 1, :, :].unsqueeze(1).repeat(1, 512, 1, 1) * E3_d)
        E2 = self.E2_rgbd(G2_cm[:, 0, :, :].unsqueeze(1).repeat(1, 256, 1, 1) * E2_rgb + G2_cm[:, 1, :, :].unsqueeze(1).repeat(1, 256, 1, 1) * E2_d)
        E1 = self.E1_rgbd(G1_cm[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * E1_rgb + G1_cm[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * E1_d)
        T5 = F.relu(self.dem1_bn(self.dem1(E5)))
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)

        T54321_5 = self.T54321_5(torch.cat((T5, F.upsample(T4, size=T5.size()[2:], mode='bilinear'),
                                            F.upsample(T3, size=T5.size()[2:], mode='bilinear'),
                                            F.upsample(T2, size=T5.size()[2:], mode='bilinear'),
                                            F.upsample(T1, size=T5.size()[2:], mode='bilinear')), 1))
        T54321_4 = self.T54321_4(torch.cat((T4, F.upsample(T5, size=T4.size()[2:], mode='bilinear'),
                                            F.upsample(T3, size=T4.size()[2:], mode='bilinear'),
                                            F.upsample(T2, size=T4.size()[2:], mode='bilinear'),
                                            F.upsample(T1, size=T4.size()[2:], mode='bilinear')), 1))
        T54321_3 = self.T54321_3(torch.cat((T3, F.upsample(T5, size=T3.size()[2:], mode='bilinear'),
                                            F.upsample(T4, size=T3.size()[2:], mode='bilinear'),
                                            F.upsample(T2, size=T3.size()[2:], mode='bilinear'),
                                            F.upsample(T1, size=T3.size()[2:], mode='bilinear')), 1))
        T54321_2 = self.T54321_2(torch.cat((T2, F.upsample(T5, size=T2.size()[2:], mode='bilinear'),
                                            F.upsample(T4, size=T2.size()[2:], mode='bilinear'),
                                            F.upsample(T3, size=T2.size()[2:], mode='bilinear'),
                                            F.upsample(T1, size=T2.size()[2:], mode='bilinear')), 1))
        T54321_1 = self.T54321_1(torch.cat((T1, F.upsample(T5, size=T1.size()[2:], mode='bilinear'),
                                            F.upsample(T4, size=T1.size()[2:], mode='bilinear'),
                                            F.upsample(T3, size=T1.size()[2:], mode='bilinear'),
                                            F.upsample(T2, size=T1.size()[2:], mode='bilinear')), 1))
        ################################Gated FPN#######################################
        G5 = self.attention_feature1(torch.cat((T54321_5, T5), 1))
        G5 = F.adaptive_avg_pool2d(F.sigmoid(G5), 1)
        D5 = self.output1(G5[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T5)

        G4 = self.attention_feature2(
            torch.cat((T54321_4, F.upsample(D5, size=T54321_4.size()[2:], mode='bilinear')), 1))
        G4 = F.adaptive_avg_pool2d(F.sigmoid(G4), 1)
        D4 = self.output2(
            F.upsample(D5, size=E4.size()[2:], mode='bilinear') + G4[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T4)

        G3 = self.attention_feature3(
            torch.cat((T54321_3, F.upsample(D4, size=T54321_3.size()[2:], mode='bilinear')), 1))
        G3 = F.adaptive_avg_pool2d(F.sigmoid(G3), 1)
        D3 = self.output3(
            F.upsample(D4, size=E3.size()[2:], mode='bilinear') + G3[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T3)

        G2 = self.attention_feature4(
            torch.cat((T54321_2, F.upsample(D3, size=T54321_2.size()[2:], mode='bilinear')), 1))
        G2 = F.adaptive_avg_pool2d(F.sigmoid(G2), 1)
        D2 = self.output4(
            F.upsample(D3, size=E2.size()[2:], mode='bilinear') + G2[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T2)

        G1 = self.attention_feature5(
            torch.cat((T54321_1, F.upsample(D2, size=T54321_1.size()[2:], mode='bilinear')), 1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1), 1)
        D1 = self.output5(
            F.upsample(D2, size=E1.size()[2:], mode='bilinear') + G1[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T1)
        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn = F.upsample(D1, size=input.size()[2:], mode='bilinear')
        output_res = self.out_res(torch.cat((D1, F.upsample(G5[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T5,
                                                            size=E1.size()[2:], mode='bilinear'),
                                             F.upsample(G4[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T4,
                                                        size=E1.size()[2:], mode='bilinear'),
                                             F.upsample(G3[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T3,
                                                        size=E1.size()[2:], mode='bilinear'),
                                             F.upsample(G2[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T2,
                                                        size=E1.size()[2:], mode='bilinear'),
                                             F.upsample(G1[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T1,
                                                        size=E1.size()[2:], mode='bilinear')), 1))
        output_res = F.upsample(output_res, size=input.size()[2:], mode='bilinear')
        pre_sal = output_fpn + output_res
        #######################################################################
        if self.training:
            return output_fpn, pre_sal
        return pre_sal


if __name__ == "__main__":
    model = GateNetv2_two_stream().cuda()
    RGB = torch.randn(1, 3, 352, 352).cuda()
    D = torch.randn(1, 3, 352, 352).cuda()
    # out = model(RGB,D)

    flops, params = profile(model,inputs=(RGB,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)
