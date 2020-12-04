import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from config import train_data
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from model_GateNet_ResNet import GateNet,Bottleneck
from torch.backends import cudnn
from torch.utils import model_zoo
import torch.nn.functional as functional

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# vis = visdom.Visdom(env='train')
cudnn.benchmark = True
torch.manual_seed(2018)
torch.cuda.set_device(0)

##########################hyperparameters###############################
ckpt_path = './model'
exp_name = 'model_gatenet'
args = {
    'iter_num': 100000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}
##########################data augmentation###############################
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(384, 384),  # change to resize
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()
##########################################################################
train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)


###multi-scale-train
# train_set = ImageFolder_multi_scale(train_data, joint_transform, img_transform, target_transform)
# train_loader = DataLoader(train_set, collate_fn=train_set.collate, batch_size=args['train_batch_size'], num_workers=12, shuffle=True, drop_last=True)

criterion = nn.BCEWithLogitsLoss().cuda()
criterion_BCE = nn.BCELoss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_mae = nn.L1Loss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model = GateNet(Bottleneck, [3,4,6,3])
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    ##############################Optim setting###############################
    # model = GateNet()
    net = model.cuda().train()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


#########################################################################

def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record, loss3_record, loss4_record, loss5_record, loss6_record, loss7_record, loss8_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            # data\binarizing\Variable
            inputs, labels = data
            labels[labels > 0.5] = 1
            labels[labels != 1] = 0
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            output_fpn, output_final = net(inputs)
            ##########loss#############

            loss1 = criterion(output_fpn, labels)
            loss2 = criterion(output_final, labels)

            total_loss = loss1+loss2
            total_loss.backward()
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)

            #############log###############
            curr_iter += 1

            log = '[iter %d], [total loss %.5f],[loss1 %.5f],[loss1 %.5f],[lr %.13f] ' % \
                  (curr_iter, total_loss_record.avg, loss1_record.avg, loss2_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return
    ##      #############end###############


if __name__ == '__main__':
    main()
