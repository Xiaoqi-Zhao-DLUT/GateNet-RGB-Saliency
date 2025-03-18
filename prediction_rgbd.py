import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from utils.config import duts,ecssd,hku_is,dut_omron,pascal_s,CAMO,CHAMELEON,COD10K,CVC_300,CVC_ClinicDB,CVC_ColonDB,ETIS_LaribPolypDB,Kvasir,SBU,ucf,dutrgbd,njud,nlpr,stere,sip,rgbd135,ssd,lfsd
from utils.misc import check_mkdir
from model.GateNetv2_rgb_d_res50 import GateNetv2_two_stream
import cv2

import ttach as tta
import torch.nn.functional as F
torch.manual_seed(2018)
torch.cuda.set_device(0)
#os.path.join(ckpt_path, exp_name, args['snapshot']+'.pth')
ckpt_path = '' # model path
exp_name = '' # model path
args = {
    'snapshot': 'GateNetv2_tasknameXX',
    'crf_refine': False,
    'save_results': True
}



img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# to_test = {'DUTS':duts,'DUT_OMORN':dut_omron,'ECSSD':ecssd,'HKU-IS':hku_is,'PASCAL-S':pascal_s}
# to_test = {'CAMO':CAMO,'CHAMELEON':CHAMELEON,'COD10K':COD10K}
# to_test = {'CVC_300':CVC_300,'CVC_ColonDB':CVC_ColonDB,'CVC_ClinicDB':CVC_ClinicDB,'ETIS_LaribPolypDB':ETIS_LaribPolypDB,'Kvasir':Kvasir}
# to_test = {'SBU':SBU,'UCF':ucf}
# to_test = {'trans10k-easy':trans10k}
to_test = {'DUT-RGBD':dutrgbd,'NJUD':njud,'NLPR':nlpr,'STERE':stere,'SIP':sip,'RGBD135':rgbd135,'SSD':ssd,'LFSD':lfsd}
# to_test = {'SSD':ssd}
# to_test = {'LFSD':lfsd}
# test_datasets = {'easy':trans10k_easy,'hard':trans10k_hard,'all':trans10k_all}
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        # tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)

def main():
    t0 = time.time()
    net = GateNetv2_two_stream().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']+'.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()
    with torch.no_grad():

        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root,'depth_scale')
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            for idx, img_name in enumerate(img_list):

                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root, 'RGB', img_name[0] + '.png')
                rgb_jpg_path = os.path.join(root, 'RGB', img_name[0] + '.jpg')
                depth_jpg_path = os.path.join(root, 'depth_scale', img_name[0] + '.jpg')
                depth_png_path = os.path.join(root, 'depth_scale', img_name[0] + '.png')
                if os.path.exists(rgb_png_path):
                    img = Image.open(rgb_png_path).convert('RGB')
                else:
                    img = Image.open(rgb_jpg_path).convert('RGB')
                if os.path.exists(depth_jpg_path):
                    depth = Image.open(depth_jpg_path).convert('L')
                else:
                    depth = Image.open(depth_png_path).convert('L')


                w_,h_ = img.size
                img_resize = img.resize([352,352],Image.BILINEAR)  # Foldconv cat是320
                depth_resize = depth.resize([352,352],Image.BILINEAR)  # Foldconv cat是320
                img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cuda()
                depth_var = Variable(depth_transform(depth_resize).unsqueeze(0), volatile=True).cuda()
                n, c, h, w = img_var.size()
                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()

                    rgb_trans = transformer.augment_image(img_var)
                    d_trans = transformer.augment_image(depth_var)
                    model_output = net(rgb_trans,d_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)

                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()

                res = F.upsample(prediction, size=[h_, w_], mode='bilinear', align_corners=False)
                res = res.data.cpu().numpy().squeeze()
                res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)


                if args['save_results']:
                    check_mkdir(os.path.join(ckpt_path, exp_name,args['snapshot']+'epoch',name))
                    cv2.imwrite(
                        os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', name, img_name[0] + '.png'), res)





if __name__ == '__main__':
    main()
