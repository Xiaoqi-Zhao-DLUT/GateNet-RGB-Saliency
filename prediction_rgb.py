import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from utils.config import duts,ecssd,hku_is,dut_omron,pascal_s,CAMO,CHAMELEON,COD10K,NC4K,CVC_300,CVC_ClinicDB,CVC_ColonDB,ETIS_LaribPolypDB,Kvasir,SBU,ucf,CUHK,DUT,ISTD,dutrgbd,njud,nlpr,stere,sip,rgbd135,ssd,lfsd,trans10k_easy,trans10k_hard,trans10k_all,ORSSD,EORSSD,ORSI_4199,GDD,MSD
from utils.misc import check_mkdir
from model.GateNetv2_rgb_res50 import GateNetv2
import ttach as tta
import torch.nn.functional as F
import cv2
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

# to_test = {'DUTS':duts,'DUT-OMRON':dut_omron,'ECSSD':ecssd,'HKU-IS':hku_is,'PASCAL-S':pascal_s}
# to_test = {'PASCAL-S':pascal_s}
# to_test = {'CAMO':CAMO,'CHAMELEON':CHAMELEON,'COD10K':COD10K,'NC4K':NC4K}
# to_test = {'CVC-300':CVC_300,'CVC-ColonDB':CVC_ColonDB,'CVC-ClinicDB':CVC_ClinicDB,'ETIS-LaribPolypDB':ETIS_LaribPolypDB,'Kvasir':Kvasir}
# to_test = {'SBU':SBU,'UCF':ucf}
# to_test = {'CUHK':CUHK,'DUT':DUT}
# to_test = {'ISTD':ISTD}
# to_test = {'DUT-RGBD_depth':dutrgbd,'NJUD_depth':njud,'NLPR_depth':nlpr,'STERE_depth':stere,'SIP_depth':sip,'LFSD_depth':lfsd}
# to_test = {'NLPR_sal_jpg':nlpr,'NJUD_sal_jpg':njud,'DUT-RGBD_sal_jpg':dutrgbd,'STERE_sal_jpg':stere,'SIP_sal_jpg':sip,'LFSD_sal_jpg':lfsd}
# to_test = {'NJUD_depth_jpg_yasuo':njud,'NLPR_depth_jpg_yasuo':nlpr,'DUT-RGBD_depth_jpg_yasuo':dutrgbd,'STERE_depth_jpg_yasuo':stere,'SIP_depth_jpg_yasuo':sip,'LFSD_depth_jpg_yasuo':lfsd}
# to_test = {'NJUD_depth_jpg_yasuo':njud}
# to_test = {'DUT-RGBD_sal_jpg':dutrgbd,'NJUD_jpg':njud,'NLPR_depth':nlpr,'STERE_depth':stere,'SIP_depth':sip,'LFSD_depth':lfsd}
# to_test = {'DUT-RGBD_contour':dutrgbd,'NJUD_contour':njud,'NLPR_contour':nlpr,'STERE_contour':stere,'SIP_contour':sip,'LFSD_contour':lfsd}
# to_test = {'DUT-RGBD_contour_scale':dutrgbd}
# to_test = {'LFSD_depth':lfsd}
# to_test = {'SIP_depth':sip}
# to_test = {'DUTS_Depth':duts}
# to_test = {'DUTS-TR':duts}
# to_test = {'DUT-RGBD_depth':dutrgbd}
# to_test = {'easy':trans10k_easy,'hard':trans10k_hard,'all':trans10k_all}
# to_test = {'easy':trans10k_easy,'hard':trans10k_hard,'all':trans10k_all}
to_test = {'EORSSD':EORSSD}
# to_test = {'GDD':GDD}
# to_test = {'MSD':MSD}
# to_test = {'weixin_test':weixin_data}
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        # tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)

def main():
    t0 = time.time()
    net = GateNetv2().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']+'.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()

    with torch.no_grad():

        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root,'images')
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            for idx, img_name in enumerate(img_list):

                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root, 'images', img_name[0] + '.png')
                rgb_jpg_path = os.path.join(root, 'images', img_name[0] + '.jpg')
                rgb_bmp_path = os.path.join(root, 'images', img_name[0] + '.bmp')
                if os.path.exists(rgb_png_path):
                    img = Image.open(rgb_png_path).convert('RGB')
                elif os.path.exists(rgb_jpg_path):
                    img = Image.open(rgb_jpg_path).convert('RGB')
                else:
                    img = Image.open(rgb_bmp_path).convert('RGB')
                w_,h_ = img.size
                img_resize = img.resize([352,352],Image.BILINEAR)  # Foldconv cat是320
                img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cuda()
                n, c, h, w = img_var.size()

                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()

                    rgb_trans = transformer.augment_image(img_var)
                    model_output = net(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)



                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()
                # prediction = to_pil(prediction.data.squeeze(0).cpu())
                # prediction = prediction.resize((w_, h_), Image.BILINEAR)

                res = F.upsample(prediction, size=[h_, w_], mode='bilinear', align_corners=False)
                res = res.data.cpu().numpy().squeeze()
                res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
                if args['save_results']:
                    check_mkdir(os.path.join(ckpt_path, exp_name,args['snapshot']+'epoch',name))
                    cv2.imwrite(os.path.join(ckpt_path, exp_name ,args['snapshot']+'epoch',name, img_name[0] + '.png'), res)




if __name__ == '__main__':
    main()
