import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from utils.config import duts,ecssd,hku_is,dut_omron,pascal_s,CAMO,CHAMELEON,COD10K,NC4K,CVC_300,CVC_ClinicDB,CVC_ColonDB,ETIS_LaribPolypDB,Kvasir,SBU,ucf,CUHK,DUT,ISTD,dutrgbd,njud,nlpr,stere,sip,rgbd135,ssd,lfsd,trans10k_easy,trans10k_hard,trans10k_all,ORSSD,EORSSD,ORSI_4199,GDD,MSD,weixin_data
from utils.misc import check_mkdir
# from model.gatenet_rgb_res50_pami import GateNet
# from model.gatenet_vgg16_pami import GateNet_v2
from model.GateNetv2_rgb_d_res50 import GateNetv2_two_stream_cross_modal_fusion_gate_decoder_gate
import ttach as tta

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = '/home/asus/'
exp_name = 'Coding/GateNet_PAMI/saved_model'
args = {
    'snapshot': 'GateNetv2_resnet50d_rgbdsod_train_strategy2_mstrain_batch16_100epoch_decay_epoch_60Model_100_gen',
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

to_test = {'DUT-RGBD':dutrgbd,'NJUD':njud,'NLPR':nlpr,'STERE':stere,'SIP':sip,'RGBD135':rgbd135,'SSD':ssd,'LFSD':lfsd}

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        # tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)



def main():
    t0 = time.time()
    net = GateNetv2_two_stream_cross_modal_fusion_gate_decoder_gate().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']+'.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()

    with torch.no_grad():
        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root,'depth_scale')
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            sum1 = 0
            sum1_res = 0
            sum2 = 0
            sum2_res = 0
            sum3 = 0
            sum3_res = 0
            sum4 = 0
            sum4_res = 0
            sum5 = 0
            sum5_res = 0
            print(name)
            for idx, img_name in enumerate(img_list):
                # print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
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
                img_resize = img.resize([352, 352], Image.BILINEAR)  # Foldconv cat是320
                depth_resize = depth.resize([352, 352], Image.BILINEAR)  # Foldconv cat是320
                img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cuda()
                depth_var = Variable(depth_transform(depth_resize).unsqueeze(0), volatile=True).cuda()

                gate1,gate2,gate3,gate4,gate5 = net(img_var,depth_var)
                sum1 += gate1[:, 0, :, :]
                sum1_res += gate1[:, 1, :, :]
                sum2 += gate2[:, 0, :, :]
                sum2_res += gate2[:, 1, :, :]
                sum3 += gate3[:, 0, :, :]
                sum3_res += gate3[:, 1, :, :]
                sum4 += gate4[:, 0, :, :]
                sum4_res += gate4[:, 1, :, :]
                sum5 += gate5[:, 0, :, :]
                sum5_res += gate5[:, 1, :, :]
            print('G1_0: %.4f' %(sum1/len(img_list)))
            print('G2_0: %.4f' % (sum2 / len(img_list)))
            print('G3_0: %.4f' % (sum3 / len(img_list)))
            print('G4_0: %.4f' % (sum4 / len(img_list)))
            print('G5_0: %.4f' % (sum5 / len(img_list)))
            print('G1_1: %.4f' %(sum1_res/len(img_list)))
            print('G2_1: %.4f' %(sum2_res/len(img_list)))
            print('G3_1: %.4f' %(sum3_res/len(img_list)))
            print('G4_1: %.4f' %(sum4_res/len(img_list)))
            print('G5_1: %.4f' %(sum5_res/len(img_list)))

if __name__ == '__main__':
    main()
