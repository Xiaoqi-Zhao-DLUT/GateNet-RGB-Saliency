import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config import test_data
from misc import check_mkdir, crf_refine
from model_GateNet_ResNet import GateNet
import torch.nn.functional as F
#import ttach as tta
torch.manual_seed(2018)
torch.cuda.set_device(0)
import time

ckpt_path = ''
exp_name = ''

args = {
    'snapshot': '',
    'crf_refine': False,
    'save_results': True
}

img_transform = transforms.Compose([
    # transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test = {'test':test_data}


Image.MAX_IMAGE_PIXELS = 1000000000

def main():
    #########################Load##########################
    net = GateNet().cuda()
    # net = GateNet_SIM_Light().cuda() # vgg16
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']),map_location={'cuda:1': 'cuda:1'}))
    net.eval()

    # transforms = tta.Compose(
    #     [
    #         tta.HorizontalFlip(),
    #         tta.Scale(scales=[1,1.5],interpolation='bilinear',align_corners=False),
    #     ]
    # )
    #
    # net = tta.SegmentationTTAWrapper(net, transforms, merge_mode='mean')
    ########################################################
    with torch.no_grad():

        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root,'GT')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.png')]
            print(img_list)
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img1 = Image.open(os.path.join(root,'Imgs/'+img_name +'.jpg')).convert('RGB')
                img = img1
                w,h = img1.size
                img1 = img1.resize([384,384],Image.BILINEAR)
                img_var = Variable(img_transform(img1).unsqueeze(0), volatile=True).cuda()
                prediction = net(img_var)
                # prediction = F.sigmoid(prediction)
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                # prediction = prediction.resize((w, h), Image.BILINEAR)
                prediction = prediction.resize((w, h), Image.NEAREST)
                if args['crf_refine']:
                    prediction = crf_refine(np.array(img), np.array(prediction))
                prediction = np.array(prediction)
                if args['save_results']:
                    Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, 'DUTS', img_name + '.png'))

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)