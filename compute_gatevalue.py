import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config import test_data
from misc import check_mkdir
from model_GateNet_vgg16 import GateNet
from pylab import *
torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = ''
exp_name = ''


args = {
    'snapshot': '100000',
    'crf_refine':False,
    'save_results': True
}

img_transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()
to_test = {'test': test_data}


def main():
    net = GateNet().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path,exp_name, args['snapshot'] + '.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root, 'images')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.jpg')]
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
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img1 = Image.open(os.path.join(root,'images',img_name + '.jpg')).convert('RGB')
                img1 = img1.resize([384,384])
                img_var = Variable(img_transform(img1).unsqueeze(0), volatile=True).cuda()
                gate1,gate2,gate3,gate4,gate5 = net(img_var)
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
            print(sum1/len(img_list))
            print(sum1_res/len(img_list))
            print(sum2/len(img_list))
            print(sum2_res/len(img_list))
            print(sum3/len(img_list))
            print(sum3_res/len(img_list))
            print(sum4/len(img_list))
            print(sum4_res/len(img_list))
            print(sum5/len(img_list))
            print(sum5_res/len(img_list))

if __name__ == '__main__':
    main()
