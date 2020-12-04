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
    'snapshot': '',
    'crf_refine':False,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test = {'test':test_data}

def main():
    t0 = time.time()
    net = RGB_sal().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()

    with torch.no_grad():

        for name, root in to_test.items():

            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root,'images')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.jpg')]
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img1 = Image.open(os.path.join(root,'images',img_name + '.jpg')).convert('RGB')
                img1 = img1.resize([384,384])
                img_var = Variable(img_transform(img1).unsqueeze(0), volatile=True).cuda()
                output1,output2,output3,output4 = net(img_var)
                a = [output1,output2,output3,output4]
                for i in range(len(a)):
                    visualize_feature_map(a[i],img_name,i)

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch,img_name,num):
    print(img_batch.size()[0:])

    feature_map = torch.squeeze(img_batch, 0)
    print(feature_map.shape)
    if(len(feature_map.size())==2):
        feature_map = torch.unsqueeze(feature_map,0)


    feature_map_combination = []
    num_pic = feature_map.shape[0]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)

    feature_map_sum = sum(ele for ele in feature_map_combination)
    feature_map_sum = feature_map_sum.cuda().data.cpu()
    plt.imshow(feature_map_sum)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('/path...'+img_name+'_'+str(num)+".png", bbox_inches='tight', dpi=18, pad_inches=0.0)

if __name__ == '__main__':
    main()
