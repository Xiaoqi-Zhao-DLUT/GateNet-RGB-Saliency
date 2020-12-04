import os
import os.path
import torch.utils.data as data
from PIL import Image
import random
import torch
from torch.nn.functional import interpolate
Image.MAX_IMAGE_PIXELS = 1000000000

def make_dataset(root):
    img_path = os.path.join(root, 'DUTS-TR-Image')
    gt_path = os.path.join(root, 'DUTS-TR-Mask')
    gt_path = os.path.join(root, 'gt')

    img_list = [os.path.splitext(f)[0]
                for f in os.listdir(gt_path) if f.endswith('.png')]
    return [(os.path.join(img_path, img_name + '.jpg'),
             os.path.join(gt_path, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')


        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_multi_scale(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


####可用可不用. GateNet论文中没有使用multi-scale to train
    def collate(self,batch):
        # size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        # size_list = [224, 256, 288, 320, 352]
        # size_list = [128, 160, 192, 224, 256]
        size_list = [128, 192, 256, 320, 384]
        size = random.choice(size_list)

        img, target = [list(item) for item in zip(*batch)]
        img = torch.stack(img, dim=0)
        img = interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
        target = torch.stack(target, dim=0)
        target = interpolate(target, size=(size, size), mode="bilinear")
        # print(img.shape)
        return img, target