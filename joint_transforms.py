import numbers
import random

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 1000000000


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


# class RandomCrop(object):
#     def __init__(self, size, padding=0):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.padding = padding
#
#     def __call__(self, img, mask):
#         if self.padding > 0:
#             img = ImageOps.expand(img, border=self.padding, fill=0)
#             mask = ImageOps.expand(mask, border=self.padding, fill=0)
#
#         assert img.size == mask.size
#         w, h = img.size
#         th, tw = self.size
#         if w == tw and h == th:
#             return img, mask
#         if w < tw or h < th:
#             return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)
#
#         x1 = random.randint(0, w - tw)
#         y1 = random.randint(0, h - th)
#         return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomCrop(object):
    def __init__(self, size,size1, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size1))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)
        return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomVerticallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask




class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)
