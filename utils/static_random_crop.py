import random

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        if len(img.shape) == 2:
            return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw)]
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]