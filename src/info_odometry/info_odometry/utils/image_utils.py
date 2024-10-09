from os.path import splitext
import numpy as np
from PIL import Image


class ImageUtils:
    def __init__(self):
        pass

    @staticmethod
    def normalize_image_feature(img_pair, rgb_max):
        # input -> img_pair: [batch, 3, 2, h, w]
        # output -> img_feature: [batch, 6, 480, 752]
        rgb_mean = img_pair.contiguous().view(img_pair.size()[:2] + (-1,)).mean(dim=-1).view(
            img_pair.size()[:2] + (1, 1, 1,))  # [batch, 3, 1, 1, 1]
        img_pair = (img_pair - rgb_mean) / rgb_max  # [batch, 3, 2, 480, 752], normalized
        x1 = img_pair[:, :, 0, :, :]
        x2 = img_pair[:, :, 1, :, :]
        img_features = torch.cat((x1, x2), dim=1)  # [batch, 6, 480, 752]
        return img_features

    @staticmethod
    def read_gen(file_name):
        ext = splitext(file_name)[-1]
        if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
            im = np.array(Image.open(file_name))
            # if im.shape[2] > 3:
            #     return im[:,:,:3]
            # else:
            #     return im
            return im
        elif ext == '.bin' or ext == '.raw':
            return np.load(file_name)
        elif ext == '.flo':
            return flow_utils.readFlow(file_name).astype(np.float32)
        return []

