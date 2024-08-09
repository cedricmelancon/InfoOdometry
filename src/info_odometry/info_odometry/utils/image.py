import torch


def normalize_img_feat(img_pair, rgb_max):
    # input -> img_pair: [batch, 3, 2, h, w]
    # output -> img_feature: [batch, 6, 480, 752]
    rgb_mean = img_pair.contiguous().view(img_pair.size()[:2] + (-1,)).mean(dim=-1).view(
        img_pair.size()[:2] + (1, 1, 1,))  # [batch, 3, 1, 1, 1]
    img_pair = (img_pair - rgb_mean) / rgb_max  # [batch, 3, 2, 480, 752], normalized
    x1 = img_pair[:, :, 0, :, :]
    x2 = img_pair[:, :, 1, :, :]
    img_features = torch.cat((x1, x2), dim=1)  # [batch, 6, 480, 752]
    return img_features
