import torch


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one
# output)
def bottle(f, x_tuple):
    # f: flownet
    # e.g. x_tuple: (x_img_list, ) -> x_img_list: [time, batch, 3, 2, H, W]
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))  # ([time, batch, 3, 2, H, W], )
    # process the size reshape for each tensor in x_tuple
    # the new batch_size = time x the old batch_size
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    # f([time x batch, 3, 2, H, W])

    if type(y) is tuple:
        y_size = y[0].size()
        return [_y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:]) for _y in y]
    else:
        y_size = y.size()
        # reshape the output into time x the old batch_size x features
        return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


def save_model(path, transition_model, pose_model, encoder, optimizer,
               epoch, metrics):
    states = {
        'transition_model': transition_model.state_dict(),
        'pose_model': pose_model.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(states, path)


def get_lr(optimizer):
    """
    currently only support optimizer with one param group
    -> please use multiple optimizers separately for multiple param groups
    """
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    assert len(lr_list) == 1
    return lr_list[0]
