import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss: str) -> nn.Module:
    """Return loss from its name

    Args:
        loss: The name of loss function

    Examples:

    >>> loss = get_loss('mse')

    """

    if loss == 'mae':
        loss_fun = nn.L1Loss()
    elif loss == 'smooth_mae':
        loss_fun = nn.SmoothL1Loss()
    elif loss == 'mse':
        loss_fun = nn.MSELoss()
    elif loss == 'mse+mae':
        loss_fun = nn.MSELoss() + nn.L1Loss()  # for regression task
    elif loss == 'ce':
        loss_fun = nn.CrossEntropyLoss()
    else:
        raise Exception("loss function is not correct " + loss)
    return loss_fun
