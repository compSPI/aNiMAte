import torch
from ctf_utils import primal_to_fourier_2D

#taken from https://github.com/yuta-hi/pytorch_similarity/blob/e4e2bde2be97221d608d1b9292ce8d9122741dc3/torch_similarity/functional.py#L35
def normalized_cross_correlation(x, y, return_map=False, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map


def fill_loss_dict(data_loss, model_output):
    latent_pred = model_output['latent_code']
    pred_nma_coords = model_output['nma_coords']
    nma_eigvals = model_output['nma_eigvals']
    mu = model_output['latent_mu']
    logvar = model_output['latent_logvar']
    masks = model_output['masks']
    if masks is not None:
        data_loss *= masks
    loss_dict = {'data_term': data_loss.mean()}

    if pred_nma_coords is not None and nma_eigvals is not None:
        nma_reg_term = (pred_nma_coords * nma_eigvals) ** 2
        loss_dict.update({'nma_reg_term': nma_reg_term.mean()})
    if mu is not None and logvar is not None:
        kld_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        loss_dict.update({'kld_term': kld_loss.mean()})
    return loss_dict


def real_proj_l2_loss(model_output, gt):
    
    proj_pred = model_output['proj']
    proj_gt = gt['proj']

    data_loss = (proj_pred - proj_gt) ** 2

    return fill_loss_dict(data_loss, model_output)


def real_proj_l1_loss(model_output, gt):
    proj_pred = model_output['proj']
    proj_gt = gt['proj']

    data_loss = torch.abs(proj_pred - proj_gt)
    return fill_loss_dict(data_loss, model_output)


def complex_proj_l2_loss(model_output, gt):
    proj_pred_real = model_output['proj']
    proj_pred_imag = model_output['proj_imag']
    proj_gt = gt['proj']

    data_loss = (proj_pred_real - proj_gt) ** 2 + proj_pred_imag ** 2

    return fill_loss_dict(data_loss, model_output)

def complex_proj_cc_loss(model_output, gt):
    proj_pred_real = model_output['proj']
    proj_pred_imag = model_output['proj_imag']
    proj_gt = gt['proj']

    data_loss = (1 - normalized_cross_correlation(proj_pred_real, proj_gt)) + proj_pred_imag ** 2

    return fill_loss_dict(data_loss, model_output)

def complex_fourier_l2_loss(model_output, gt):
    fproj_pred = model_output['fproj']
    fproj_gt = primal_to_fourier_2D(gt['proj'])

    data_loss = (torch.abs(fproj_pred - fproj_gt)) ** 2

    return fill_loss_dict(data_loss, model_output)

def complex_proj_l1_loss(model_output, gt):
    proj_pred_real = model_output['proj']
    proj_pred_imag = model_output['proj_imag']
    proj_gt = gt['proj']

    data_loss = torch.abs(proj_pred_real - proj_gt) + proj_pred_imag ** 2

    return fill_loss_dict(data_loss, model_output)
