from collections import OrderedDict
from .sync_batchnorm import convert_model
import torch
from .DeeplabV2 import *

def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = False

def release_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = True

def init_model(cfg):

    model = Res_Deeplab(num_classes = cfg.num_classes).cuda()

    if cfg.fixbn:
        freeze_bn(model)
    else:
        release_bn(model)


    if cfg.model=='deeplab' and cfg.init_weight != 'None':
        params = torch.load(cfg.init_weight)
        print('Model restored with weights from : {}'.format(cfg.init_weight))
        if 'init-' in cfg.init_weight and cfg.model=='deeplab':
            new_params = model.state_dict().copy()
            for i in params:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = params[i]
            model.load_state_dict(new_params, strict=True)

        else:
            new_params = model.state_dict().copy()
            for i in params:
                i_parts = i.split('.')[0]
                if not i_parts == 'layer5':
                    new_params[i] = params[i]
            model.load_state_dict(new_params, strict=True)

    if cfg.restore_from != 'None':
        params = torch.load(cfg.restore_from)
        model.load_state_dict(params)
        print('Model initialize with weights from : {}'.format(cfg.restore_from))

    if cfg.multigpu:
        model = convert_model(model)
        model = nn.DataParallel(model)
    if cfg.train:
        model.train().cuda()
        print('Mode --> Train')
    else:
        model.eval().cuda()
        print('Mode --> Eval')
    return model

