import numpy as np
from torch.utils.data import DataLoader
from . import joint_transforms
from .base_dataset import BaseDataSet
import torchvision.transforms as standard_transforms
from . import transforms
from torch.utils import data
import torchvision.transforms.functional as TF

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def init_concat_dataset(cfg, plabel_path=None,source_plabel_path=None, selected=None, centroid=None, wei_path=None,fuse=False, target_selected=None, source_list='none', target_list='none'):
    source_env = cfg[cfg.source]
    target_env = cfg[cfg.target]
    cfg.num_classes=cfg.num_classes
    cfg.source_size = source_env.input_size
    cfg.target_size = target_env.input_size
    cfg.source_data_dir  = source_env.data_dir
    cfg.source_data_list = source_env.data_list
    cfg.target_data_dir  = target_env.data_dir
    cfg.target_data_list = target_env.data_list
    
    source_joint_list = [       
            joint_transforms.RandomSizeAndCrop(cfg.crop_src,
                                                True,
                                                scale_min=cfg.scale_min,
                                                scale_max=cfg.scale_max,
                                                pre_size=cfg.input_src,
                                                rec=cfg.rec
                                                ),
            joint_transforms.Resize(cfg.crop_src)
            ]

    target_joint_list = [
            joint_transforms.RandomSizeAndCrop(cfg.crop_tgt,
                                           True,
                                           scale_min=cfg.scale_min,
                                           scale_max=cfg.scale_max,
                                           pre_size=cfg.input_tgt,
                                           rec=cfg.rec
                                           ),
            joint_transforms.Resize(cfg.crop_tgt)
            ]

    if cfg.mirror:
        source_joint_list.append(joint_transforms.RandomHorizontallyFlip())
        target_joint_list.append(joint_transforms.RandomHorizontallyFlip())


    target_joint_transform = joint_transforms.Compose(target_joint_list)
    source_joint_transform = joint_transforms.Compose(source_joint_list)

    train_transform_list = [
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
            ]

    train_transform = standard_transforms.Compose(train_transform_list)
    label_transform = transforms.MaskToTensor()
    if selected is None:
        src_prop=cfg.src_prop
    else:
        src_prop=None
        
    source_dataset = BaseDataSet(source_env.data_dir, source_env.data_list, 
                        cfg.source, cfg.num_classes,
                        max_prop=src_prop,
                        joint_transform =  source_joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        selected=selected,
                        centroid = centroid,
                        wei_path = wei_path, 
                        set='train', plabel_path=source_plabel_path)
    target_dataset = BaseDataSet(target_env.data_dir, target_env.data_list,
                        cfg.target, cfg.num_classes,
                        joint_transform =  target_joint_transform,
                        transform = train_transform,
                        selected=target_selected,
                        label_transform = label_transform,
                        set='train', plabel_path=plabel_path)
    mixtrainset = data.ConcatDataset([source_dataset, target_dataset])
    mix_trainloader = data.DataLoader(mixtrainset, batch_size=cfg.batch_size, shuffle=True,
                                                          num_workers=cfg.worker, pin_memory=True, drop_last=True)

    return mix_trainloader, cfg    


def init_source_dataset(cfg, plabel_path=None, selected=None, fuse=False, source_list=None):
    source_env = cfg[cfg.source]
    target_env = cfg[cfg.target]
    if cfg.source=='synthia':
        cfg.num_classes=16
    else:
        cfg.num_classes=19
    cfg.source_size = source_env.input_size
    cfg.target_size = target_env.input_size
    cfg.source_data_dir  = source_env.data_dir
    cfg.source_data_list = source_env.data_list
    cfg.target_data_dir  = target_env.data_dir
    cfg.target_data_list = target_env.data_list
    source_joint_list = [       
            joint_transforms.RandomSizeAndCrop(cfg.crop_src,
                                                True,
                                                scale_min=cfg.scale_min,
                                                scale_max=cfg.scale_max,
                                                pre_size=cfg.input_src
                                                ),
            joint_transforms.Resize(cfg.crop_src)
            ]


    if cfg.mirror:
        source_joint_list.append(joint_transforms.RandomHorizontallyFlip())


    source_joint_transform = joint_transforms.Compose(source_joint_list)

    train_transform_list = [
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
            ]

    train_transform = standard_transforms.Compose(train_transform_list)
    label_transform = transforms.MaskToTensor()

    if source_list is None:
        source_list = source_env.data_list
    trainloader = data.DataLoader(
            BaseDataSet(source_env.data_dir, source_list, cfg.source, cfg.num_classes,
                        max_iters=cfg.num_steps*cfg.batch_size,
                        joint_transform =  source_joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        set='train'),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True)

    return trainloader,cfg

def init_test_dataset(config, dataset_name, set, selected=None, prop=None, label_ori=None, fuse=False, batchsize=1, list_path='none'):
    env = config[dataset_name]
    max_prop = None
    if dataset_name=='gta5' and set=='train':
        max_prop = config.pool_prop
    if dataset_name=='synthia' and set=='train':
        max_prop = config.pool_prop

    if list_path != 'none':
        data_list = list_path
        max_prop=None
    else:
        data_list = env.data_list

    if prop is not None:
        max_prop = prop
    if selected is not None:
        max_prop=None
    
    if label_ori is None:
        if dataset_name == 'gta5' or dataset_name=='synthia':
            label_ori=False
        else:
            label_ori=True

    if not label_ori:
        joint_transform = [joint_transforms.Resize((1024, 512))]
        joint_transform = joint_transforms.Compose(joint_transform)
        transform_list = [
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
            ]
    else:
        joint_transform = None
        transform_list = [
            transforms.Resize((1024, 512)),
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
            ]

    train_transform = standard_transforms.Compose(transform_list)

    if label_ori and dataset_name=='gta5':
        label_transform = [transforms.ResizeLabel((1914, 1052)),
                transforms.MaskToTensor()]
        label_transform = standard_transforms.Compose(label_transform)
    elif label_ori and dataset_name=='synthia':
        label_transform = [transforms.ResizeLabel((1280, 760)),
                transforms.MaskToTensor()]
        label_transform = standard_transforms.Compose(label_transform)
    else:
        label_transform = transforms.MaskToTensor()
    targetloader = data.DataLoader(
            BaseDataSet(env.data_dir, data_list, dataset_name, config.num_classes, 
                        joint_transform =  joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        max_prop=max_prop,
                        selected=selected,
                        set=set),
            batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return targetloader
