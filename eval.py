import os, sys
import argparse
import numpy as np
import torch
from model.DeeplabV2 import *#Res_Deeplab

from torch.utils import data
import torch.nn as nn
import os.path as osp
import yaml
from utils.logger import Logger 
from dataset.dataset import *
from easydict import EasyDict as edict
from tqdm import tqdm

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--frm", type=str, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='cityscapes')
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--model", default='deeplab')
    return parser.parse_args()

def print_iou(iou, acc, miou, macc):
    for ind_class in range(iou.shape[0]):
        print('===> {0:2d} : {1:.2%} {2:.2%}'.format(ind_class, iou[ind_class, 0].item(), acc[ind_class, 0].item()))
    print('mIoU: {:.2%} mAcc : {:.2%} '.format(miou, macc))

def compute_iou(model, testloader, args):
    model = model.eval()

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    union = torch.zeros(args.num_classes, 1,dtype=torch.float).cuda().float()
    inter = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    preds = torch.zeros(args.num_classes, 1, dtype=torch.float).cuda().float()
    with torch.no_grad():
        for index, batch in tqdm(enumerate(testloader)):
            image, label, edge, _, name = batch
#            edge = F.interpolate(edge.unsqueeze(0), (512, 1024)).view(1,512,1024)
            output =  model(image.cuda())
            label = label.cuda()
            output = interp(output).squeeze()
            C, H, W = output.shape
            Mask = (label.squeeze())<C

            pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            pred_e = pred_e.repeat(1, H, W).cuda()
            pred = output.argmax(dim=0).float()
            pred_mask = torch.eq(pred_e, pred).byte()
            pred_mask = pred_mask*Mask

            label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
            label_e = label_e.repeat(1, H, W).cuda()
            label = label.view(1, H, W)
            label_mask = torch.eq(label_e, label.float()).byte()
            label_mask = label_mask*Mask

            tmp_inter = label_mask+pred_mask
            cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
            cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

            union+=cu_union
            inter+=cu_inter
            preds+=cu_preds

        iou = inter/union
        acc = inter/preds
        mIoU = iou.mean().item()
        mAcc = acc.mean().item()
        print_iou(iou, acc, mIoU, mAcc)
        return iou, mIoU, acc, mAcc

def main():
    args = get_arguments()
    with open('./config/config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    cfg.num_classes=args.num_classes
    if args.single:
        #from model.fuse_deeplabv2 import Res_Deeplab
        if args.model=='deeplab':
            model = Res_Deeplab(num_classes=args.num_classes)
        else:
            model = FCN8s(num_classes = args.num_classes).cuda() 

        #model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.frm,map_location='cuda:0'))
        model.eval().cuda()
        testloader = init_test_dataset(cfg, args.dataset, set='val')
        iou, mIoU, acc, mAcc = compute_iou(model, testloader, args)
        return

    sys.stdout = Logger(osp.join(cfg['result'], args.frm+'.txt'))

    best_miou = 0.0
    best_iter = 0
    best_iou = np.zeros((args.num_classes, 1))

   

    for i in range(args.start, 25):
        model_path = osp.join(cfg['snapshot'], args.frm, 'GTA5_{0:d}.pth'.format(i*2000))# './snapshots/GTA2Cityscapes/source_only/GTA5_{0:d}.pth'.format(i*2000)
        model = Res_Deeplab(num_classes=args.num_classes)
        #model = nn.DataParallel(model)

        model.load_state_dict(torch.load(model_path))
        model.eval().cuda()
        testloader = init_test_dataset(cfg, args.dataset, set='train') 

        iou, mIoU, acc, mAcc = compute_iou(model, testloader)

        print('Iter {}  finished, mIoU is {:.2%}'.format(i*2000, mIoU))

if __name__ == '__main__':
    main()
