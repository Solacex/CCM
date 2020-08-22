import torch
import os.path as osp
import neptune
import torch.nn as nn
import neptune 
from dataset import dataset
from tqdm import tqdm
import numpy as np
class BaseTrainer(object):
    def __init__(self, models, optimizers, loaders, up_s, up_t, config,  writer):
        self.model = models
        self.optim = optimizers
        self.loader = loaders
        self.config = config
        self.output = {}
        self.up_src = up_s
        self.up_tgt = up_t
        self.writer = writer

    def forward(self):
        pass
    def backward(self):
        pass

    def iter(self):
        pass

        
    def train(self):
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter==0 and self.config.neptune:
                neptune.init(project_qualified_name='solacex/segmentation-DA')
                neptune.create_experiment(params=self.config, name=self.config['note'])
            if i_iter % self.config.print_freq ==0:
                self.print_loss(i_iter)
            if i_iter % self.config.save_freq ==0 and i_iter != 0:
                self.save_model(i_iter)
            if self.config.val and i_iter % self.config.val_freq ==0 and i_iter!=0:
                self.validate()
        neptune.stop()

    def save_model(self, iter):
        tmp_name = '_'.join((self.config.source, str(iter))) + '.pth'
        torch.save(self.model.state_dict(), osp.join(self.config['snapshot'], tmp_name))

    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor +'  '+ loss_infor)
        if self.config.neptune:
            for key in self.losses.keys():
                neptune.send_metric(key, self.losses[key].item())
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/'+key, self.losses[key], iter)

    def validate(self):
        self.model = self.model.eval()
        testloader = dataset.init_test_dataset(self.config, self.config.target, set='val')
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        union = torch.zeros(self.config.num_classes, 1,dtype=torch.float).cuda().float()
        inter = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        preds = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(testloader)):
                image, label, _, _, name = batch
                output =  self.model(image.cuda())
                label = label.cuda()
                output = interp(output).squeeze()
                C, H, W = output.shape
                Mask = (label.squeeze())<C

                pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                pred_e = pred_e.repeat(1, H, W).cuda()
                pred = output.argmax(dim=0).float()
                pred_mask = torch.eq(pred_e, pred).byte()
                pred_mask = pred_mask*Mask.byte()

                label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                label_e = label_e.repeat(1, H, W).cuda()
                label = label.view(1, H, W)
                label_mask = torch.eq(label_e, label.float()).byte()
                label_mask = label_mask*Mask.byte()

                tmp_inter = label_mask+pred_mask.byte()
                cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()

                union+=cu_union
                inter+=cu_inter
                preds+=cu_preds

            iou = inter/union
            acc = inter/preds
            if C==16:
                iou = iou.squeeze()
                class13_iou = torch.cat((iou[:3], iou[6:]))
                class13_miou = class13_iou.mean().item()
                print('13-Class mIoU:{:.2%}'.format(class13_miou))
            mIoU = iou.mean().item()
            mAcc = acc.mean().item()
            iou = iou.cpu().numpy()
            print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
            if self.config.neptune:
                neptune.send_metric('mIoU', mIoU)
                neptune.send_metric('mAcc', mAcc)
        return mIoU

