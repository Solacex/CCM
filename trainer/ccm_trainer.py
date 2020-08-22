import torch
from utils.optimize import adjust_learning_rate
from .base_trainer import BaseTrainer
from utils.flatwhite import *
from easydict import EasyDict as edict
import os.path as osp
from dataset import dataset
import neptune
import math
from PIL import Image
from utils.meters import AverageMeter, GroupAverageMeter 
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import operator
import pickle
import random 
from utils.kmeans import kmeans_cluster
from utils.func import Acc, thres_cb_plabel,gene_plabel_prop, mask_fusion 
from utils.pool import Pool
from utils.flatwhite import *
from trainer.base_trainer import *

class Trainer(BaseTrainer):
    def __init__(self, model, config, writer):
        self.model = model
        self.config = config
        self.writer = writer
    def entropy_loss(self, p):
        p = F.softmax(p, dim=1)
        log_p = F.log_softmax(p, dim=1)
        loss = -torch.sum(p * log_p, dim=1)
        return loss

    def iter(self, batch):
        img_s, label_s, _, _, name = batch
        b, c, h, w = img_s.shape
        pred_s = self.model.forward(img_s.cuda())
        label_s = label_s.long().cuda()

        loss_s = F.cross_entropy(pred_s, label_s, ignore_index=255)

        loss_e = self.entropy_loss(pred_s)
        loss_e = loss_e.mean()

        self.losses.loss_source = loss_s
        self.losses.loss_entropy = loss_e
        loss = loss_s + self.config.lamb * loss_e
        loss.backward()


    def train(self):
        if self.config.neptune:
            neptune.init(project_qualified_name="solacex/segmentation-DA")
            neptune.create_experiment(params=self.config, name=self.config["note"])
        if self.config.resume:
            self.resume()
        else:
            self.round_start = 0

        for r in range(self.round_start, self.config.round):
            torch.manual_seed(1234)
            torch.cuda.manual_seed(1234)
            np.random.seed(1234)
            random.seed(1234)
            self.model = self.model.train()

            self.source_all = get_list(self.config.gta5.data_list)
            self.target_all = get_list(self.config.cityscapes.data_list)#[:100]

            self.cb_thres = self.gene_thres(self.config.cb_prop)
            self.save_pred(r)
            self.plabel_path = osp.join(self.config.plabel, self.config.note, str(r))

            # Semantic Layout Matching
            self.source_selected = self.semantic_layout_matching(r, self.config.src_count)
            # Pixel-wise simialrity matching
            self.source_pixel_selection(r)

            
            self.optim = torch.optim.SGD(
                self.model.optim_parameters(self.config.learning_rate),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

            self.loader, _ = dataset.init_concat_dataset(
                self.config,
                plabel_path=self.plabel_path,
                selected=self.source_selected,
                source_plabel_path = self.source_plabel_path,
                target_selected=self.target_all)

            self.config.num_steps=5000

            for epoch in range(self.config.epochs):
                for i_iter, batch in tqdm(enumerate(self.loader)):
                    cu_step = epoch * len(self.loader) + i_iter
                    self.model = self.model.train()
                    self.losses = edict({})
                    self.optim.zero_grad()
                    adjust_learning_rate(self.optim, cu_step, self.config)
                    self.iter(batch)

                    self.optim.step()
                    if i_iter % self.config.print_freq == 0:
                        self.print_loss(i_iter)
                    if i_iter % self.config.val_freq ==0 and i_iter!=0:
                        miou = self.validate()
                miou = self.validate()
            self.config.learning_rate = self.config.learning_rate / (math.sqrt(2))
        if  self.config.neptune:
            neptune.stop()

    def resume(self):
        iter_num = self.config.init_weight[-5]#.split(".")[0].split("_")[1]
        iter_num = int(iter_num)
        self.round_start = int(math.ceil((iter_num+1) / self.config.epochs) )
        print("Resume from Round {}".format(self.round_start))
        if self.config.lr_decay == "sqrt":
            self.config.learning_rate = self.config.learning_rate / (
                (math.sqrt(2)) ** self.round_start
            )

    def gene_thres(self, prop, num_cls=19):
        print('[Calculate Threshold using config.cb_prop]') # r in section 3.3

        probs = {}
        freq = {}
        loader = dataset.init_test_dataset(self.config, self.config.target, set="train", selected=self.target_all, batchsize=1)
        for index, batch in tqdm(enumerate(loader)):
            img, label, _, _, _ = batch
            with torch.no_grad():
                pred = F.softmax(self.model.forward(img.cuda()), dim=1)
            pred_probs = pred.max(dim=1)[0]
            pred_probs = pred_probs.squeeze()
            pred_label = torch.argmax(pred, dim=1).squeeze()
            for i in range(num_cls):
                cls_mask = pred_label == i
                cnt = cls_mask.sum()
                if cnt == 0:
                    continue
                cls_probs = torch.masked_select(pred_probs, cls_mask)
                cls_probs = cls_probs.detach().cpu().numpy().tolist()
                cls_probs.sort()
                if i not in probs:
                    probs[i] = cls_probs[::5] # reduce the consumption of memory
                else:
                    probs[i].extend(cls_probs[::5])

        growth = {}
        thres = {}
        for k in probs.keys():
            cls_prob = probs[k]
            cls_total = len(cls_prob)
            freq[k] = cls_total
            cls_prob = np.array(cls_prob)
            cls_prob = np.sort(cls_prob)
            index = int(cls_total * prop)
            cls_thres = cls_prob[-index]
            cls_thres2 = cls_prob[index]
            thres[k] = cls_thres
        print(thres)
        return thres

    def save_pred(self, round):
        # Using the threshold to generate pseudo labels and save  
        print("[Generate pseudo labels]")
        loader = dataset.init_test_dataset(self.config, self.config.target, set="train", selected=self.target_all)
        interp = nn.Upsample(size=(1024, 2048), mode="bilinear", align_corners=True)

        self.plabel_path = osp.join(self.config.plabel, self.config.note, str(round))

        mkdir(self.plabel_path)
        self.config.target_data_dir = self.plabel_path
        self.pool = Pool() # save the probability of pseudo labels for the pixel-wise similarity matchinng, which is detailed around Eq. (9)
        accs = AverageMeter()  # Counter
        props = AverageMeter() # Counter
        cls_acc = GroupAverageMeter() # Class-wise Acc/Prop of Pseudo labels

        self.mean_memo = {i: [] for i in range(self.config.num_classes)}
        with torch.no_grad():
            for index, batch in tqdm(enumerate(loader)):
                image, label, _, _, name = batch
                label = label.cuda()
                img_name = name[0].split("/")[-1]
                dir_name = name[0].split("/")[0]
                img_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                temp_dir = osp.join(self.plabel_path, dir_name)
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)

                output = self.model.forward(image.cuda())
                output = interp(output)
                # pseudo labels selected by glocal threshold
                mask, plabel = thres_cb_plabel(output, self.cb_thres, num_cls=self.config.num_classes)
                # pseudo labels selected by local threshold
                mask2, plabel2 = gene_plabel_prop( output, self.config.cb_prop)
                # mask fusion 
                # The fusion strategy is detailed in Sec. 3.3 of paper
                mask, plabel = mask_fusion(output, mask, mask2)
                self.pool.update_pool(output, mask=mask.float())
                acc, prop, cls_dict = Acc(plabel, label, num_cls=self.config.num_classes)
                cnt = (plabel != 255).sum().item()
                accs.update(acc, cnt)
                props.update(prop, 1)
                cls_acc.update(cls_dict)
                plabel = plabel.view(1024, 2048)
                plabel = plabel.cpu().numpy()

                plabel = np.asarray(plabel, dtype=np.uint8)
                plabel = Image.fromarray(plabel)

                plabel.save("%s/%s.png" % (temp_dir, img_name.split(".")[0]))

        print('The Accuracy :{:.2%} and proportion :{:.2%} of Pseudo Labels'.format(accs.avg.item(), props.avg.item()))
        if self.config.neptune:
            neptune.send_metric("Acc", accs.avg)
            neptune.send_metric("Prop", props.avg)


    def semantic_layout_matching(self, round, num_to_sel):
        print('[Semantic Layout Matching]')

        loader = dataset.init_test_dataset(self.config, self.config.source, set='train')
        self.get_target_slm() # Calculate the SLM of the target domain
        self.target_his_h = torch.stack(list(self.target_his_h.values()))
        self.target_his_w = torch.stack(list(self.target_his_w.values()))

        self.target_his_h = self.k_means(self.target_his_h)
        self.target_his_w = self.k_means(self.target_his_w)

        source_selected = []
        score_dict = {}
        name_pair = {}

        for index, batch in tqdm(enumerate(loader)):
            img, label, _, _, name  = batch
            name = name[0]

            label = label.cuda().squeeze()
            h,w = label.shape
            cu_h = torch.zeros(h, self.config.num_classes).float().cuda()
            cu_w = torch.zeros(w, self.config.num_classes).float().cuda()
            for i in range(self.config.num_classes):
                mask = label==i
                mask_h = mask.sum(dim=1).float()
                mask_w = mask.sum(dim=0).float()
                cu_h[:,i]=mask_h
                cu_w[:,i]=mask_w
            cu_h = F.normalize(cu_h, p=1, dim=0)
            cu_w = F.normalize(cu_w, p=1, dim=0)

            cu_h = cu_h.t()
            cu_w = cu_w.t()
            score1 = self.his_kl_simi(self.target_his_h, cu_h)
            score2 = self.his_kl_simi(self.target_his_w, cu_w)

            score_dict[name] = score1 + score2

        sorted_pair = sorted(score_dict.items(), key=operator.itemgetter(1))
        sorted_name  = [m[0] for m in sorted_pair]
        distance = [m[1] for m in sorted_pair]
        
        self.selected = sorted_name[:num_to_sel-1]
        return self.selected

    def get_target_slm(self):
        print('Generate SLM[Semantic layour matrix] of target samples')
        loader = dataset.init_test_dataset(self.config, self.config.target, set='train',selected=self.target_all)
        self.target_his_h = {}
        self.target_his_w = {}

        for index, batch in tqdm(enumerate(loader)):
            img, label, _, _, name  = batch
            name = name[0]
            with torch.no_grad():
                pred = self.model.forward(img.cuda())
            pred = F.softmax(pred, dim=1)
            label = pred.argmax(dim=1).squeeze()
            h, w = label.shape
            pred = pred.squeeze()
            cu_h = pred.sum(dim=2)
            cu_w = pred.sum(dim=1)
            cu_h = cu_h.t()
            cu_w = cu_w.t()
            cu_h = F.normalize(cu_h, p=1, dim=0)
            cu_w = F.normalize(cu_w, p=1, dim=0)
            self.target_his_h[name] = cu_h
            self.target_his_w[name] = cu_w

    def k_means(self, hist):
        result = []
        n,hw,c = hist.shape
        hists = torch.chunk(hist, c, dim=2)

        for hist in hists:
            hist = hist.squeeze()
            centers, codes = kmeans_cluster(hist, self.config.num_center)
            result.append(centers)
            #print(centers.shape)
        result = torch.stack(result)
        result = result.permute(1,0,2)
        return result

    def calc_pixel_simi(self, pred, label, pool):

        _, c, h, w = pred.shape
        to_calc = label.unique()
        wei = torch.ones_like(label).float()
        plabel = torch.ones_like(label)*255
        none = torch.ones_like(label)*255
        for i in range(len(to_calc)):

            i = to_calc[i].item()
            if i > self.config.num_classes-1:
                continue
            target = pool[i]
            cls_mask = label==i

            cls_cnt = cls_mask.sum().float()
            to_sel = (self.config.src_pix_prop * cls_cnt).int().item()
            cls_wei = self.kl_simi(target, pred)
            cls_wei = (cls_wei * cls_mask.float())
            srted, _ = torch.sort(cls_wei.view(-1), descending=True)
            thres = srted[to_sel]
            sel_cls_mask = cls_wei > thres
            sel_cls_mask = sel_cls_mask * cls_mask
            plabel = torch.where(sel_cls_mask, i*torch.ones_like(label), plabel)
        return plabel

    def source_pixel_selection(self, round):
        print("Pixel-wise similarity matching")
        loader = dataset.init_test_dataset(self.config, self.config.source, set="train", selected=self.source_selected, label_ori=True)
        if self.config.source=='gta5':
            src_w = 1914
            src_h = 1052
        elif self.config.source=='synthia':
            src_w = 1280
            src_h = 760

        interp = nn.Upsample(size=(src_h, src_w), mode="bilinear", align_corners=True)
        self.source_plabel_path = osp.join(self.config.plabel, self.config.note, str(round), self.config.source)
        mkdir(self.source_plabel_path)

        target_template = self.pool.pool

        self.mean_memo = {i: [] for i in range(self.config.num_classes)}
        self.target_cnts = {i:0 for i in range(self.config.num_classes)}
        with torch.no_grad():
            for index, batch in tqdm(enumerate(loader)):
                image, label, _, _, name = batch
                label = label.cuda()
                name = name[0]
                output = self.model.forward(image.cuda())
                output = interp(output)
                pred = F.softmax(output, dim=1)
                pred_label = pred.argmax(dim=1)
                plabel = self.calc_pixel_simi(pred, label, target_template)

                plabel = plabel.view(src_h, src_w)
                plabel = plabel.cpu().numpy()

                plabel = np.asarray(plabel, dtype=np.uint8)
                plabel = Image.fromarray(plabel)

                plabel.save("%s/%s.png" % (self.source_plabel_path, name.split(".")[0]))

    def his_kl_simi(self, p1, p2):
        c,hw = p2.shape
        n, c, hw = p1.shape
        p2=p2.squeeze()
        log_p1 = torch.log(p1+1e-30)
        log_p2 = torch.log(p2+1e-30)

        kl = p1 * (log_p1-log_p2)
        kl = kl.sum(dim=2)
        kl = kl.mean()
        return kl

    def kl_simi(self, p1, p2):
        c = p1.shape
        n,c,h,w = p2.shape

        p2=p2.squeeze()
        p2=p2.permute(1,2,0)
        log_p1 = torch.log(p1+1e-30)
        log_p2 = torch.log(p2+1e-30)

        kl = p1 * (log_p1-log_p2)
        kl = kl.sum(dim=2)
        kl = kl.squeeze()
        kl = kl.max()-kl
        return kl

    def save_model(self, iter):
        tmp_name = "_".join(("GTA5", str(iter))) + ".pth"
        torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], tmp_name))
