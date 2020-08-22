import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Pool(object):
    def __init__(self, num_cls=19, alpha=0.1, T=2):
        super(Pool,self).__init__()
        self.num_cls = num_cls
        self.alpha  = alpha
        self.pool = {}
        self.T =T
        self.cnts ={}
        
    def update(self, vecs, cnts):
        for key, vec in vecs.items():
            if key not in self.pool:
                self.pool[key] = vec
                self.cnts[key] = cnts[key]
            else:
                past_cnt = self.cnts[key]
                self.pool[key] = (self.pool[key]*past_cnt+ vec*cnts[key])/float(past_cnt+cnts[key])


    def get_mean(self, P, label=None,mask=None):
        assert P.dim()==4
        P = F.softmax(P/self.T, dim=1)
        if label is None:
            label = P.argmax(dim=1).squeeze()

        _, C, H, W = P.shape
        
        mean_vec = {}
        mean_cnt = {}
        for i in range(self.num_cls):
            cls_mask = (label==i).float()
            if mask is not None:
                cls_mask = (mask * cls_mask).float()
                
            num = cls_mask.sum().float()
            if num==0:
                continue
            tmp_p = cls_mask * P.permute(1,0,2,3)
            vec = tmp_p.sum(dim=(1, 2, 3))/num 
            mean_vec[i] = vec
            mean_cnt[i] = num
        return mean_vec, mean_cnt
    

    def update_pool(self, P, mask=None):
        mean_vec, mean_cnt = self.get_mean(P, mask=mask)
        self.update(mean_vec, mean_cnt)

