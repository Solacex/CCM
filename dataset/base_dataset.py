import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import imageio
class BaseDataSet(data.Dataset):
    def __init__(self, root, list_path,dataset, num_class,  joint_transform=None, transform=None, label_transform = None, max_iters=None, ignore_label=255, set='val', plabel_path=None, max_prop=None, selected=None,centroid=None, wei_path=None):
        
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.set = set
        self.dataset = dataset
        self.transform = transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform
        self.plabel_path = plabel_path
        self.centroid = centroid
        self.wei_path = wei_path

        if self.set !='train':
            self.list_path = (self.list_path).replace('train', self.set)

        self.img_ids =[]
        if selected is not None:
            self.img_ids = selected
        else:
            with open(self.list_path) as f:
                for item in f.readlines():
                    fields = item.strip().split('\t')[0]
                    if ' ' in fields:
                        fields = fields.split(' ')[0]
                    self.img_ids.append(fields)

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        elif max_prop is not None:
            total = len(self.img_ids)
            to_sel = int(np.floor(total * max_prop))
            index = list( np.random.choice(total, to_sel, replace=False) )
            self.img_ids = [self.img_ids[i] for i in index]

        self.files = []
        if num_class==19:
            self.gta5_id2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
            self.syn_id2train = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
            self.id2train = self.gta5_id2train
        elif num_class==16:
            self.gta5_id2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 23: 9, 24: 10, 25: 11,
                              26: 12,  28: 13, 32: 14, 33: 15}
            self.syn_id2train = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11,
                              8: 12, 19: 13, 12: 14, 11: 15}
            if self.dataset =='synthia':
                self.id2train = self.syn_id2train
            else:
                self.id2train = self.gta5_id2train

        elif num_class==13:
            self.gta5_id2train = {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5,
                                23: 6, 24: 7, 25: 8, 26: 9, 28: 10, 32: 11, 33: 12}
            self.syn_id2train = {3: 0, 4: 1, 2: 2, 15: 3, 9: 4, 6: 5,
                                1: 6, 10: 7, 17: 8, 8: 9, 19: 10, 12: 11, 11: 12}

        if self.dataset =='synthia':
            imageio.plugins.freeimage.download()

        if dataset=='gta5' and self.plabel_path is None:
            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                label_file = osp.join(self.root, "labels/%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name,
                    "centroid":(0,0)
                })
        elif dataset=='gta5' and self.plabel_path is not None:
            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                label_file = osp.join(self.plabel_path, "%s" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name,
                    "centroid":(0,0)
                })

        elif dataset=='cityscapes' and self.plabel_path is None:
            for name in self.img_ids:
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
                label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
                label_file =osp.join(self.root, 'gtFine/%s/%s' % (self.set, label_name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name,
                    "hard_loc":(0,0)
                })
        elif dataset=='cityscapes' and self.plabel_path is not None:
            for name in self.img_ids:
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
                label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
                label_file =osp.join(self.plabel_path, '%s' % (label_name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name,
                    "hard_loc":(0,0)
                })
                
        elif dataset=='synthia' and self.plabel_path is None:
            for name in self.img_ids:
                img_file = osp.join(self.root, "RGB/%s" % (name))
                label_file =osp.join(self.root, 'GT/LABELS/%s' % (name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name,
                    "hard_loc":(0,0)
                })
        elif dataset=='synthia' and self.plabel_path is not None:
            for name in self.img_ids:
                img_file = osp.join(self.root, "RGB/%s" % (name))
                label_file =osp.join(self.plabel_path, '%s' % (name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name,
                    "hard_loc":(0,0)
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        try:
            image = Image.open(datafiles["img"]).convert('RGB')
            if self.dataset=='synthia' and self.plabel_path is None :
                label = np.asarray(imageio.imread(datafiles["label"], format='PNG-FI'))[:,:,0] 
            else:
                label = Image.open(datafiles["label"])
            name = datafiles["name"]

            wei_name =name.replace('png', 'npy')

            if self.wei_path is not None:
                wei = np.load(osp.join(self.wei_path, wei_name))
                wei = Image.fromarray(wei)
            else:
                wei = 0


            label = np.asarray(label, np.uint8)
             # re-assign labels to match the format of Cityscapes
            label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
            if self.plabel_path is None:
            #    print(self.id2train)

                for k, v in self.id2train.items():
                    label_copy[label == k] = v
            else:
                label_copy = label
            label = Image.fromarray(label_copy.astype(np.uint8))
            if self.centroid is not None :
                file_name = name.split('/')[-1]
                centroid = self.centroid[file_name]
            else:
                centroid=None
            if self.wei_path is None:
                if self.joint_transform is not None:
                    image, label = self.joint_transform(image, label, centroid)
                if self.transform is not None:
                    image = self.transform(image)
                if self.label_transform is not None:
                    label = self.label_transform(label)
                wei = torch.ones_like(label).float()
            else:
                if self.joint_transform is not None:
                    image, label = self.joint_transform(image, (label, wei), centroid)
                    label, wei = label
                if self.transform is not None:
                    image = self.transform(image)
                if self.label_transform is not None:
                    label, wei = self.label_transform((label, wei))
        except Exception as e:

            print(index)
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        return image, label, wei, 0, name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
