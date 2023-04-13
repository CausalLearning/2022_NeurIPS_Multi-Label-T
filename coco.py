#for coco dataset load
import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *
import sys

urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def download_coco2014(root, phase):
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    data = os.path.join(root, 'data/')
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file,data)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.Popen('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    anno2 = os.path.join(data, '{}_anno2.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not (os.path.exists(anno) and os.path.exists(anno2)):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        anno_list_2 = []
        for k, v in img_id.items():
            anno_list_2.append(v['labels'])
        json.dump(anno_list_2, open(anno2, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del anno_list_2
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(data.Dataset):
    def __init__(self, root, transform=None, phase='train', Train=True, noise_rate=[0,0],random_seed=1):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.true_labels=self.get_true_labels()
        self.num_classes = len(self.cat2idx)
        self.img_list=np.array(self.img_list)
        if(phase=='train'):
            self.labels= generate_noisy_labels(self.true_labels , noise_rate,random_seed)
            if(Train):
                self.img_list , self.labels, self.true_labels , _, _, _=dataset_split(self.img_list ,self.labels,self.true_labels, num_classes=self.num_classes)
            else:
                _, _, _, self.img_list , self.labels, self.true_labels =dataset_split(self.img_list ,self.labels,self.true_labels, num_classes=self.num_classes)
        else:
            self.labels= self.true_labels

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def get_true_labels(self):
        list_path = os.path.join(self.root, 'data', '{}_anno2.json'.format(self.phase))
        labels=json.load(open(list_path, 'r'))
        true_labels=np.zeros((len(labels),len(self.cat2idx)))-1
        for i,label in enumerate(labels):
            true_labels[i,label]=1
        return true_labels
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        target=self.labels[index]
        return self.get(item),target

    def get(self, item):
        filename = item['file_name']
        #labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # target = np.zeros(self.num_classes, np.float32) - 1
        # target[labels] = 1
        return img


def generate_noisy_labels(labels, noise_rate,random_seed):
	
    N, nc = labels.shape
    np.random.seed(random_seed)
    rand_mat = np.random.rand(N,nc)
    mask = np.zeros((N,nc), dtype = np.float)
    for j in range(nc):
        yj = labels[:,j]
        mask[yj!=1,j] = rand_mat[yj!=1,j]<noise_rate[0]
        mask[yj==1,j] = rand_mat[yj==1,j]<noise_rate[1]

    noisy_labels = np.copy(labels)
    noisy_labels[mask==1] = -noisy_labels[mask==1]

    for i in range(nc):
        noise_rate_p= sum(noisy_labels[labels[:,i]==1,i]==-1)/sum(labels[:,i]==1)
        noise_rate_n= sum(noisy_labels[labels[:,i]==-1,i]==1)/sum(labels[:,i]==-1)
        print('noise_rate_class',str(i),'noise_rate_n',noise_rate_n,'noise_rate_p',noise_rate_p,'n',sum(labels[:,i]==-1),'p',sum(labels[:,i]==1))
        
    return noisy_labels
    
def dataset_split(train_images, train_labels, true_labels, split_per=0.9, random_seed=1, num_classes=10):
    num_samples = len(train_labels)
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(len(train_labels))
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = train_images[train_set_index], train_images[val_set_index]
    train_labels, val_labels = train_labels[train_set_index], train_labels[val_set_index]
    train_true_labels, val_true_labels = true_labels[train_set_index], true_labels[val_set_index]

    return train_set, train_labels, train_true_labels, val_set, val_labels, val_true_labels