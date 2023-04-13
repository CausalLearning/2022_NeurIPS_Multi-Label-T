#for voc2007/2012 dataset load
import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
import sys
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}

voc12urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
    'test_images_2012': 'http://pjreddie.com/media/files/VOC2012test.tar',
}

def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    labels_list = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                #item = (name, labels)
                #images.append(item)
                images.append(name)
                labels_list.append(labels)
            rownum += 1
    return np.stack(images), np.stack(labels_list)


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


def download_voc2007(root):
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['devkit'], cached_file))
            util.download_url(urls['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls['trainval_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['trainval_2007'], cached_file))
            util.download_url(urls['trainval_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):

        # download test annotations
        parts = urlparse(urls['test_anno_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_anno_2007'], cached_file))
            util.download_url(urls['test_anno_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls['test_images_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_images_2007'], cached_file))
            util.download_url(urls['test_images_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')


class Voc2007Classification(data.Dataset):
    def __init__(self, root, set_name, transform=None, target_transform=None,noise_rate=[0,0],random_seed=1):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        if(set_name=='train' or set_name=='val' or set_name=='trainval'):
            set = 'trainval'
        else:
            set = set_name
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        download_voc2007(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images, self.true_labels = read_object_labels_csv(file_csv)
        
        self.true_labels[self.true_labels==0]=1
        
        if(noise_rate[0]==0 and noise_rate[1]==0):
            self.labels=self.true_labels 
        else:
            self.labels= generate_noisy_labels(self.true_labels , noise_rate,random_seed)
        
        if(set_name=='train'):
            self.images, self.labels, self.true_labels , _, _, _=dataset_split(self.images,self.labels,self.true_labels, num_classes=len(self.classes))
        elif(set_name=='val'):
            _, _, _, self.images, self.labels, self.true_labels=dataset_split(self.images,self.labels,self.true_labels, num_classes=len(self.classes))
        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            set_name, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        #img = np.asarray(img,dtype="float32")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


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
    num_samples = int(train_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = train_images[train_set_index], train_images[val_set_index]
    train_labels, val_labels = train_labels[train_set_index], train_labels[val_set_index]
    train_true_labels, val_true_labels = true_labels[train_set_index], true_labels[val_set_index]

    return train_set, train_labels, train_true_labels, val_set, val_labels, val_true_labels
    
    
def download_voc2012(root):
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(voc12urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(voc12urls['devkit'], cached_file))
            util.download_url(voc12urls['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(voc12urls['trainval_2012'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(voc12urls['trainval_2012'], cached_file))
            util.download_url(voc12urls['trainval_2012'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, 'VOC2012/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(voc12urls['test_images_2012'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(voc12urls['test_images_2012'], cached_file))
            util.download_url(voc12urls['test_images_2012'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

class Voc2012Classification(data.Dataset):
    def __init__(self, root, set_name, transform=None, target_transform=None,noise_rate=[0,0],random_seed=1):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        if(set_name=='train' or set_name=='val' or set_name=='trainval'):
            set = 'trainval'
        else:
            set = set_name
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        download_voc2012(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2012')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2012', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images, self.true_labels = read_object_labels_csv(file_csv)
        
        self.true_labels[self.true_labels==0]=1
        
        if(noise_rate[0]==0 and noise_rate[1]==0):
            self.labels=self.true_labels 
        else:
            self.labels= generate_noisy_labels(self.true_labels , noise_rate,random_seed)
        
        if(set_name=='train'):
            self.images, self.labels, self.true_labels , _, _, _=dataset_split(self.images,self.labels,self.true_labels, num_classes=len(self.classes))
        elif(set_name=='val'):
            _, _, _, self.images, self.labels, self.true_labels=dataset_split(self.images,self.labels,self.true_labels, num_classes=len(self.classes))
        print('[dataset] VOC 2012 classification set=%s number of classes=%d  number of images=%d' % (
            set_name, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)