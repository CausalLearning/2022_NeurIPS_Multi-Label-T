import torch
from torchvision import transforms
from models import *
import torch.optim as optim
import torch.nn as nn
import time
import argparse
import os
import csv
import copy,sys
import numpy as np
from test import *
from voc import *
from coco import *
from util import *
from torch.autograd import Variable
import copy

parser = argparse.ArgumentParser('')
parser.add_argument('--bs',"--batch_size", type=int, default=16, help="batch size")
parser.add_argument('--nc',"--num_classes", type=int, default=20, help="num_classes")
parser.add_argument("--warmup_epoch", type=int, default=30, help="warmup epochs")
parser.add_argument("--nepochs", type=int, default=30, help="max epochs")
parser.add_argument("--sample_epoch", type=int, default=10, help="epoch for sample selection for ours")
parser.add_argument("--sample_th", type=float, default=0.5, help="th for sample selection")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--dataset", type=str, default='voc2007')
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--root", type=str, default='/home/lishikun/data/voc/')
parser.add_argument("--out", type=str, default='/home/lishikun/results/multi-label_p0.6_voc2007_ours_resnet50/')
parser.add_argument("--noise_rate_p", type=float, default=0.1)
parser.add_argument("--noise_rate_n", type=float, default=0.1)
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estimator", default='ours', type=str, help='ours, dualT or T')
parser.add_argument("--filter_outlier", default=False, action = "store_true")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if device== 'cuda:0':
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists(args.out):
    os.mkdir(args.out)
if not os.path.exists(args.out+'saved-models/'):
    os.mkdir(args.out+'saved-models/')
if not os.path.exists(args.out+'logs/'):
    os.mkdir(args.out+'logs/')

# Setup folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = args.out+'logs'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
    run += 1
    current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
log_file = open('{}/log.txt'.format(current_dir), 'a')
print(args, file=log_file)

__console__=sys.stdout
sys.stdout=log_file

noise_rate=[args.noise_rate_n,args.noise_rate_p]
args.noise_rate=noise_rate
print(args.noise_rate)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    MultiScaleCrop(224, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_transform= transforms.Compose([
    Warp(224),
    transforms.ToTensor(),
    normalize,
])
if(args.dataset=='voc2007'):
    train_dataset = Voc2007Classification(args.root, 'train',noise_rate=args.noise_rate,transform=train_transform,random_seed=args.seed)
    val_dataset = Voc2007Classification(args.root, 'val',noise_rate=args.noise_rate,transform=val_transform,random_seed=args.seed)
    test_dataset = Voc2007Classification(args.root, 'test',transform=val_transform,random_seed=args.seed)
elif(args.dataset=='voc2012'):
    train_dataset = Voc2012Classification(args.root, 'train',noise_rate=args.noise_rate,transform=train_transform,random_seed=args.seed)
    val_dataset = Voc2012Classification(args.root, 'val',noise_rate=args.noise_rate,transform=val_transform,random_seed=args.seed)
    test_dataset = Voc2007Classification(args.root, 'test',transform=val_transform,random_seed=args.seed)
else:
    train_dataset = COCO2014(args.root, phase='train',Train=True,noise_rate=args.noise_rate,transform=train_transform,random_seed=args.seed)
    val_dataset = COCO2014(args.root, phase='train', Train=False,noise_rate=args.noise_rate,transform=val_transform,random_seed=args.seed)
    test_dataset = COCO2014(args.root, phase='val',transform=val_transform,random_seed=args.seed)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                          shuffle=True, num_workers=args.nworkers,drop_last=True)
                          
estimate_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                          shuffle=False, num_workers=args.nworkers,drop_last=False)                          
args.true_train_labels = train_dataset.true_labels

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs,
                        shuffle=False, num_workers=args.nworkers,drop_last=False)
                        
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs,
                        shuffle=False, num_workers=args.nworkers,drop_last=False)
def get_noisy_prob(transition_mat, clean_prob):
    return torch.matmul(transition_mat.T, clean_prob.T).squeeze().T
    
def train(net, train_loader, optimizer,t_m):
    net.train()
    running_loss = 0
    t_m = torch.Tensor(t_m).cuda()
    loss_fn = F.binary_cross_entropy
    for i, (images, n_target) in enumerate(train_loader):
        images = images.cuda().float()
        n_target = n_target.cuda().float()
        n_target[n_target==0]=1
        n_target[n_target==-1]=0
        output = net(images)
        probs = torch.sigmoid(output)

        try:
            loss=0
            for i in range(n_target.shape[1]):
                target = n_target[:,i].long()
                out_softmax=torch.vstack((1-probs[:,i],probs[:,i])).t()
                noisy_prob = get_noisy_prob(t_m[i], out_softmax)
                pro1 = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()
                pro2 = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
                beta = pro1 / pro2
                beta = Variable(beta, requires_grad=True)
                cross_loss = loss_fn(out_softmax[:,1], target.float(), reduction='none')
                _loss = beta * cross_loss
                loss+=torch.mean(_loss)
            loss = loss/n_target.shape[1]     
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        except:
            print('grad_error')
            continue

    train_loss =running_loss / len(train_loader)
    print("training loss" ,train_loss)
    return train_loss
    
if __name__ == '__main__':

    print('seed',args.seed)
    net = get_resnet50(args.nc,pretrained=True)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net=net.to(device)
    if torch.cuda.device_count() > 1:
        optimizer_es = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay) 
    else:
        optimizer_es = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay) 
                                  
    true_tm = np.zeros((args.nc,2,2))
    for i in range(args.nc):
        true_tm[i,0,0]=1-noise_rate[0]
        true_tm[i,0,1]=noise_rate[0]
        true_tm[i,1,0]=noise_rate[1]
        true_tm[i,1,1]=1-noise_rate[1]  

    # Estimate Transition Matrices
    if(args.estimator=="ours"):
        from noise_rate_estimation_ours import *
        t_m=estimate_noise_rate(net,train_loader, val_loader, estimate_loader,optimizer_es,args,true_tm=true_tm)
    elif(args.estimator=="dualT"): 
        from noise_rate_estimation_DualT import *    
        t_m=estimate_noise_rate(net,train_loader, val_loader, estimate_loader,optimizer_es,args,true_tm=true_tm,filter_outlier=args.filter_outlier)
    elif(args.estimator=="T"): 
        from noise_rate_estimation_T import *    
        t_m=estimate_noise_rate(net,train_loader, val_loader, estimate_loader,optimizer_es,args,true_tm=true_tm,filter_outlier=args.filter_outlier)
    print('t_m',t_m)

    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay) 
    else:
        optimizer =  torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    # Train the network with reweighting
    
    val_metric=[]
    test_metric=[]
     
    best_val=0
    for i in range(args.nepochs):
        start = time.time()
        train_loss = train(net, train_loader, optimizer,t_m)
        map, OP, OR, OF1, CP, CR, CF1= test(net, val_loader)
        
        print('Epoch',i,' val_map, OP, OR, OF1, CP, CR, CF1 ',round(map,2), round(OP,2), round(OR,2), round(OF1,2), round(CP,2),round(CR,2),round(CF1,2))  
        val_metric.append([map, OP, OR, OF1, CP, CR, CF1])
        if(map>best_val):
            best_val=map
            best_state_dict = copy.deepcopy(net.state_dict())
        
        map, OP, OR, OF1, CP, CR, CF1= test(net, test_loader)
        print('Epoch',i,'test_map, OP, OR, OF1, CP, CR, CF1 ',round(map,2), round(OP,2), round(OR,2), round(OF1,2), round(CP,2),round(CR,2),round(CF1,2))  
        test_metric.append([map, OP, OR, OF1, CP, CR, CF1])

        end = time.time()
        print('time', round(end-start,3))
        log_file.flush()
    
    # early stop
    net.load_state_dict(best_state_dict)

    val_metric=np.array(val_metric)
    test_metric=np.array(test_metric)
    best_map,_, _,best_OF1,_, _, best_CF1=np.argmax(val_metric,axis=0)
    
    print('Best map',' val_map, OP, OR, OF1, CP, CR, CF1 ',round(val_metric[best_map,0],2), round(val_metric[best_map,1],2), round(val_metric[best_map,2],2), round(val_metric[best_map,3],2), round(val_metric[best_map,4],2),round(val_metric[best_map,5],2),round(val_metric[best_map,6],2))
    
    print('Best map','test_map, OP, OR, OF1, CP, CR, CF1 ',round(test_metric[best_map,0],2), round(test_metric[best_map,1],2), round(test_metric[best_map,2],2), round(test_metric[best_map,3],2), round(test_metric[best_map,4],2),round(test_metric[best_map,5],2),round(test_metric[best_map,6],2))
     

    log_file.close()
