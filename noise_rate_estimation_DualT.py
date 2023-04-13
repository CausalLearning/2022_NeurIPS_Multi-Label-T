# DualT estimator
import torch
import sys 
import numpy as np
import tools
import copy
import torch.nn as nn
def loss_func_bce(out, batch_y):
    bce = torch.nn.BCELoss()
    loss = bce(torch.sigmoid(out), batch_y.float())
    return loss

def loss_func_bce2(out, batch_y):
    bce = torch.nn.BCELoss(reduction='sum')
    loss = bce(torch.sigmoid(out), batch_y.float())
    return loss
    
def estimate_noise_rate(model,train_loader, val_loader, estimate_loader,optimizer_es,args,true_tm,filter_outlier=False):
    print('Estimate transition matirx......Waiting......')
    
    A= torch.zeros((args.nc, args.warmup_epoch, len(train_loader.dataset), 2))  
    val_list = [] # for val_loss
    val_list_2 = [] # for val_acc
    val_list_list = [] # for each class's val_loss
    val_list_list_2=[] # for each class's val_acc
    best_val=100
    for i in range(args.nc):
        val_list_list.append([])
        val_list_list_2.append([])
    # warmup training
    for epoch in range(args.warmup_epoch):
      
        print('epoch {}'.format(epoch + 1))
        model.train()
        train_loss = 0.
     
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda().float()
            batch_y = batch_y.cuda().float()
            batch_y[batch_y==0]=1
            batch_y[batch_y==-1]=0
            optimizer_es.zero_grad()
            out = model(batch_x)
            loss = loss_func_bce(out, batch_y)
            train_loss += loss.item()
            loss.backward()
            optimizer_es.step()
          
            
        print('Train Loss: {:.6f}'.format(train_loss / (len(train_loader.dataset))*args.bs))
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.
            val_loss_2=[0]*args.nc
            val_acc=0
            val_acc_2=[0]*args.nc
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.cuda().float()
                batch_y = batch_y.cuda().float()
                batch_y[batch_y==0]=1
                batch_y[batch_y==-1]=0
                out = model(batch_x)
                loss = loss_func_bce(out, batch_y)
                val_loss += loss.item()
                pred = out>0.5
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()
                for i in range(args.nc):
                    loss_2 = loss_func_bce2(out[:,i], batch_y[:,i])
                    val_loss_2[i] += loss_2.item()
                    val_correct_2 = (pred[:,i] == batch_y[:,i]).sum()
                    val_acc_2[i] += val_correct_2.item()
            if(val_loss<best_val):
                best_val = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
            
        print('Val Loss acc: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_loader.dataset))*args.bs, val_acc / (len(val_loader.dataset)*args.nc))) 
        val_list.append(val_loss / (len(val_loader.dataset))*args.bs)
        for i in range(args.nc):
            val_list_list[i].append(val_loss_2[i])
            val_list_list_2[i].append(val_acc_2[i])
        val_list_2.append(val_acc/ (len(val_loader.dataset)*args.nc))
        
        
        index_num = int(len(estimate_loader.dataset) / args.bs)
        with torch.no_grad():
            model.eval()
            for index,(batch_x,batch_y) in enumerate(estimate_loader):
                batch_x = batch_x.cuda().float()
                out = model(batch_x)
                out = torch.sigmoid(out)
                out = out.cpu()
                for i in range(args.nc):
                    out_i = torch.stack([1-out[:,i],out[:,i]]).t()
                    if index <= index_num:
                        A[i, epoch, index*args.bs:(index+1)*args.bs, :] = out_i
                    else:
                        A[i, epoch, index_num*args.bs:len(estimate_loader.dataset), :] = out_i 
    
    model.load_state_dict(best_state_dict)

    # estimation
    True_T = true_tm
    error=0
    est_T = np.zeros_like(True_T)

    target = copy.deepcopy(estimate_loader.dataset.labels)
    target[target==-1]=0
    
    for i in range(args.nc):
        val_array = np.array(val_list_list[i]) # we use the val loss here for selecting each class's model
        model_index = np.argmax(-val_array)
        print('model_index',model_index)
        prob_=A[i]
        transition_matrix_ = tools.fit(prob_[model_index, :, :], 2, filter_outlier)
        transition_matrix = tools.norm(transition_matrix_)
        
        T_ = transition_matrix
        
        T=copy.deepcopy(T_)
        
        if(True_T[i,0,1]==0): #  for multi-label learning with missing labels
            T[0,1]=0
            T[0,0]=1
        if(True_T[i,1,0]==0): #  for partial multi-label learning
            T[1,0]=0
            T[1,1]=1 
        
        estimate_error = tools.error(T, True_T[i])
        print('class', i, 'T estimation',T[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]], 'error', estimate_error) 
        
        pred= np.argmax(prob_[model_index, :, :],axis=-1)
        T_spadesuit = np.zeros((2,2))    
        for j in range(len(target)): 
            T_spadesuit[int(pred[j])][int(target[j,i])]+=1    
        T_spadesuit = np.array(T_spadesuit)
        sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(2,1)).transpose()
        T_spadesuit = T_spadesuit/sum_matrix
        T_spadesuit = np.nan_to_num(T_spadesuit)
        dual_t_matrix = np.matmul(T_, T_spadesuit)

        if(True_T[i,0,1]==0): #  for multi-label learning with missing labels
            dual_t_matrix[0,1]=0
            dual_t_matrix[0,0]=1
        if(True_T[i,1,0]==0): #  for partial multi-label learning
            dual_t_matrix[1,0]=0
            dual_t_matrix[1,1]=1 
        
        
        estimate_error = tools.error(dual_t_matrix, True_T[i])
        est_T[i]= dual_t_matrix
        error+=estimate_error
        print('class', i, 'Dual-T estimation', dual_t_matrix[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]], 'error', estimate_error,'\n')  
    print('total error', error)
    
    return est_T