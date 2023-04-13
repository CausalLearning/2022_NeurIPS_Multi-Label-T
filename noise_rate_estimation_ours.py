# our estimator
import torch
import sys 
import numpy as np
# Our estimator
import tools
import copy
from sklearn.mixture import GaussianMixture
import torch.nn as nn

def loss_func_bce(out, batch_y):
    bce = torch.nn.BCELoss()
    loss = bce(torch.sigmoid(out), batch_y.float())
    return loss

def loss_func_bce2(out, batch_y):
    bce = torch.nn.BCELoss(reduction='sum')
    loss = bce(torch.sigmoid(out), batch_y.float())
    return loss
    
def estimate_noise_rate(model,train_loader, val_loader, estimate_loader,optimizer_es,args,true_tm):
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
       
    val_array = np.array(val_list)
    val_array_2 = np.array(val_list_2)
    model_index_all = np.argmax(-val_array)
    model_index_all_2 = np.argmax(val_array_2)
    print('model_index',model_index_all,model_index_all_2)
    model.load_state_dict(best_state_dict)
    
    th=args.sample_th
    bce = torch.nn.BCELoss(reduction='none')
    
    Y = copy.deepcopy(estimate_loader.dataset.labels)
    
    print('Y',Y[0])
    Y[Y==-1]=0
    select_vec=np.ones_like(Y)

    true_train_labels=copy.deepcopy(args.true_train_labels)
    true_train_labels[true_train_labels==-1]=0
    True_T = true_tm
    
    # estimation
    for i in range(args.nc):
        model_index = args.sample_epoch-1
        epoch_list = [model_index-4,model_index-3,model_index-2,model_index-1,model_index]
        print('epoch_list',epoch_list)
            
        for z in range(2):
            all_loss=[]
            for j in epoch_list:
                output = A[i, j,:,1]
                output = output[Y[:,i]==z]
                target = Y[:,i][Y[:,i]==z]
                losses = bce(output, torch.from_numpy(target).float())              
                losses = (losses-losses.min())/(losses.max()-losses.min()+1e-30) 
                all_loss.append(losses)

            history = torch.stack(all_loss)
            input_loss = history[-5:].mean(0)
            input_loss = input_loss.reshape(-1,1)
        
            gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
            gmm.fit(input_loss)
            prob = gmm.predict_proba(input_loss)
            prob = prob[:,gmm.means_.argmin()]
            select_vec[Y[:,i]==z,i]=(prob>th)
            
        if(True_T[i,0,1]==0): #  for multi-label learning with missing labels
            select_vec[Y[:,i]==1,i]=1
        if(True_T[i,1,0]==0): #  for partial multi-label learning
            select_vec[Y[:,i]==0,i]=1

        
        print('class ',i,' positive ', sum(Y[:,i]),' negative ', sum(1-Y[:,i]), ' selected positive ', sum(Y[select_vec[:,i]==1,i]),' negative ', sum(1-Y[select_vec[:,i]==1,i]))
        
        print('positive precision',sum((true_train_labels[:,i]==1)*(select_vec[:,i]==1)*(Y[:,i]==1))/sum(Y[select_vec[:,i]==1,i]),  'negative precision', sum((true_train_labels[:,i]==0)*(select_vec[:,i]==1)*(Y[:,i]==0))/sum(1-Y[select_vec[:,i]==1,i]))    
    
    error=0
    est_T =  np.zeros_like(True_T)
    temp_list=[]
    for i in range(args.nc):
        temp_estimation=[]
        selected_i=select_vec[:,i]
        selected_Y_i = Y[selected_i==1,:]
        if(sum(selected_Y_i[:,i])<50 or sum(1-selected_Y_i[:,i])<50):
            continue
        for k in range(args.nc):
            if(i==k):
                continue
            print('i',i,' k',k)
            P_noisy_11=sum(Y[:,i]*Y[:,k])/len(Y)
            P_noisy_10=sum(Y[:,i]*(1-Y[:,k]))/len(Y)
            P_noisy_01=sum((1-Y[:,i])*Y[:,k])/len(Y)
            P_noisy_00=1-P_noisy_11-P_noisy_10-P_noisy_01
            
            Pc_clean_1_noisy_1=sum(selected_Y_i[:,k]*selected_Y_i[:,i])/sum(selected_Y_i[:,i])
            Pc_clean_1_noisy_0=sum((1-selected_Y_i[:,k])*selected_Y_i[:,i])/sum(selected_Y_i[:,i])
            Pc_clean_0_noisy_1=sum(selected_Y_i[:,k]*(1-selected_Y_i[:,i]))/sum(1-selected_Y_i[:,i])
            Pc_clean_0_noisy_0=sum((1-selected_Y_i[:,k])*(1-selected_Y_i[:,i]))/sum(1-selected_Y_i[:,i])
        
            pi_1 = -(1.0*(P_noisy_00 - 1.0*Pc_clean_0_noisy_0 + P_noisy_10))/(Pc_clean_0_noisy_0 - 1.0*Pc_clean_1_noisy_0)
            print( 'pi_1',pi_1 )
            
            if( pi_1<0 or pi_1>1 ):
                continue
    
            P_clean_1_noisy_1=Pc_clean_1_noisy_1*pi_1
            P_clean_1_noisy_0=Pc_clean_1_noisy_0*pi_1
            P_clean_0_noisy_1=Pc_clean_0_noisy_1*(1-pi_1)
            P_clean_0_noisy_0=Pc_clean_0_noisy_0*(1-pi_1)

            lo_i_0 = -(1.0*P_clean_1_noisy_0*P_noisy_11 - 1.0*P_clean_1_noisy_1*P_noisy_10)/(P_clean_0_noisy_0*P_clean_1_noisy_1 - 1.0*P_clean_0_noisy_1*P_clean_1_noisy_0)
            lo_i_1 =(1.0*(P_clean_0_noisy_0*P_noisy_01 - 1.0*P_clean_0_noisy_1*P_noisy_00))/(P_clean_0_noisy_0*P_clean_1_noisy_1 - 1.0*P_clean_0_noisy_1*P_clean_1_noisy_0)
            print('lo_i_0, lo_i_1',lo_i_0, lo_i_1)
            
            if(lo_i_0<0 and lo_i_0>-0.3/args.nc):
                lo_i_0=0
            if(lo_i_1<0 and lo_i_1>-0.3/args.nc):
                lo_i_1=0
            if(lo_i_1<-0.3/args.nc or lo_i_1<-0.3/args.nc):
                continue
                
            if(True_T[i,0,1]==0): #  for multi-label learning with missing labels
                lo_i_0=0
            if(True_T[i,1,0]==0): #  for partial multi-label learning
                lo_i_1=0

            if(lo_i_0>=0 and lo_i_0<=1 and lo_i_1>=0 and lo_i_1<=1 and lo_i_0+lo_i_1<1):
                T=np.array([[1-lo_i_0, lo_i_0], [lo_i_1,1-lo_i_1]])
                
                estimate_error = tools.error(T, True_T[i])
                print('class i ', i, ' class k ',k,' our estimation', T[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]], 'error', estimate_error,'\n')
                temp_estimation.append(T[range(2),[1,0]])
                continue             
            else:
                continue
        # select best one from temp estimations
        if(len(temp_estimation)==0):
            temp_list.append(i)
        else:
            temp_error=[0]*len(temp_estimation)
            for k in range(args.nc):
                if(i==k):
                    continue
                P_noisy_11=sum(Y[:,i]*Y[:,k])/len(Y)
                P_noisy_10=sum(Y[:,i]*(1-Y[:,k]))/len(Y)
                P_noisy_01=sum((1-Y[:,i])*Y[:,k])/len(Y)
                P_noisy_00=1-P_noisy_11-P_noisy_10-P_noisy_01
                
                selected_i=select_vec[:,i]
                selected_Y_i = Y[selected_i==1,:]
                
                Pc_clean_1_noisy_1=sum(selected_Y_i[:,k]*selected_Y_i[:,i])/sum(selected_Y_i[:,i])
                Pc_clean_1_noisy_0=sum((1-selected_Y_i[:,k])*selected_Y_i[:,i])/sum(selected_Y_i[:,i])
                Pc_clean_0_noisy_1=sum(selected_Y_i[:,k]*(1-selected_Y_i[:,i]))/sum(1-selected_Y_i[:,i])
                Pc_clean_0_noisy_0=sum((1-selected_Y_i[:,k])*(1-selected_Y_i[:,i]))/sum(1-selected_Y_i[:,i])       
              
                
                pi_1 = -(1.0*(P_noisy_00 - 1.0*Pc_clean_0_noisy_0 + P_noisy_10))/(Pc_clean_0_noisy_0 - 1.0*Pc_clean_1_noisy_0)

       
                P_clean_1_noisy_1=Pc_clean_1_noisy_1*pi_1
                P_clean_1_noisy_0=Pc_clean_1_noisy_0*pi_1
                P_clean_0_noisy_1=Pc_clean_0_noisy_1*(1-pi_1)
                P_clean_0_noisy_0=Pc_clean_0_noisy_0*(1-pi_1)
                
                for index, lo in enumerate(temp_estimation): 
                    lo_i_0, lo_i_1 =lo[0],lo[1]
                    temp_error[index] += abs(lo_i_0*(P_clean_0_noisy_0*P_clean_1_noisy_1 - 1.0*P_clean_0_noisy_1*P_clean_1_noisy_0)+(1.0*P_clean_1_noisy_0*P_noisy_11 - 1.0*P_clean_1_noisy_1*P_noisy_10))
                    temp_error[index] += abs(lo_i_1*(P_clean_0_noisy_0*P_clean_1_noisy_1 - 1.0*P_clean_0_noisy_1*P_clean_1_noisy_0) - (1.0*(P_clean_0_noisy_0*P_noisy_01 - 1.0*P_clean_0_noisy_1*P_noisy_00)))
            temp_error=np.array(temp_error)
            print('temp_error',temp_error)
            lo=temp_estimation[np.argmin(temp_error)]
            lo_i_0, lo_i_1 =lo[0],lo[1]
            T=np.array([[1-lo_i_0, lo_i_0], [lo_i_1,1-lo_i_1]])
            
            estimate_error = tools.error(T, True_T[i])
            error+=estimate_error
            est_T[i] = T
            print('class', i, ' final ours estimation', T[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]], 'error', estimate_error,'\n')

    # use dualT estimator for unsolvable classes
    for i in temp_list:
        val_array = np.array(val_list_list[i]) # we use the val loss here for selecting each class's model
        model_index = np.argmax(-val_array)
        print('model_index',model_index)
        prob_=copy.deepcopy(A[i])
        transition_matrix_ = tools.fit(prob_[model_index, :, :], 2, False)
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
        print('class', i, 'T estimation max',T[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]], 'error', estimate_error) 
        
        pred= np.argmax(prob_[model_index, :, :],axis=-1)
        T_spadesuit = np.zeros((2,2))    
        for j in range(len(Y)): 
            T_spadesuit[int(pred[j])][int(Y[j,i])]+=1    
        T_spadesuit = np.array(T_spadesuit)
        sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(2,1)).transpose()
        T_spadesuit = T_spadesuit/sum_matrix
        T_spadesuit = np.nan_to_num(T_spadesuit)
        dual_t_matrix = np.matmul(T_, T_spadesuit)

        if(True_T[i,0,1]==0):  #  for multi-label learning with missing labels
            dual_t_matrix[0,1]=0
            dual_t_matrix[0,0]=1
        if(True_T[i,1,0]==0):  #  for partial multi-label learning
            dual_t_matrix[1,0]=0
            dual_t_matrix[1,1]=1 
        
        
        estimate_error = tools.error(dual_t_matrix, True_T[i])
        est_T[i]= dual_t_matrix
        error+=estimate_error
        print('class', i, 'Dual-T estimation max', dual_t_matrix[range(2),[1,0]], 'True_T', True_T[i,range(2),[1,0]], 'error', estimate_error,'\n')  
        
    print('total error', error)
    
    return est_T