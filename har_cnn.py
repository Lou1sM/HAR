import sys
import os
import argparse
import math
from pdb import set_trace
from scipy import stats
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils import data
from dl_utils import misc, label_funcs


class StepDataset(data.Dataset):
    def __init__(self,x,y,device,window_size,step_size,transforms=[]):
        self.device=device
        self.x, self.y = x,y
        self.window_size = window_size
        self.step_size = step_size
        for transform in transforms:
            self.x = transform(self.x)
        self.x, self.y = self.x.to(self.device),self.y.to(self.device)
    def __len__(self): return (len(self.x)-self.window_size)//self.step_size + 1
    def __getitem__(self,idx):
        batch_x = self.x[idx*self.step_size:(idx*self.step_size) + self.window_size].unsqueeze(0)
        batch_y = self.y[idx]
        return batch_x, batch_y, idx

    def temporal_consistency_loss(self,sequence):
        total_loss = 0
        for start_idx in range(len(sequence)-self.window_size):
            window = sequence[start_idx:start_idx+self.window_size]
            mu = window.mean(axis=0)
            window_var = sum([(item-mu) for item in self.window])/self.window_size
            if window_var < self.split_thresh: total_loss += window_var
        return total_loss

class EncByLayer(nn.Module):
    def __init__(self,x_filters,y_filters,x_strides,y_strides,max_pools,verbose):
        super(EncByLayer,self).__init__()
        self.verbose = verbose
        num_layers = len(x_filters)
        assert all(len(x)==num_layers for x in (y_filters,x_strides,y_strides,max_pools))
        ncvs = [1]+[4*2**i for i in range(num_layers)]
        conv_layers = [nn.Sequential(
                nn.Conv2d(ncvs[i],ncvs[i+1],(x_filters[i],y_filters[i]),(x_strides[i],y_strides[i])),
                nn.BatchNorm2d(ncvs[i+1]),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(max_pools[i])
                )
            for i in range(num_layers)]
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self,x):
        if self.verbose: print(x.shape)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if self.verbose: print(x.shape)
        return x

class Var_BS_MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Var_BS_MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.LeakyReLU(0.3)
        self.fc2 = nn.Linear(hidden_size,output_size)
        #self.act2 = nn.Softmax()

    def forward(self,x):
        x = self.fc1(x)
        if x.shape[0] != 1:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        #x = self.act2(x)
        return x

class HARLearner():
    def __init__(self,dset_train,dset_val,enc,mlp,batch_size,num_classes):
        self.dset_train = dset_train
        self.dset_val = dset_val
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.enc = enc
        self.mlp = mlp
        self.pseudo_label_lf = nn.CrossEntropyLoss(reduction='none')
        self.rec_lf = nn.MSELoss()

        #self.dl = data.DataLoader(self.dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),batch_size,drop_last=True),pin_memory=False)
        #self.determin_dl = data.DataLoader(self.dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),batch_size,drop_last=False),pin_memory=False)

        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=ARGS.enc_lr)
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(),lr=ARGS.mlp_lr)

    def get_latents(self):
        self.enc.eval()
        collected_latents = []
        for idx, (xb,yb,tb) in enumerate(self.determin_dl):
            batch_latents = self.enc(xb)
            batch_latents = batch_latents.view(batch_latents.shape[0],-1).detach().cpu().numpy()
            collected_latents.append(batch_latents)
        collected_latents = np.concatenate(collected_latents,axis=0)
        return collected_latents

    def train(self,num_epochs,frac_gt_labels,selected_acts,exp_dir):
        best_gt_acc = 0
        best_non_gt_acc = 0
        best_non_gt_f1 = 0
        dl_train = data.DataLoader(self.dset_train,batch_sampler=data.BatchSampler(data.RandomSampler(self.dset_train),self.batch_size,drop_last=False),pin_memory=False)
        dl_val = data.DataLoader(self.dset_val,batch_sampler=data.BatchSampler(data.RandomSampler(self.dset_val),self.batch_size,drop_last=False),pin_memory=False)
        if frac_gt_labels == 0:
            gt_idx = np.array([], dtype=np.int)
        elif frac_gt_labels <= 0.5:
            gt_idx = np.arange(len(self.dset_train), step=int(1/frac_gt_labels))
        else:
            non_gt_idx = np.arange(len(self.dset_train), step=int(1/(1-frac_gt_labels)))
            gt_idx = np.delete(np.arange(len(self.dset_train)),non_gt_idx)
        gt_mask = torch.zeros_like(self.dset_train.y)
        gt_mask[gt_idx] = 1
        assert abs(len(gt_idx)/len(self.dset_train) - frac_gt_labels) < .01
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_pseudo_label_losses = []
            epoch_losses = []
            train_pred_list = []
            train_idx_list = []
            val_pred_list = []
            val_gt_list = []
            val_idx_list = []
            best_loss = np.inf
            best_train_acc = 0
            best_train_f1 = 0
            self.enc.train()
            self.mlp.train()
            for batch_idx, (xb,yb,idx) in enumerate(dl_train):
                latent = self.enc(xb)
                batch_mask = gt_mask[idx]
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                label_loss = self.pseudo_label_lf(label_pred,yb.long())
                loss = (label_loss*batch_mask).mean()
                if math.isnan(loss): set_trace()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                train_pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
                #total_gt_list.append(yb.detach().cpu().numpy())
                train_idx_list.append(idx.detach().cpu().numpy())
                if ARGS.test: break
            #total_gt_array = self.dset_train.y.detach().cpu().numpy()[total_idx_array]
            train_pred_array = np.concatenate(train_pred_list)
            train_idx_array = np.concatenate(train_idx_list)
            train_pred_array_ordered = np.array([item[0] for item in sorted(zip(train_pred_array,train_idx_array),key=lambda x:x[1])])
            train_acc = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.accuracy(train_pred_array_ordered,self.dset_train.y.detach().cpu().numpy())
            train_f1 = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.mean_f1(train_pred_array_ordered,self.dset_train.y.detach().cpu().numpy())
            if ARGS.test or len(gt_idx) == 0 or train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_f1 = train_f1
            self.enc.eval()
            self.mlp.eval()
            best_val_acc = 0
            best_val_f1 = 0
            for batch_idx, (xb,yb,idx) in enumerate(dl_val):
                latent = self.enc(xb)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                val_pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
                val_gt_list.append(yb.detach().cpu().numpy())
                val_idx_list.append(idx.detach().cpu().numpy())
                if ARGS.test: break
            val_pred_array = np.concatenate(val_pred_list)
            val_idx_array = np.concatenate(val_idx_list)
            val_pred_array_ordered = np.array([item[0] for item in sorted(zip(val_pred_array,val_idx_array),key=lambda x:x[1])])
            if ARGS.test: break
            val_acc = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.accuracy(val_pred_array_ordered,self.dset_val.y.detach().cpu().numpy())
            val_f1 = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.mean_f1(val_pred_array_ordered,self.dset_val.y.detach().cpu().numpy())
            if not ARGS.suppress_prints:
                print(f'MLP gt acc: {val_acc}')
                print(f'MLP non-gt mean_f1: {val_f1}')
            if ARGS.test or len(gt_idx) == 0 or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                misc.torch_save({'enc':self.enc,'mlp':self.mlp},exp_dir,'best_model.pt')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
            if ARGS.test: continue
        misc.check_dir(exp_dir)
        summary_file_path = os.path.join(exp_dir,'summary.txt')
        print(f'Best Train Acc: {best_train_acc}')
        print(f'Best Train F1: {best_train_f1}')
        print(f'Best Val Acc: {best_val_acc}')
        print(f'Best Val f1: {best_val_f1}')
        with open(summary_file_path,'w') as f:
            f.write(f'Train Acc: {best_train_acc}\n')
            f.write(f'Train F1: {best_train_acc}\n')
            f.write(f'Val Acc: {best_val_acc}\n')
            f.write(f'Val F1: {best_val_acc}\n')
            f.write(str(ARGS))
        if ARGS.save:
            misc.torch_save({'enc':self.enc,'mlp':self.mlp},exp_dir,f'har_learner{ARGS.exp_name}.pt')
            misc.np_save(val_pred_array_ordered,exp_dir,f'preds{ARGS.exp_name}.npy')

    def cross_train(self,user_dsets,num_epochs,frac_gt_labels,selected_acts,exp_dir):
        best_gt_acc = 0
        best_non_gt_acc = 0
        results_matrix = np.zeros((len(user_dsets),len(user_dsets)))
        for dset_idx,dset in enumerate(user_dsets):
            dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),self.batch_size,drop_last=False),pin_memory=False)
            if frac_gt_labels == 0:
                gt_idx = np.array([], dtype=np.int)
            elif frac_gt_labels <= 0.5:
                gt_idx = np.arange(len(self.dset), step=int(1/frac_gt_labels))
            else:
                non_gt_idx = np.arange(len(self.dset), step=int(1/(1-frac_gt_labels)))
                gt_idx = np.delete(np.arange(len(self.dset)),non_gt_idx)
            gt_mask = torch.zeros_like(self.dset.y)
            gt_mask[gt_idx] = 1
            assert abs(len(gt_idx)/len(self.dset) - frac_gt_labels) < .01
            for epoch in range(num_epochs):
                pred_list = []
                idx_list = []
                best_acc = 0
                best_f1 = 0
                self.enc.train()
                self.mlp.train()
                for batch_idx, (xb,yb,idx) in enumerate(dl):
                    latent = self.enc(xb)
                    batch_mask = gt_mask[idx]
                    label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                    label_loss = self.pseudo_label_lf(label_pred,yb.long())
                    loss = (label_loss*batch_mask).mean()
                    if math.isnan(loss): set_trace()
                    loss.backward()
                    self.enc_opt.step(); self.enc_opt.zero_grad()
                    self.mlp_opt.step(); self.mlp_opt.zero_grad()
                    pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
                    idx_list.append(idx.detach().cpu().numpy())
                    if ARGS.test: break
                pred_array = np.concatenate(pred_list)
                idx_array = np.concatenate(idx_list)
                pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
                acc = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
                f1 = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
                if ARGS.test or len(gt_idx) == 0 or acc > best_acc:
                    best_acc = acc
                    best_f1 = f1
            results_matrix[dset_idx,dset_idx] = best_acc
            self.enc.eval()
            self.mlp.eval()
            for dset_idx_val,dset_val in enumerate(user_dsets):
                if dset_idx_val == dset_idx: continue
                best_val_acc = 0
                best_val_f1 = 0
                pred_list_val = []
                idx_list_val = []
                dl_val = data.DataLoader(dset_val,batch_sampler=data.BatchSampler(data.RandomSampler(dset_val),self.batch_size,drop_last=False),pin_memory=False)
                for batch_idx, (xb,yb,idx) in enumerate(dl_val):
                    latent = self.enc(xb)
                    label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                    pred_list_val.append(label_pred.argmax(axis=1).detach().cpu().numpy())
                    idx_list_val.append(idx.detach().cpu().numpy())
                    if ARGS.test: break
                pred_array_val = np.concatenate(pred_list_val)
                idx_array_val = np.concatenate(idx_list_val)
                pred_array_ordered_val = np.array([item[0] for item in sorted(zip(pred_array_val,idx_array_val),key=lambda x:x[1])])
                if ARGS.test: break
                val_acc = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.accuracy(pred_array_ordered_val,dset_val.y.detach().cpu().numpy())
                val_f1 = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.mean_f1(pred_array_ordered_val,dset_val.y.detach().cpu().numpy())
                if ARGS.test or len(gt_idx) == 0 or val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    misc.torch_save({'enc':self.enc,'mlp':self.mlp},exp_dir,'best_model.pt')
                results_matrix[dset_idx,dset_idx_val] = best_val_acc
        print(results_matrix)
        return results_matrix


def preproc_xys(x,y,step_size,window_size,action_name_dict):
    x = x[y!=0]
    y = y[y!=0]
    xnans = np.isnan(x).any(axis=1)
    x = x[~xnans]
    y = y[~xnans]
    x = x[y!=-1]
    y = y[y!=-1]
    num_windows = (len(x) - window_size)//step_size + 1
    mode_labels = np.concatenate([stats.mode(y[w*step_size:w*step_size + window_size]).mode for w in range(num_windows)])
    selected_ids = set(mode_labels)
    selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
    mode_labels, trans_dict, changed = label_funcs.compress_labels(mode_labels)
    assert len(selected_acts) == len(set(mode_labels))
    x = torch.tensor(x,device='cuda').float()
    y = torch.tensor(mode_labels,device='cuda').float()
    return x, y, selected_acts

def make_wisdm_v1_dset(args,subj_ids):
    activities_list = ['Jogging','Walking','Upstairs','Downstairs','Standing','Sitting']
    action_name_dict = dict(zip(range(len(activities_list)),activities_list))
    x = np.load('datasets/wisdm_v1/X.npy')
    y = np.load('datasets/wisdm_v1/y.npy')
    train_ids = subj_ids[:-2]
    val_ids = subj_ids[-2:]
    users = np.load('datasets/wisdm_v1/users.npy')
    train_idxs_to_user = np.zeros(users.shape[0]).astype(np.bool)
    for subj_id in train_ids:
        new_users = users==subj_id
        train_idxs_to_user = np.logical_or(train_idxs_to_user,new_users)
    val_idxs_to_user = np.zeros(users.shape[0]).astype(np.bool)
    for subj_id in val_ids:
        new_users = users==subj_id
        val_idxs_to_user = np.logical_or(val_idxs_to_user,new_users)
    x_train = x[train_idxs_to_user]
    y_train = y[train_idxs_to_user]
    x_val = x[val_idxs_to_user]
    y_val = y[val_idxs_to_user]
    xnans = np.isnan(x).any(axis=1)
    x = x[~xnans]
    y = y[~xnans]
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,action_name_dict)
    x_val,y_val,selected_acts = preproc_xys(x_val,y_val,args.step_size,args.window_size,action_name_dict)
    dset_train = StepDataset(x_train,y_train,device='cuda',window_size=args.window_size,step_size=args.step_size)
    dset_val = StepDataset(x_val,y_val,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset_train, dset_val, selected_acts

def make_wisdm_watch_dset(args,subj_ids):
    with open('datasets/wisdm-dataset/activity_key.txt') as f: r=f.readlines()
    activities_list = [x.split(' = ')[0] for x in r if ' = ' in x]
    action_name_dict = dict(zip(range(len(activities_list)),activities_list))
    train_ids = subj_ids[:-2]
    val_ids = subj_ids[-2:]
    x_train = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}.npy') for s in train_ids])
    y_train = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_labels.npy') for s in train_ids])
    x_val = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}.npy') for s in val_ids])
    y_val = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_labels.npy') for s in val_ids])
    certains_train = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_certains.npy') for s in train_ids])
    certains_val = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_certains.npy') for s in val_ids])
    x_train = x_train[certains_train]
    y_train = y_train[certains_train]
    x_val = x_val[certains_val]
    y_val = y_val[certains_val]
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,action_name_dict)
    x_val,y_val,selected_acts = preproc_xys(x_val,y_val,args.step_size,args.window_size,action_name_dict)
    dset_train = StepDataset(x_train,y_train,device='cuda',window_size=args.window_size,step_size=args.step_size)
    dset_val = StepDataset(x_val,y_val,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset_train, dset_val, selected_acts

def make_uci_dset(args,subj_ids):
    action_name_dict = {1:'walking',2:'walking upstairs',3:'walking downstairs',4:'sitting',5:'standing',6:'lying',7:'stand_to_sit',9:'sit_to_stand',10:'sit_to_lit',11:'lie_to_sit',12:'stand_to_lie',13:'lie_to_stand'}
    train_ids = subj_ids[:-2]
    val_ids = subj_ids[-2:]
    x_train = np.concatenate([np.load(f'datasets/UCI2/np_data/user{s}.npy') for s in train_ids])
    y_train = np.concatenate([np.load(f'datasets/UCI2/np_data/user{s}_labels.npy') for s in train_ids])
    x_val = np.concatenate([np.load(f'datasets/UCI2/np_data/user{s}.npy') for s in val_ids])
    y_val = np.concatenate([np.load(f'datasets/UCI2/np_data/user{s}_labels.npy') for s in val_ids])
    x_train = x_train[y_train<7] # Labels still begin at 1 at this point as
    y_train = y_train[y_train<7] # haven't been compressed, so select 1,..,6
    #x_train = x_train[y_train!=-1]
    #y_train = y_train[y_train!=-1]
    x_val = x_val[y_val<7] # Labels still begin at 1 at this point as
    y_val = y_val[y_val<7] # haven't been compressed, so select 1,..,6
    #x_val = x_val[y_val!=-1]
    #y_val = y_val[y_val!=-1]
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,action_name_dict)
    x_val,y_val,selected_acts = preproc_xys(x_val,y_val,args.step_size,args.window_size,action_name_dict)
    dset_train = StepDataset(x_train,y_train,device='cuda',window_size=args.window_size,step_size=args.step_size)
    dset_val = StepDataset(x_val,y_val,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset_train, dset_val, selected_acts

def make_pamap_dset(args,subj_ids):
    action_name_dict = {1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'Nordic walking',9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing soccer',24:'rope jumping'}
    train_ids = subj_ids[:-2]
    val_ids = subj_ids[-2:]
    x_train = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{s}.npy') for s in train_ids])
    y_train = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{s}_labels.npy') for s in train_ids])
    x_val = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{s}.npy') for s in val_ids])
    y_val = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{s}_labels.npy') for s in val_ids])
    x_train,y_train,selected_acts = preproc_xys(x_train,y_train,args.step_size,args.window_size,action_name_dict)
    x_val,y_val,selected_acts = preproc_xys(x_val,y_val,args.step_size,args.window_size,action_name_dict)
    dset_train = StepDataset(x_train,y_train,device='cuda',window_size=args.window_size,step_size=args.step_size)
    dset_val = StepDataset(x_val,y_val,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset_train, dset_val, selected_acts

def train(args,subj_ids):
    prep_start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dset=='PAMAP': dset_train, dset_val, selected_acts = make_pamap_dset(args,subj_ids)
    elif args.dset == 'UCI-raw': dset_train, dset_val, selected_acts = make_uci_dset(args,subj_ids)
    elif args.dset == 'WISDM-watch': dset_train, dset_val, selected_acts = make_wisdm_watch_dset(args,subj_ids)
    elif args.dset == 'WISDM-v1': dset_train, dset_val, selected_acts = make_wisdm_v1_dset(args,subj_ids)
    num_labels = label_funcs.get_num_labels(dset_train.y)
    if args.dset == 'PAMAP':
        x_filters = (50,40,3,3)
        y_filters = (5,3,2,1)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = (2,2,2,2)
    elif args.dset == 'UCI-raw':
        x_filters = (50,40,4,4)
        y_filters = (1,1,3,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
    elif args.dset == 'WISDM-watch':
        x_filters = (50,40,4,4)
        y_filters = (1,1,3,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
    elif args.dset == 'WISDM-v1':
        x_filters = (50,40,4,4)
        y_filters = (1,1,2,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = ((2,1),(3,1),(2,1),1)
    enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,verbose=args.verbose)
    mlp = Var_BS_MLP(32,25,num_labels)
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        mlp.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    mlp.cuda()

    har = HARLearner(enc=enc,mlp=mlp,dset_train=dset_train,dset_val=dset_val,batch_size=args.batch_size,num_classes=num_labels)
    exp_dir = os.path.join(f'experiments/{args.exp_name}')

    train_start_time = time.time()
    har.train(args.num_epochs,args.frac_gt_labels,selected_acts=selected_acts,exp_dir=exp_dir)

    train_end_time = time.time()
    total_prep_time = misc.asMinutes(train_start_time-prep_start_time)
    total_train_time = misc.asMinutes(train_end_time-train_start_time)
    print(f"Prep time: {total_prep_time}\tTrain time: {total_train_time}")


if __name__ == "__main__":

    dset_options = ['PAMAP','UCI-raw','WISDM-v1','WISDM-watch']
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--alpha',type=float,default=.5)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--dset',type=str,default='PAMAP',choices=dset_options)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--exp_name',type=str,default="jim")
    parser.add_argument('--frac_gt_labels',type=float,default=0.1)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--subj_ids',type=str,nargs='+',default=['first'])
    parser.add_argument('--suppress_prints',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.test and ARGS.save:
        print("Shouldn't be saving for a test run"); sys.exit()
    if ARGS.test: ARGS.num_meta_epochs = 2
    else: import umap
    if ARGS.dset == 'PAMAP':
        all_possible_ids = [str(x) for x in range(101,110)]
    elif ARGS.dset in ['UCI-pre','UCI-raw']:
        def two_digitify(x): return '0' + str(x) if len(str(x))==1 else str(x)
        all_possible_ids = [two_digitify(x) for x in range(1,30)]
    elif ARGS.dset == 'WISDM-watch':
        all_possible_ids = [str(x) for x in range(1600,1651)]
    elif ARGS.dset == 'WISDM-v1':
        all_possible_ids = [str(x) for x in range(1,37)] #Paper says 29 users but ids go up to 36
    if ARGS.all_subjs: subj_ids=all_possible_ids
    elif ARGS.subj_ids == ['first']: subj_ids = all_possible_ids[:1]
    else: subj_ids = ARGS.subj_ids
    bad_ids = [x for x in subj_ids if x not in all_possible_ids]
    if len(bad_ids) > 0:
        print(f"You have specified non-existent ids: {bad_ids}"); sys.exit()
    if ARGS.parallel:
        train(ARGS,subj_ids=subj_ids)
    else:
        orig_name = ARGS.exp_name
        for subj_id in subj_ids:
            print(f"Training and predicting on id {subj_id}")
            ARGS.exp_name = f"{orig_name}{subj_id}"
            train(ARGS,subj_ids=[subj_id])
