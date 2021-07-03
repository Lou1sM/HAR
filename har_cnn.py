from pprint import pprint
from scipy.stats import multivariate_normal
import sys
import os
import pickle
import matplotlib.pyplot as plt
import copy
import argparse
import math
from pdb import set_trace
from scipy import stats
from hmmlearn import hmm
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from dl_utils import misc, tensor_funcs, label_funcs


def display(latents_to_display):
    umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,n_components=2,random_state=42).fit_transform(latents_to_display.squeeze())
    misc.scatter_clusters(umapped_latents,labels=None,show=True)

class Preprocced_Dataset(data.Dataset):
    def __init__(self,x,y,device):
        self.device=device
        self.x, self.y = x,y
        self.x, self.y = self.x.to(self.device),self.y.to(self.device)
    def __len__(self): return len(self.x)
    def __getitem__(self,idx):
        batch_x = self.x[idx]
        batch_y = self.y[idx]
        return batch_x, batch_y, idx

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
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if self.verbose: print(x.shape)
        return x

class Dense_Net(nn.Module):
    def __init__(self,*sizes):
        super(Dense_Net,self).__init__()
        self.net= nn.Sequential(
            *[nn.Sequential(*[nn.Linear(sizes[i-1],sizes[i]),nn.BatchNorm1d(sizes[i]),nn.LeakyReLU(sizes[i])]) for i in range(1,len(sizes)-1)], nn.Linear(sizes[-2],sizes[-1]))
        #self.fc1 = nn.Linear(input_size,hidden_size1)
        #self.bn1 = nn.BatchNorm1d(hidden_size1)
        #self.act1 = nn.LeakyReLU(0.3)
        #self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        #self.bn2 = nn.BatchNorm1d(hidden_size2)
        #self.act2 = nn.LeakyReLU(0.3)
        #self.fc3 = nn.Linear(hidden_size2,output_size)

    def forward(self,x):
        x = self.net(x)
        #x = self.fc1(x)
        #x = self.bn1(x)
        #x = self.act1(x)
        #x = self.fc2(x)
        #x = self.bn2(x)
        #x = self.act2(x)
        #x = self.fc3(x)
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
    def __init__(self,dset,enc,dec,mlp,device,batch_size,num_classes):
        self.dset = dset
        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.enc = enc.to(device)
        self.dec = dec.to(device)
        self.mlp = mlp.to(device)
        self.pseudo_label_lf = nn.CrossEntropyLoss(reduction='none')
        self.rec_lf = nn.MSELoss()

        self.dl = data.DataLoader(self.dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),batch_size,drop_last=True),pin_memory=False)
        self.determin_dl = data.DataLoader(self.dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),batch_size,drop_last=False),pin_memory=False)

        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=ARGS.enc_lr)
        self.dec_opt = torch.optim.Adam(self.dec.parameters(),lr=ARGS.dec_lr)
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

    def rec_train(self,num_epochs):
        self.enc.train()
        self.dec.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            best_loss = np.inf
            for idx, (xb,yb,tb) in enumerate(self.dl):
                latent = self.enc(xb)
                latent = tensor_funcs.noiseify(latent,ARGS.noise)
                pred = self.dec(latent)
                loss = self.rec_lf(pred,xb)
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                epoch_loss += (loss.item()-epoch_loss)/(idx+1)
                if ARGS.test: break
            if ARGS.test: break
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
        torch.save(self.enc.state_dict(),'enc_pretrained.pt')
        torch.save(self.dec.state_dict(),'dec_pretrained.pt')

    def pseudo_label_train(self,mask,probs,pseudo_labels,num_epochs,writer,gt_idx,meta_loop_idx,exp_dir,prev_best_acc):
        self.enc.train()
        if isinstance(pseudo_labels,np.ndarray):
            pseudo_labels = torch.tensor(pseudo_labels)
        probs = torch.tensor(probs,device=self.device)
        if isinstance(self.dset,Preprocced_Dataset):
            pseudo_label_dset = Preprocced_Dataset(self.dset.x,pseudo_labels,device='cuda')
        else:
            pseudo_label_dset = StepDataset(self.dset.x,pseudo_labels,device='cuda',window_size=self.dset.window_size,step_size=self.dset.step_size)
        pseudo_label_dl = data.DataLoader(pseudo_label_dset,batch_sampler=data.BatchSampler(data.RandomSampler(pseudo_label_dset),self.batch_size,drop_last=False),pin_memory=False)
        all_pseudo_label_losses = []
        start_indexing_at = meta_loop_idx*num_epochs*len(pseudo_label_dl)
        best_gt_acc = prev_best_acc
        best_non_gt_acc = 0
        best_non_gt_f1 = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_rec_losses = []
            epoch_pseudo_label_losses = []
            epoch_losses = []
            total_pred_list = []
            total_gt_list = []
            total_idx_list = []
            best_loss = np.inf
            assert (pseudo_label_dset.x==self.dset.x).all()
            for batch_idx, (xb,yb,idx) in enumerate(pseudo_label_dl):
                latent = self.enc(xb)
                latent = tensor_funcs.noiseify(latent,ARGS.noise)
                pseudo_label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                pseudo_label_loss = self.pseudo_label_lf(pseudo_label_pred,yb.long())
                try: pseudo_label_loss = (pseudo_label_loss*probs[idx]).mean()
                except: set_trace()
                rec_pred = self.dec(latent)
                rec_loss = self.rec_lf(rec_pred,xb)
                loss = pseudo_label_loss.mean() + rec_loss
                if math.isnan(loss): set_trace()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                total_idx = start_indexing_at + epoch*len(pseudo_label_dl) + batch_idx
                total_pred_list.append(pseudo_label_pred.argmax(axis=1).detach().cpu().numpy())
                total_gt_list.append(yb.detach().cpu().numpy())
                total_idx_list.append(idx.detach().cpu().numpy())
                writer.add_scalar('Loss/pseudo_label_loss',pseudo_label_loss.item(),total_idx)
                epoch_pseudo_label_losses.append(pseudo_label_loss.item())
                all_pseudo_label_losses.append(pseudo_label_loss)
                epoch_losses.append(loss)
                if ARGS.test: break
            total_pred_array = np.concatenate(total_pred_list)
            total_idx_array = np.concatenate(total_idx_list)
            total_gt_array = self.dset.y.detach().cpu().numpy()[total_idx_array]
            total_pred_array_ordered = np.array([item[0] for item in sorted(zip(total_pred_array,total_idx_array),key=lambda x:x[1])])
            if ARGS.test: break
            non_gt_acc2 = label_funcs.accuracy(np.delete(total_pred_array,gt_idx),np.delete(total_gt_array,gt_idx))
            non_gt_acc = label_funcs.accuracy(np.delete(total_pred_array_ordered,gt_idx),np.delete(self.dset.y.detach().cpu().numpy(),gt_idx))
            non_gt_f1 = mean_f1(np.delete(total_pred_array_ordered,gt_idx),np.delete(self.dset.y.detach().cpu().numpy(),gt_idx))
            gt_acc = -1 if len(gt_idx) == 0 else label_funcs.accuracy(total_pred_array_ordered[gt_idx],self.dset.y.detach().cpu().numpy()[gt_idx])
            gt_mean_f1 = -1 if len(gt_idx) == 0 else mean_f1(total_pred_array_ordered[gt_idx],self.dset.y.detach().cpu().numpy()[gt_idx])
            full_acc = label_funcs.accuracy(total_pred_array,total_gt_array)
            full_acc2 = label_funcs.accuracy(total_pred_array_ordered,self.dset.y.detach().cpu().numpy())
            assert full_acc==full_acc2
            if not ARGS.suppress_prints:
                print(f'MLP non-gt acc: {non_gt_acc} {non_gt_acc2}')
                print(f'MLP gt acc: {gt_acc}')
                print(f'MLP non-gt mean_f1: {non_gt_f1}')
                print(f'MLP gt mean_f1: {gt_mean_f1}')
            if gt_acc > best_gt_acc:
                best_gt_acc=gt_acc
                best_non_gt_acc=non_gt_acc
                best_non_gt_f1=non_gt_f1
                misc.torch_save({'enc':self.enc,'dec':self.dec,'mlp':self.mlp},exp_dir,'best_model.pt')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
            assert (pseudo_label_dset.x==self.dset.x).all()
        return total_pred_array_ordered, best_gt_acc, best_non_gt_acc, best_non_gt_f1

    def train_meta_loop(self,num_pre_epochs,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts,frac_gt_labels,exp_dir):
        writer = SummaryWriter()
        self.rec_train(num_pre_epochs)
        if frac_gt_labels == 0:
            gt_idx = np.array([], dtype=np.int)
        elif frac_gt_labels <= 0.5:
            gt_idx = np.arange(len(self.dset), step=int(1/frac_gt_labels))
        else:
            non_gt_idx = np.arange(len(self.dset), step=int(1/(1-frac_gt_labels)))
            gt_idx = np.delete(np.arange(len(self.dset)),non_gt_idx)
        assert abs(len(gt_idx)/len(self.dset) - frac_gt_labels) < .01
        old_pred_labels = -np.ones(self.dset.y.shape)
        plt.switch_backend('agg')
        prev_weighted_probs = np.zeros((len(self.dset),self.num_classes))
        best_gt_acc = 0
        best_non_gt_acc = 0
        best_mlp_acc = 0
        peak_acc_at = 0
        for epoch_num in range(num_meta_epochs):
            if not ARGS.suppress_prints: print('Meta Epoch:', epoch_num)
            if ARGS.test:
                num_tiles = len(self.dset.y)//self.num_classes
                new_pred_labels = np.tile(np.arange(self.num_classes),num_tiles).astype(np.long)
                additional = len(self.dset.y) - (num_tiles*self.num_classes)
                if additional > 0:
                    new_pred_labels = np.concatenate((new_pred_labels,np.ones(additional)))
                new_pred_labels = new_pred_labels.astype(np.long)
                mask = torch.ones(len(self.dset.y)).bool()
                weighted_probs = torch.ones(len(self.dset.y)).bool()
                probs = torch.ones(len(self.dset.y)).bool()
                old_pred_labels = new_pred_labels
            else:
                latents = self.get_latents()
                if ARGS.umap_abl: umapped_latents = latents
                else:
                    umapped_latents = umap.UMAP(min_dist=0,n_neighbors=60,n_components=2,random_state=42).fit_transform(latents.squeeze())
                model = hmm.GaussianHMM(self.num_classes,'full')
                model.params = 'mc'
                model.init_params = 'mc'
                model.startprob_ = np.ones(self.num_classes)/self.num_classes
                num_action_blocks = len([item for idx,item in enumerate(self.dset.y) if self.dset.y[idx-1] != item])
                prob_new_action = num_action_blocks/len(self.dset)
                model.transmat_ = (np.eye(self.num_classes) * (1-prob_new_action)) + (np.ones((self.num_classes,self.num_classes))*prob_new_action/self.num_classes)
                model.fit(umapped_latents)
                new_pred_labels = model.predict(umapped_latents)
                new_pred_probs = model.predict_proba(umapped_latents)
                mask = torch.ones(len(self.dset.y))
                fig = plt.figure()
                misc.scatter_clusters(umapped_latents,self.dset.y,show=False)
                writer.add_figure(f'umapped_latents/{epoch_num}',fig)
                if ARGS.save: np.save('test_umapped_latents.npy',umapped_latents)
                #subsample_size = min(30000,gt_idx.shape[0])
                subsample_size = min(30000,self.dset.y.shape[0])
                #trans_dict, leftovers = label_funcs.get_trans_dict(new_pred_labels[gt_idx],self.dset.y[gt_idx],subsample_size=subsample_size)
                trans_dict, leftovers = label_funcs.get_trans_dict(new_pred_labels,self.dset.y,subsample_size=subsample_size)
                new_pred_labels = np.array([trans_dict[l] for l in new_pred_labels])
                new_pred_labels[gt_idx] = self.dset.y.detach().cpu().int().numpy()[gt_idx]
                new_pred_labels = new_pred_labels.astype(np.int)
                mvns = [multivariate_normal(m,c) for m,c in zip(model.means_,model.covars_)]
                probs=np.array([mvns[label].pdf(mean) for mean,label in zip(umapped_latents,new_pred_labels)])
                weighted_probs = probs if epoch_num==0 else ARGS.alpha*probs + (1-ARGS.alpha)*prev_weighted_probs
                weighted_probs *= new_pred_probs.max(axis=1)
                # Scale so max prob is 1 for each dpoint
                weighted_probs = weighted_probs/max(weighted_probs)
                assert weighted_probs.max() == 1
                if ARGS.probs_abl1: weighted_probs = np.ones(probs.shape)
                if ARGS.probs_abl2: weighted_probs = np.zeros(probs.shape)
                if not ARGS.suppress_prints: print(weighted_probs.mean())
                weighted_probs[gt_idx] = 1
            if not ARGS.suppress_prints:
                print('pseudo label training')
                print('best_gt_acc:', best_gt_acc)
                print('best_non_gt_acc:', best_non_gt_acc)
            mlp_preds, new_best_gt_acc, new_best_non_gt_acc, new_best_non_gt_f1 = self.pseudo_label_train(mask=mask,probs=weighted_probs,pseudo_labels=new_pred_labels,num_epochs=num_pseudo_label_epochs,writer=writer,gt_idx=gt_idx,prev_best_acc=best_gt_acc,meta_loop_idx=epoch_num,exp_dir=exp_dir)
            counts = {selected_acts[item]:sum(new_pred_labels==item) for item in set(new_pred_labels)}
            mlp_counts = {selected_acts[item]:sum(mlp_preds==item) for item in set(mlp_preds)}
            prev_weighted_probs = weighted_probs
            if ARGS.test or len(gt_idx) == 0 or new_best_gt_acc > best_gt_acc:
                peak_acc_at = epoch_num
                latent_acc = label_funcs.accuracy(new_pred_labels,self.dset.y)
                best_gt_acc = new_best_gt_acc
                best_non_gt_acc = new_best_non_gt_acc
            if not ARGS.suppress_prints:
                print('Counts:',counts)
                print('MLP Counts:',mlp_counts)
                print('Latent accuracy:', latent_acc)
            if ARGS.test: continue
        misc.check_dir(exp_dir)
        summary_file_path = os.path.join(exp_dir,'summary.txt')
        print(f'Best GT Acc: {best_gt_acc}')
        print(f'Best Non-gt Acc: {best_non_gt_acc}')
        print(f'Peak Acc: {peak_acc_at}')
        with open(summary_file_path,'w') as f:
            f.write(f'Non-gt Acc: {best_non_gt_acc}\n')
            f.write(f'GT Acc: {best_gt_acc}\n')
            f.write(f'Latent Acc: {latent_acc}\n')
            f.write(str(ARGS))
        # Save models
        if ARGS.save:
            misc.torch_save({'enc':self.enc,'dec':self.dec,'mlp':self.mlp},exp_dir,f'har_learner{ARGS.exp_name}.pt')
            misc.np_save(umapped_latents,exp_dir,f'umapped_latents{ARGS.exp_name}.npy')
            misc.np_save(new_pred_labels,exp_dir,f'preds{ARGS.exp_name}.npy')
            with open(os.path.join(exp_dir,f'HMM{ARGS.exp_name}.pkl'), 'wb') as f: pickle.dump(model,f)

    def train_meta_loop_simple(self,num_pre_epochs,num_epochs,num_pseudo_label_epochs,selected_acts,frac_gt_labels,exp_dir):
        writer = SummaryWriter()
        self.rec_train(num_pre_epochs)
        best_gt_acc = 0
        best_non_gt_acc = 0
        best_mlp_acc = 0
        peak_acc_at = 0
        dl = data.DataLoader(self.dset,batch_sampler=data.BatchSampler(data.RandomSampler(self.dset),self.batch_size,drop_last=False),pin_memory=False)
        if frac_gt_labels == 0:
            gt_idx = np.array([], dtype=np.int)
        elif frac_gt_labels <= 0.5:
            gt_idx = np.arange(len(self.dset), step=int(1/frac_gt_labels))
        else:
            non_gt_idx = np.arange(len(self.dset), step=int(1/(1-frac_gt_labels)))
            gt_idx = np.delete(np.arange(len(self.dset)),non_gt_idx)
        gt_mask = np.zeros(len(self.dset))
        gt_mask[gt_idx] = 1
        assert abs(len(gt_idx)/len(self.dset) - frac_gt_labels) < .01
        self.enc.train()
        self.mlp.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_rec_losses = []
            epoch_pseudo_label_losses = []
            epoch_losses = []
            total_pred_list = []
            total_gt_list = []
            total_idx_list = []
            best_loss = np.inf
            for batch_idx, (xb,yb,idx) in enumerate(dl):
               latent = self.enc(xb)
               latent = tensor_funcs.noiseify(latent,ARGS.noise)
               batch_mask = gt_mask[idx]
               label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
               label_loss = self.pseudo_label_lf(label_pred,yb.long())
               rec_pred = self.dec(latent)
               rec_loss = self.rec_lf(rec_pred,xb)
               loss = label_loss[batch_mask].mean() + rec_loss
               if math.isnan(loss): set_trace()
               loss.backward()
               self.enc_opt.step(); self.enc_opt.zero_grad()
               self.dec_opt.step(); self.dec_opt.zero_grad()
               self.mlp_opt.step(); self.mlp_opt.zero_grad()
               total_pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
               total_gt_list.append(yb.detach().cpu().numpy())
               total_idx_list.append(idx.detach().cpu().numpy())
               epoch_pseudo_label_losses.append(loss.item())
               epoch_losses.append(loss)
               if ARGS.test: break
            total_pred_array = np.concatenate(total_pred_list)
            total_idx_array = np.concatenate(total_idx_list)
            total_gt_array = self.dset.y.detach().cpu().numpy()[total_idx_array]
            total_pred_array_ordered = np.array([item[0] for item in sorted(zip(total_pred_array,total_idx_array),key=lambda x:x[1])])
            if ARGS.test: break
            non_gt_acc2 = label_funcs.accuracy(np.delete(total_pred_array,gt_idx),np.delete(total_gt_array,gt_idx))
            non_gt_acc = label_funcs.accuracy(np.delete(total_pred_array_ordered,gt_idx),np.delete(self.dset.y.detach().cpu().numpy(),gt_idx))
            non_gt_f1 = mean_f1(np.delete(total_pred_array_ordered,gt_idx),np.delete(self.dset.y.detach().cpu().numpy(),gt_idx))
            gt_acc = -1 if len(gt_idx) == 0 else label_funcs.accuracy(total_pred_array_ordered[gt_idx],self.dset.y.detach().cpu().numpy()[gt_idx])
            gt_mean_f1 = -1 if len(gt_idx) == 0 else mean_f1(total_pred_array_ordered[gt_idx],self.dset.y.detach().cpu().numpy()[gt_idx])
            full_acc = label_funcs.accuracy(total_pred_array,total_gt_array)
            full_acc2 = label_funcs.accuracy(total_pred_array_ordered,self.dset.y.detach().cpu().numpy())
            assert full_acc==full_acc2
            if not ARGS.suppress_prints:
               print(f'MLP non-gt acc: {non_gt_acc} {non_gt_acc2}')
               print(f'MLP gt acc: {gt_acc}')
               print(f'MLP non-gt mean_f1: {non_gt_f1}')
               print(f'MLP gt mean_f1: {gt_mean_f1}')
            if ARGS.test or len(gt_idx) == 0 or gt_acc > best_gt_acc:
               best_gt_acc=gt_acc
               best_non_gt_acc=non_gt_acc
               best_non_gt_f1=non_gt_f1
               misc.torch_save({'enc':self.enc,'dec':self.dec,'mlp':self.mlp},exp_dir,'best_model.pt')
            if epoch_loss < best_loss:
               best_loss = epoch_loss
               count = 0
            else:
               count += 1
            if count > 4: break
            mlp_counts = {selected_acts[item]:sum(total_pred_array_ordered==item) for item in set(total_pred_array_ordered)}
            if ARGS.test: continue
        misc.check_dir(exp_dir)
        summary_file_path = os.path.join(exp_dir,'summary.txt')
        print(f'Best GT Acc: {best_gt_acc}')
        print(f'Best Non-gt Acc: {best_non_gt_acc}')
        print(f'Peak Acc: {peak_acc_at}')
        with open(summary_file_path,'w') as f:
            f.write(f'Non-gt Acc: {best_non_gt_acc}\n')
            f.write(f'GT Acc: {best_gt_acc}\n')
            f.write(str(ARGS))
        # Save models
        if ARGS.save:
            misc.torch_save({'enc':self.enc,'dec':self.dec,'mlp':self.mlp},exp_dir,f'har_learner{ARGS.exp_name}.pt')
            misc.np_save(total_pred_array_ordered,exp_dir,f'preds{ARGS.exp_name}.npy')


def make_wisdm_v1_dset(args,subj_ids):
    activities_list = ['Jogging','Walking','Upstairs','Downstairs','Standing','Sitting']
    action_name_dict = dict(zip(range(len(activities_list)),activities_list))
    x = np.load('datasets/wisdm_v1/X.npy')
    y = np.load('datasets/wisdm_v1/y.npy')
    users = np.load('datasets/wisdm_v1/users.npy')
    idxs_to_user = np.zeros(users.shape[0]).astype(np.bool)
    for subj_id in subj_ids:
        new_users = users==subj_id
        idxs_to_user = np.logical_or(idxs_to_user,new_users)
    x = x[idxs_to_user]
    y = y[idxs_to_user]
    xnans = np.isnan(x).any(axis=1)
    x = x[~xnans]
    y = y[~xnans]
    num_windows = (len(x) - args.window_size)//args.step_size + 1
    mode_labels = np.concatenate([stats.mode(y[w*args.step_size:w*args.step_size + args.window_size]).mode for w in range(num_windows)])
    selected_ids = set(mode_labels)
    selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
    mode_labels, trans_dict, changed = label_funcs.compress_labels(mode_labels)
    assert len(selected_acts) == len(set(mode_labels))
    x = torch.tensor(x,device='cuda').float()
    y = torch.tensor(mode_labels,device='cuda').float()
    dset = StepDataset(x,y,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset, selected_acts

def make_wisdm_watch_dset(args,subj_ids):
    with open('datasets/wisdm-dataset/activity_key.txt') as f: r=f.readlines()
    activities_list = [x.split(' = ')[0] for x in r if ' = ' in x]
    action_name_dict = dict(zip(range(len(activities_list)),activities_list))
    x = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}.npy') for s in subj_ids])
    y = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_labels.npy') for s in subj_ids])
    certains = np.concatenate([np.load(f'datasets/wisdm-dataset/np_data/{s}_certains.npy') for s in subj_ids])
    x = x[certains]
    y = y[certains]
    xnans = np.isnan(x).any(axis=1)
    x = x[~xnans]
    y = y[~xnans]
    num_windows = (len(x) - args.window_size)//args.step_size + 1
    mode_labels = np.concatenate([stats.mode(y[w*args.step_size:w*args.step_size + args.window_size]).mode for w in range(num_windows)])
    selected_ids = set(mode_labels)
    selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
    mode_labels, trans_dict, changed = label_funcs.compress_labels(mode_labels)
    assert len(selected_acts) == len(set(mode_labels))
    x = torch.tensor(x,device='cuda').float()
    y = torch.tensor(mode_labels,device='cuda').float()
    dset = StepDataset(x,y,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset, selected_acts

def make_uci_dset(args,subj_ids):
    action_name_dict = {1:'walking',2:'walking upstairs',3:'walking downstairs',4:'sitting',5:'standing',6:'lying',7:'stand_to_sit',9:'sit_to_stand',10:'sit_to_lit',11:'lie_to_sit',12:'stand_to_lie',13:'lie_to_stand'}
    if args.dset == 'UCI-pre':
        selected_acts = list(action_name_dict.values())
        x = np.load('UCI2/X_train.npy')
        y = np.load('UCI2/y_train.npy')
        x = x[y!=-1]
        y = y[y!=-1]
        x = torch.tensor(x,device='cuda').float()
        y = torch.tensor(y,device='cuda').float() - 1 #To begin at 0 rather than 1
        dset = Preprocced_Dataset(x,y,device='cuda')
    elif args.dset == 'UCI-raw':
        x = np.concatenate([np.load(f'datasets/UCI2/np_data/user{subj_id}.npy') for subj_id in subj_ids])
        y = np.concatenate([np.load(f'datasets/UCI2/np_data/user{subj_id}_labels.npy') for subj_id in subj_ids])
        xnans = np.isnan(x).any(axis=1)
        x = x[~xnans]
        y = y[~xnans]
        x = x[y<7] # Labels still begin at 1 at this point as
        y = y[y<7] # haven't been compressed, so select 1,..,6
        x = x[y!=-1]
        y = y[y!=-1]
        num_windows = (len(x) - args.window_size)//args.step_size + 1
        mode_labels = np.concatenate([stats.mode(y[w*args.step_size:w*args.step_size + args.window_size]).mode for w in range(num_windows)])
        selected_ids = set(mode_labels)
        selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
        mode_labels, trans_dict, changed = label_funcs.compress_labels(mode_labels)
        assert len(selected_acts) == len(set(mode_labels))
        x = torch.tensor(x,device='cuda').float()
        y = torch.tensor(mode_labels,device='cuda').float()
        dset = StepDataset(x,y,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset, selected_acts

def make_pamap_dset(args,subj_ids):
    action_name_dict = {1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'Nordic walking',9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing soccer',24:'rope jumping'}
    x = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{subj_id}.npy') for subj_id in subj_ids])
    y = np.concatenate([np.load(f'datasets/PAMAP2_Dataset/np_data/subject{subj_id}_labels.npy') for subj_id in subj_ids])
    x = x[y!=0]
    y = y[y!=0]
    xnans = np.isnan(x).any(axis=1)
    x = x[~xnans]
    y = y[~xnans]
    num_windows = (len(x) - args.window_size)//args.step_size + 1
    mode_labels = np.concatenate([stats.mode(y[w*args.step_size:w*args.step_size + args.window_size]).mode for w in range(num_windows)])
    selected_ids = set(mode_labels)
    selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
    mode_labels, trans_dict, changed = label_funcs.compress_labels(mode_labels)
    assert len(selected_acts) == len(set(mode_labels))
    x = torch.tensor(x,device='cuda').float()
    y = torch.tensor(mode_labels,device='cuda').float()
    dset = StepDataset(x,y,device='cuda',window_size=args.window_size,step_size=args.step_size)
    return dset, selected_acts

def train(args,subj_ids):
    prep_start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dset=='PAMAP': dset, selected_acts = make_pamap_dset(args,subj_ids)
    elif args.dset in ['UCI-pre','UCI-raw']: dset, selected_acts = make_uci_dset(args,subj_ids)
    elif args.dset == 'WISDM-watch': dset, selected_acts = make_wisdm_watch_dset(args,subj_ids)
    elif args.dset == 'WISDM-v1': dset, selected_acts = make_wisdm_v1_dset(args,subj_ids)
    num_labels = label_funcs.get_num_labels(dset.y)
    mlp = Var_BS_MLP(32,25,num_labels)
    if args.dset == 'PAMAP':
        x_filters = (50,40,5,3)
        y_filters = (5,3,2,1)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = (2,3,2,2)
        enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,verbose=args.verbose)
        dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,(30,8),(1,1)),#24
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(16,8,(30,8),(3,1)),#9
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8,4,(20,6),(2,2)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4,1,(10,6),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(1),
            )
    elif args.dset == 'UCI-pre':
        enc = Dense_Net(561,1024,256,32)
        dec = Dense_Net(32,256,1024,561)
    elif args.dset == 'UCI-raw':
        x_filters = (50,40,5,4)
        y_filters = (1,1,3,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,verbose=args.verbose)
        dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,(30,2),(1,1)),#24
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(16,8,(30,3),(3,3)),#9
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8,4,(20,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4,1,(10,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(1),
            )
    elif args.dset == 'WISDM-watch':
        x_filters = (50,40,5,4)
        y_filters = (1,1,3,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,verbose=args.verbose)
        dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,(30,2),(1,1)),#24
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(16,8,(30,3),(3,3)),#9
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8,4,(20,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4,1,(10,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(1),
            )
    elif args.dset == 'WISDM-v1':
        x_filters = (50,40,5,4)
        y_filters = (1,1,2,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,verbose=args.verbose)
        dec = nn.Sequential(
            nn.ConvTranspose2d(32,16,(30,2),(1,1)),#24
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(16,8,(30,2),(3,1)),#9
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8,4,(20,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4,1,(10,1),(2,1)),#2
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(1),
            )
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        dec.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    dec.cuda()
    mlp.cuda()

    har = HARLearner(enc=enc,dec=dec,mlp=mlp,dset=dset,device='cuda',batch_size=args.batch_size,num_classes=num_labels)
    exp_dir = os.path.join(f'experiments/{args.exp_name}')

    train_start_time = time.time()
    if args.simple:
        har.train_meta_loop_simple(num_pre_epochs=args.num_pre_epochs, num_epochs=args.num_meta_epochs, num_pseudo_label_epochs=args.num_pseudo_label_epochs, selected_acts=selected_acts, frac_gt_labels=args.frac_gt_labels, exp_dir=exp_dir)
    else:
        har.train_meta_loop(num_pre_epochs=args.num_pre_epochs, num_meta_epochs=args.num_meta_epochs, num_pseudo_label_epochs=args.num_pseudo_label_epochs, prob_thresh=args.prob_thresh, frac_gt_labels=args.frac_gt_labels, selected_acts=selected_acts, exp_dir=exp_dir)
    train_end_time = time.time()
    total_prep_time = misc.asMinutes(train_start_time-prep_start_time)
    total_train_time = misc.asMinutes(train_end_time-train_start_time)
    print(f"Prep time: {total_prep_time}\n Train time: {total_train_time}")


def f1(bin_classifs_pred,bin_classifs_gt):
    tp = sum(bin_classifs_pred*bin_classifs_gt)
    if tp==0: return 0
    fp = sum(bin_classifs_pred*~bin_classifs_gt)
    fn = sum(~bin_classifs_pred*bin_classifs_gt)

    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    return (2*prec*rec)/(prec+rec)


def mean_f1(labels1,labels2):
    subsample_size = min(len(labels1),30000)
    trans_labels = label_funcs.translate_labellings(labels1,labels2,subsample_size)
    #assert label_funcs.unique_labels(trans_labels) == label_funcs.unique_labels(labels2)
    lab_f1s = []
    for lab in label_funcs.unique_labels(trans_labels):
        lab_booleans1 = trans_labels==lab
        lab_booleans2 = labels2==lab
        lab_f1s.append(f1(lab_booleans1,lab_booleans2))
    return sum(lab_f1s)/len(lab_f1s)

if __name__ == "__main__":

    dset_options = ['PAMAP','UCI-pre','UCI-raw','WISDM-v1','WISDM-watch']
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--alpha',type=float,default=.5)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--dset',type=str,default='PAMAP',choices=dset_options)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--exp_name',type=str,default="jim")
    parser.add_argument('--frac_gt_labels',type=float,default=0.1)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--noise',type=float,default=1.)
    parser.add_argument('--num_meta_epochs',type=int,default=30)
    parser.add_argument('--num_pre_epochs',type=int,default=5)
    parser.add_argument('--num_pseudo_label_epochs',type=int,default=5)
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--probs_abl1',action='store_true')
    parser.add_argument('--probs_abl2',action='store_true')
    parser.add_argument('--prob_thresh',type=float,default=.95)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--simple',action='store_true')
    parser.add_argument('--subj_ids',type=str,nargs='+',default=['first'])
    parser.add_argument('--suppress_prints',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--umap_abl',action='store_true')
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
    if ARGS.subj_ids == ['first']: subj_ids = all_possible_ids[:1]
    elif ARGS.all_subjs: subj_ids=all_possible_ids
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
