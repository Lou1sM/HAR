import sys
from scipy.stats import multivariate_normal
from hmmlearn import hmm
from copy import deepcopy
from scipy import stats
import os
import argparse
import math
from pdb import set_trace
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils import data
from dl_utils import misc, label_funcs
from make_dsets import make_dset_train_val, make_dsets_by_user


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

class DecByLayer(nn.Module):
    def __init__(self,x_filters,y_filters,x_strides,y_strides,verbose):
        super(DecByLayer,self).__init__()
        self.verbose = verbose
        num_layers = len(x_filters)
        assert all(len(x)==num_layers for x in (y_filters,x_strides,y_strides))
        ncvs = [4*2**i for i in reversed(range(num_layers))]+[1]
        conv_trans_layers = [nn.Sequential(
                nn.ConvTranspose2d(ncvs[i],ncvs[i+1],(x_filters[i],y_filters[i]),(x_strides[i],y_strides[i])),
                nn.BatchNorm2d(ncvs[i+1]),
                nn.LeakyReLU(0.3),
                )
            for i in range(num_layers)]
        self.conv_trans_layers = nn.ModuleList(conv_trans_layers)

    def forward(self,x):
        if self.verbose: print(x.shape)
        for conv_trans_layer in self.conv_trans_layers:
            x = conv_trans_layer(x)
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
    def __init__(self,enc,mlp,dec,batch_size,num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.enc = enc
        self.dec = dec
        self.mlp = mlp
        self.pseudo_label_lf = nn.CrossEntropyLoss(reduction='none')
        self.rec_lf = nn.MSELoss()

        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=ARGS.enc_lr)
        self.dec_opt = torch.optim.Adam(self.dec.parameters(),lr=ARGS.dec_lr)
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(),lr=ARGS.mlp_lr)

    def get_latents(self,dset):
        self.enc.eval()
        collected_latents = []
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),self.batch_size,drop_last=False),pin_memory=False)
        for idx, (xb,yb,tb) in enumerate(determin_dl):
            batch_latents = self.enc(xb)
            batch_latents = batch_latents.view(batch_latents.shape[0],-1).detach().cpu().numpy()
            collected_latents.append(batch_latents)
        collected_latents = np.concatenate(collected_latents,axis=0)
        return collected_latents

    def reinit_nets(self):
        for m in self.enc.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
        for m in self.dec.modules():
            if isinstance(m,nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
        for m in self.mlp.modules():
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)

    def train(self,dset_train,dset_val,num_epochs,frac_gt_labels,selected_acts,exp_dir):
        best_gt_acc = 0
        best_non_gt_acc = 0
        best_non_gt_f1 = 0
        dl_train = data.DataLoader(dset_train,batch_sampler=data.BatchSampler(data.RandomSampler(dset_train),self.batch_size,drop_last=False),pin_memory=False)
        dl_val = data.DataLoader(dset_val,batch_sampler=data.BatchSampler(data.RandomSampler(dset_val),self.batch_size,drop_last=False),pin_memory=False)
        if frac_gt_labels == 0:
            gt_idx = np.array([], dtype=np.int)
        elif frac_gt_labels <= 0.5:
            gt_idx = np.arange(len(dset_train), step=int(1/frac_gt_labels))
        else:
            non_gt_idx = np.arange(len(dset_train), step=int(1/(1-frac_gt_labels)))
            gt_idx = np.delete(np.arange(len(dset_train)),non_gt_idx)
        gt_mask = torch.zeros_like(dset_train.y)
        gt_mask[gt_idx] = 1
        assert abs(len(gt_idx)/len(dset_train) - frac_gt_labels) < .01
        for epoch in range(num_epochs):
            epoch_loss = 0
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
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                train_pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
                train_idx_list.append(idx.detach().cpu().numpy())
                if ARGS.test: break
            train_pred_array = np.concatenate(train_pred_list)
            train_idx_array = np.concatenate(train_idx_list)
            train_pred_array_ordered = np.array([item[0] for item in sorted(zip(train_pred_array,train_idx_array),key=lambda x:x[1])])
            train_acc = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.accuracy(train_pred_array_ordered,dset_train.y.detach().cpu().numpy())
            train_f1 = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.mean_f1(train_pred_array_ordered,dset_train.y.detach().cpu().numpy())
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
            val_acc = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.accuracy(val_pred_array_ordered,dset_val.y.detach().cpu().numpy())
            val_f1 = -1 if ARGS.test or len(gt_idx) == 0 else label_funcs.mean_f1(val_pred_array_ordered,dset_val.y.detach().cpu().numpy())
            if not ARGS.suppress_prints:
                print(f'MLP val acc: {val_acc}')
                print(f'MLP val mean_f1: {val_f1}')
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

    def cross_train(self,user_dsets,num_epochs,frac_gt_labels,exp_dir):
        results_matrix = np.zeros((len(user_dsets),len(user_dsets)))
        pred_predee = [[] for _ in range(len(user_dsets))]
        self_diag_preds = []
        for dset_idx,(dset,sa) in enumerate(user_dsets.values()):
            all_user_preds = {}
            n = label_funcs.get_num_labels(dset.y)
            if n != self.num_classes and ARGS.fussy_label_numbers:
                print(f"Not training on {dset_idx} because only {n} labels, should be {self.num_classes}")
                continue
            acc, f1, pred_array = self.train_with_fract_gts_on(dset,num_epochs,frac_gt_labels,reinit=True)
            self_diag_preds.append(pred_array)
            results_matrix[dset_idx,dset_idx] = round(acc,4)
            for dset_idx_val,(dset_val,sa) in enumerate(user_dsets.values()):
                if dset_idx_val == dset_idx: continue
                acc_val, f1_val, pred_array_ordered_val = self.val_on(dset_val)
                pred_predee[dset_idx_val].append(pred_array_ordered_val)
                results_matrix[dset_idx,dset_idx_val] = round(acc_val,4)
        pseudo_label_dsets = []
        for dset_idx,(dset,sa) in enumerate(user_dsets.values()):
            preds = pred_predee[dset_idx]
            gt_labels = dset.y.detach().cpu().numpy()
            pred_list_translated =  preds if ARGS.test else label_funcs.debable(preds,pivot=gt_labels)
            pred_array = np.stack(pred_list_translated)
            mode_ensemble_preds = stats.mode(pred_array,axis=0).mode[0]
            full_ensemble_acc = -1 if ARGS.test else label_funcs.accuracy(mode_ensemble_preds,gt_labels)
            all_agrees = np.ones(len(dset)).astype(np.bool) if ARGS.test else np.all(pred_array==pred_array[0,:], axis=0)
            agreed_preds = mode_ensemble_preds[all_agrees]
            agreed_gt_labels = gt_labels[all_agrees]
            all_agree_ensemble_acc = -1 if ARGS.test or not all_agrees.any() else label_funcs.accuracy(agreed_preds,agreed_gt_labels)
            print(f"Ensemble acc on {dset_idx}: {full_ensemble_acc}, {all_agree_ensemble_acc} {all_agrees.sum()/len(all_agrees)}")
            pseudo_label_dset = deepcopy(dset)
            pseudo_label_dsets.append((pseudo_label_dset,all_agrees))
            # Convert pred_array to multihot form
            multihot_pred_array = np.stack([(pred_array==lab).sum(axis=0) for lab in range(self.num_classes)])
            temperature = .5
            ensemble_prob_tensor = cudify(multihot_pred_array*temperature).float().softmax(dim=0)
            all_agrees_t = cudify(all_agrees)
            prob_mask = npify(ensemble_prob_tensor.max(axis=0)[0])
            acc, pseudo_label_pred_array = self.pseudo_label_meta_loop(pseudo_label_dset,mode_ensemble_preds,num_meta_epochs=ARGS.num_pseudo_label_epochs,ensemble_mask=prob_mask)
            val_acc, val_f1, val_pred_array = self.val_on(dset)
            print("ensemble_preds label_counts on agreed points:", label_funcs.label_counts(agreed_preds[agreed_preds==agreed_gt_labels]))
            print("ensemble_preds label_counts on all points:", label_funcs.label_counts(mode_ensemble_preds))
            print(f"Acc from pseudo_label training, train: {acc}\tval: {val_acc}")
            if all_agree_ensemble_acc >= .95:
                other_acc, other_f1, other_pred_array = self.train_with_fract_gts_on(dset,num_epochs,all_agrees.mean())
                print(f"Acc from regular frac_gt training with same frac of gts: {other_acc}")
            print()

        print(results_matrix)
        return results_matrix

    def pseudo_label_meta_loop(self,pseudo_label_dset,pseudo_labels_array,num_meta_epochs,ensemble_mask):
        prev_probs = np.zeros((len(pseudo_label_dset),self.num_classes))
        for epoch_num in range(num_meta_epochs):
            if ARGS.test:
                new_pred_labels = label_funcs.dummy_labels(self.num_classes,len(pseudo_label_dset))
                new_pred_labels = new_pred_labels.astype(np.long)
                combined_mask = torch.ones(len(pseudo_label_dset.y)).bool()
                weighted_probs = torch.ones(len(pseudo_label_dset.y)).bool()
            else:
                latents = self.get_latents(pseudo_label_dset)
                print('umapping')
                umapped_latents = latents if ARGS.no_umap else umap.UMAP(min_dist=0,n_neighbors=60,n_components=2,random_state=42).fit_transform(latents.squeeze())
                print('modelling')
                model = hmm.GaussianHMM(self.num_classes,'full')
                model.params = 'mc'
                model.init_params = 'mc'
                model.startprob_ = np.ones(self.num_classes)/self.num_classes
                num_action_blocks = len([item for idx,item in enumerate(pseudo_label_dset.y) if pseudo_label_dset.y[idx-1] != item])
                prob_new_action = num_action_blocks/len(pseudo_label_dset)
                model.transmat_ = (np.eye(self.num_classes) * (1-prob_new_action)) + (np.ones((self.num_classes,self.num_classes))*prob_new_action/self.num_classes)
                model.fit(umapped_latents)
                new_pred_labels = model.predict(umapped_latents)
                new_pred_probs = model.predict_proba(umapped_latents)
                subsample_size = min(30000,pseudo_label_dset.y.shape[0])
                trans_dict, leftovers = label_funcs.get_trans_dict(new_pred_labels,pseudo_label_dset.y,subsample_size=subsample_size)
                new_pred_labels = np.array([trans_dict[l] for l in new_pred_labels])
                new_pred_labels = new_pred_labels.astype(np.int)
                mvns = [multivariate_normal(m,c) for m,c in zip(model.means_,model.covars_)]
                probs=np.array([mvns[label].pdf(mean) for mean,label in zip(umapped_latents,new_pred_labels)])
                probs *= new_pred_probs.max(axis=1) #Include probs from HMM itself too, as well as normals
                probs = np.clip(probs,0,1)
                combined_mask = ensemble_mask - probs
                increase_conf_idxs = new_pred_labels==pseudo_labels_array
                combined_mask[increase_conf_idxs] += 2*probs[increase_conf_idxs]
                replace_idxs = probs>ensemble_mask
                pseudo_labels_array[replace_idxs] = new_pred_labels[replace_idxs]
                combined_mask[replace_idxs] = abs(combined_mask[replace_idxs])
                pseudo_label_dset.y = cudify(pseudo_labels_array)
                if not ARGS.suppress_prints: print('probs mean', probs.mean())
            acc, f1, pred_array = self.train_on(pseudo_label_dset,num_epochs=5,multiplicative_mask=cudify(combined_mask),rlmbda=ARGS.rlmbda)
            prev_probs = probs
            if ARGS.test: continue
        return acc, pred_array

    def train_on(self,dset,num_epochs,multiplicative_mask='none',lf=None,compute_acc=True,reinit=True,rlmbda=0):
        if reinit: self.reinit_nets()
        self.enc.train()
        self.dec.train()
        self.mlp.train()
        if lf is None: lf = self.pseudo_label_lf
        best_acc = 0
        best_f1 = 0
        best_pred_array_ordered = -np.ones(len(dset.y))
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),self.batch_size,drop_last=False),pin_memory=False)
        for epoch in range(num_epochs):
            pred_list = []
            idx_list = []
            best_f1 = 0
            for batch_idx, (xb,yb,idx) in enumerate(dl):
                latent = self.enc(xb)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                label_loss = lf(label_pred,yb.long())
                if multiplicative_mask is not 'none':
                    batch_mask = multiplicative_mask[:self.batch_size] if ARGS.test else multiplicative_mask[idx]
                    loss = rlmbda + (label_loss*batch_mask).mean()
                else: loss = rlmbda + label_loss.mean()
                if math.isnan(loss): set_trace()
                if rlmbda>0:
                    rec_loss = self.rec_lf(self.dec(latent),xb)
                    loss += rec_loss
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
                idx_list.append(idx.detach().cpu().numpy())
                if ARGS.test: break
            if ARGS.test:
                return -1, -1, label_funcs.dummy_labels(self.num_classes,len(dset.y))
            pred_array = np.concatenate(pred_list)
            idx_array = np.concatenate(idx_list)
            pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
            if compute_acc:
                acc = -1 if ARGS.test else label_funcs.accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
                f1 = -1 if ARGS.test else label_funcs.mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
                if ARGS.test or acc > best_acc:
                    best_pred_array_ordered = pred_array_ordered
                    best_acc = acc
                    best_f1 = f1
        return best_acc,best_f1,best_pred_array_ordered

    def train_with_fract_gts_on(self,dset,num_epochs,frac_gt_labels,reinit=False,rlmbda=0):
        if frac_gt_labels == 0:
            gt_idx = np.array([], dtype=np.int)
        elif frac_gt_labels <= 0.5:
            gt_idx = np.arange(len(dset), step=int(1/frac_gt_labels))
        else:
            non_gt_idx = np.arange(len(dset), step=int(1/(1-frac_gt_labels)))
            gt_idx = np.delete(np.arange(len(dset)),non_gt_idx)
        gt_mask = torch.zeros_like(dset.y)
        gt_mask[gt_idx] = 1
        approx = len(gt_idx)/len(dset)
        if abs(approx - frac_gt_labels) > .01:
            print(f"frac_gts approximation is {approx}, instead of {frac_gt_labels}")
        return self.train_on(dset,num_epochs,gt_mask,reinit=reinit,rlmbda=rlmbda)

    def val_on(self,dset):
        self.enc.eval()
        self.dec.eval()
        self.mlp.eval()
        pred_list = []
        idx_list = []
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),self.batch_size,drop_last=False),pin_memory=False)
        for batch_idx, (xb,yb,idx) in enumerate(dl):
            latent = self.enc(xb)
            label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
            pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
            idx_list.append(idx.detach().cpu().numpy())
            if ARGS.test: break
        pred_array = np.concatenate(pred_list)
        if ARGS.test:
            return -1, -1, label_funcs.dummy_labels(self.num_classes,len(dset.y))
        idx_array = np.concatenate(idx_list)
        pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
        if label_funcs.get_num_labels(dset.y) == 1: set_trace()
        acc = -1 if ARGS.test else label_funcs.accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
        f1 = -1 if ARGS.test else label_funcs.mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
        return acc,f1,pred_array_ordered

def true_cross_entropy_with_logits(pred,target):
    return (-(pred*target).sum(dim=1) + torch.logsumexp(pred,dim=1)).mean()

def cudify(x): return torch.tensor(x,device='cuda')
def npify(t): return t.detach().cpu().numpy()

def main(args,subj_ids):
    prep_start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dset == 'PAMAP':
        x_filters = (50,40,3,3)
        y_filters = (5,3,2,1)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = (2,2,2,2)
        x_filters_trans = (6,20,20,40)
        y_filters_trans = (8,8,6,6)
        x_strides_trans = (1,2,3,4)
        y_strides_trans = (1,1,2,1)
        num_labels = 12
    elif args.dset == 'UCI':
        x_filters = (50,40,4,4)
        y_filters = (1,1,3,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        y_filters_trans = (2,3,1,1)
        x_strides_trans = (1,3,2,2)
        y_strides_trans = (1,3,1,2)
        num_labels = 6
    elif args.dset == 'WISDM-v1':
        x_filters = (50,40,4,4)
        y_filters = (1,1,2,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        y_filters_trans = (2,2,1,1)
        x_strides_trans = (1,3,2,2)
        y_strides_trans = (1,1,1,1)
        num_labels = 5
    elif args.dset == 'WISDM-watch':
        x_filters = (50,40,4,4)
        y_filters = (1,1,3,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        y_filters_trans = (2,3,1,1)
        x_strides_trans = (1,3,2,2)
        y_strides_trans = (1,3,1,2)
        num_labels = 17
    enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,verbose=args.verbose)
    dec = DecByLayer(x_filters_trans,y_filters_trans,x_strides_trans,y_strides_trans,verbose=args.verbose)
    mlp = Var_BS_MLP(32,25,num_labels)
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        mlp.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    dec.cuda()
    mlp.cuda()

    har = HARLearner(enc=enc,mlp=mlp,dec=dec,batch_size=args.batch_size,num_classes=num_labels)
    exp_dir = os.path.join(f'experiments/{args.exp_name}')

    train_start_time = time.time()
    if ARGS.cross_train:
        dsets_by_id = make_dsets_by_user(args,subj_ids)
        bad_ids = []
        for user_id, (dset,sa) in dsets_by_id.items():
            n = label_funcs.get_num_labels(dset.y)
            if n < num_labels/2:
                print(f"Excluding user {user_id}, only has {n} different labels, instead of {num_labels}")
                bad_ids.append(user_id)
        dsets_by_id = {k:v for k,v in dsets_by_id.items() if k not in bad_ids}
        har.cross_train(dsets_by_id,args.num_epochs,args.frac_gt_labels,exp_dir=exp_dir)
    else:
        dset_train, dset_val, selected_acts = make_dset_train_val(args,subj_ids)
        har.train(dset_train,dset_val,args.num_epochs,args.frac_gt_labels,exp_dir=exp_dir,selected_acts=selected_acts)
    train_end_time = time.time()
    total_prep_time = misc.asMinutes(train_start_time-prep_start_time)
    total_train_time = misc.asMinutes(train_end_time-train_start_time)
    print(f"Prep time: {total_prep_time}\tTrain time: {total_train_time}")


if __name__ == "__main__":

    dset_options = ['PAMAP','UCI','WISDM-v1','WISDM-watch']
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--num_subjs',type=int)
    group.add_argument('--subj_ids',type=str,nargs='+',default=['first'])
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--alpha',type=float,default=.5)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--cross_train',action='store_true')
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--dset',type=str,default='PAMAP',choices=dset_options)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--exp_name',type=str,default="jim")
    parser.add_argument('--frac_gt_labels',type=float,default=0.1)
    parser.add_argument('--fussy_label_numbers',action='store_true')
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--num_pseudo_label_epochs',type=int,default=3)
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--rlmbda',type=float,default=.1)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--suppress_prints',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--no_umap',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.test and ARGS.save:
        print("Shouldn't be saving for a test run"); sys.exit()
    if ARGS.test: ARGS.num_meta_epochs = 2
    elif not ARGS.no_umap: import umap
    if ARGS.dset == 'PAMAP':
        all_possible_ids = [str(x) for x in range(101,110)]
    elif ARGS.dset == 'UCI':
        def two_digitify(x): return '0' + str(x) if len(str(x))==1 else str(x)
        all_possible_ids = [two_digitify(x) for x in range(1,30)]
    elif ARGS.dset == 'WISDM-watch':
        all_possible_ids = [str(x) for x in range(1600,1651)]
    elif ARGS.dset == 'WISDM-v1':
        all_possible_ids = [str(x) for x in range(1,37)] #Paper says 29 users but ids go up to 36
    if ARGS.all_subjs: subj_ids=all_possible_ids
    elif ARGS.num_subjs is not None: subj_ids = all_possible_ids[:ARGS.num_subjs]
    elif ARGS.subj_ids == ['first']: subj_ids = all_possible_ids[:1]
    else: subj_ids = ARGS.subj_ids
    bad_ids = [x for x in subj_ids if x not in all_possible_ids]
    if len(bad_ids) > 0:
        print(f"You have specified non-existent ids: {bad_ids}"); sys.exit()
    if ARGS.parallel:
        main(ARGS,subj_ids=subj_ids)
    else:
        orig_name = ARGS.exp_name
        for subj_id in subj_ids:
            print(f"Training and predicting on id {subj_id}")
            ARGS.exp_name = f"{orig_name}{subj_id}"
            main(ARGS,subj_ids=[subj_id])
