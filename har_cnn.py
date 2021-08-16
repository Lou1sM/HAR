import sys
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
from dl_utils.misc import asMinutes, np_save
from dl_utils.label_funcs import accuracy, mean_f1, debable, translate_labellings, compress_labels, get_num_labels, label_counts, dummy_labels, unique_labels, get_trans_dict
from dl_utils.tensor_funcs import noiseify, numpyify
from make_dsets import make_dset_train_val, make_dsets_by_user, ChunkDataset


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
        self.rec_lf = nn.MSELoss(reduction='none')

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

    def train_on(self,dset,num_epochs,multiplicative_mask='none',lf=None,compute_acc=True,reinit=True,rlmbda=0,custom_sampler='none'):
        if reinit: self.reinit_nets()
        self.enc.train()
        self.dec.train()
        self.mlp.train()
        if lf is None: lf = self.pseudo_label_lf
        best_acc = 0
        best_f1 = 0
        best_pred_array_ordered = -np.ones(len(dset.y))
        sampler = data.RandomSampler(dset) if custom_sampler is 'none' else custom_sampler
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(sampler,self.batch_size,drop_last=False),pin_memory=False)
        for epoch in range(num_epochs):
            pred_list = []
            idx_list = []
            conf_list = []
            best_f1 = 0
            for batch_idx, (xb,yb,idx) in enumerate(dl):
                latent = self.enc(xb)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                label_loss = lf(label_pred,yb.long())
                if multiplicative_mask is not 'none':
                    batch_mask = multiplicative_mask[:self.batch_size] if ARGS.test else multiplicative_mask[idx]
                    loss = (label_loss*batch_mask).mean()
                else: loss = label_loss.mean()
                if math.isnan(loss): set_trace()
                if rlmbda>0:
                    rec_loss = self.rec_lf(self.dec(latent),xb).mean()
                    loss += rec_loss
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                if rlmbda>0: self.dec_opt.step(); self.dec_opt.zero_grad()
                conf,pred = label_pred.max(axis=1)
                pred_list.append(numpyify(pred))
                conf_list.append(numpyify(conf))
                idx_list.append(idx.detach().cpu().numpy())
                if ARGS.test: break
            if ARGS.test:
                return -1, -1, dummy_labels(self.num_classes,len(dset.y)), np.ones(len(dset))
            pred_array = np.concatenate(pred_list)
            idx_array = np.concatenate(idx_list)
            conf_array = np.concatenate(conf_list)
            pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
            conf_array_ordered = np.array([item[0] for item in sorted(zip(conf_array,idx_array),key=lambda x:x[1])])
            if compute_acc:
                acc = -1 if ARGS.test else accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
                f1 = -1 if ARGS.test else mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
                print(acc)
                m = numpyify(multiplicative_mask.bool())
                print(accuracy(pred_array_ordered[m],dset.y.detach().cpu().numpy()[m]))
                if ARGS.test or acc > best_acc:
                    best_pred_array_ordered = pred_array_ordered
                    best_conf_array_ordered = conf_array_ordered
                    best_acc = acc
                    best_f1 = f1
        return best_acc,best_f1,best_pred_array_ordered,best_conf_array_ordered

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
            return -1, -1, dummy_labels(self.num_classes,len(dset.y))
        idx_array = np.concatenate(idx_list)
        pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
        if get_num_labels(dset.y) == 1: set_trace()
        acc = -1 if ARGS.test else accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
        f1 = -1 if ARGS.test else mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
        return acc,f1,pred_array_ordered

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

    def cross_train(self,user_dsets,num_epochs,frac_gt_labels,exp_dir):
        results_matrix = np.zeros((len(user_dsets),len(user_dsets)))
        pred_predee = [[] for _ in range(len(user_dsets))]
        self_diag_preds = []
        for dset_idx,(dset,sa) in enumerate(user_dsets.values()):
            all_user_preds = {}
            n = get_num_labels(dset.y)
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
            pred_list_translated =  preds if ARGS.test else debable(preds,pivot=gt_labels)
            pred_array = np.stack(pred_list_translated)
            mode_ensemble_preds = stats.mode(pred_array,axis=0).mode[0]
            full_ensemble_acc = -1 if ARGS.test else accuracy(mode_ensemble_preds,gt_labels)
            all_agrees = np.ones(len(dset)).astype(np.bool) if ARGS.test else np.all(pred_array==pred_array[0,:], axis=0)
            agreed_preds = mode_ensemble_preds[all_agrees]
            agreed_gt_labels = gt_labels[all_agrees]
            all_agree_ensemble_acc = -1 if ARGS.test or not all_agrees.any() else accuracy(agreed_preds,agreed_gt_labels)
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
            print("ensemble_preds label_counts on agreed points:", label_counts(agreed_preds[agreed_preds==agreed_gt_labels]))
            print("ensemble_preds label_counts on all points:", label_counts(mode_ensemble_preds))
            print(f"Acc from pseudo_label training, train: {acc}\tval: {val_acc}")
            if all_agree_ensemble_acc >= .95:
                other_acc, other_f1, other_pred_array = self.train_with_fract_gts_on(dset,num_epochs,all_agrees.mean())
                print(f"Acc from regular frac_gt training with same frac of gts: {other_acc}")
            print()

        print(results_matrix)
        return results_matrix

    def pseudo_label_train(self,pseudo_label_dset,mask,num_epochs,position_in_meta_loop=0):
        self.enc.train()
        pseudo_label_dl = data.DataLoader(pseudo_label_dset,batch_sampler=data.BatchSampler(data.RandomSampler(pseudo_label_dset),self.batch_size,drop_last=False),pin_memory=False)
        rlmbda = 1 - (position_in_meta_loop/3)
        noise = ARGS.noise*(1 - (position_in_meta_loop/3))
        for epoch_num in range(num_epochs):
            pred_list = []
            idx_list = []
            for batch_idx, (xb,yb,idx) in enumerate(pseudo_label_dl):
                batch_mask = mask[idx]
                latent = self.enc(xb)
                latent = noiseify(latent,noise)
                if batch_mask.any():
                    try:
                        pseudo_label_pred = self.mlp(latent[:,:,0,0])
                    except ValueError: set_trace()
                    loss = (self.pseudo_label_lf(pseudo_label_pred,yb)*batch_mask).mean()
                else:
                    loss = torch.tensor(0,device=xb.device)
                #if not batch_mask.all() and rlmbda > 0:
                latents_to_rec_train = latent
                rec_pred = self.dec(latents_to_rec_train)
                rec_loss = (self.rec_lf(rec_pred,xb).mean((1,2,3))*(1-batch_mask)).mean()
                loss += rlmbda*rec_loss
                if math.isnan(loss): set_trace()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                pred_list.append(pseudo_label_pred.argmax(axis=1).detach().cpu().numpy())
                idx_list.append(idx.detach().cpu().numpy())
            if ARGS.test:
                pred_array_ordered = dummy_labels(self.num_classes,len(pseudo_label_dset))
                break
            pred_array = np.concatenate(pred_list)
            idx_array = np.concatenate(idx_list)
            pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
        return pred_array_ordered

    def pseudo_label_cluster_meta_loop(self,dset,meta_pivot_pred_labels,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
        old_pred_labels = -np.ones(dset.y.shape)
        np_gt_labels = dset.y.detach().cpu().numpy().astype(int)
        super_mask = np.ones(len(dset)).astype(np.bool)
        for epoch_num in range(num_meta_epochs):
            print('Meta Epoch:', epoch_num)
            if ARGS.test:
                num_tiles = len(dset.y)//self.num_classes
                new_pred_labels = np.tile(np.arange(self.num_classes),num_tiles).astype(np.long)
                additional = len(dset.y) - (num_tiles*self.num_classes)
                if additional > 0:
                    new_pred_labels = np.concatenate((new_pred_labels,np.ones(additional)))
                new_pred_labels = new_pred_labels.astype(np.long)
                mask = np.ones(len(dset.y)).astype(np.bool)
                old_pred_labels = new_pred_labels
            else:
                latents = self.get_latents(dset)
                umapped_latents = latents if ARGS.no_umap else umap.UMAP(min_dist=0,n_neighbors=60,n_components=2,random_state=42).fit_transform(latents.squeeze())
                model = hmm.GaussianHMM(self.num_classes,'full')
                model.params = 'mc'
                model.init_params = 'mc'
                model.startprob_ = np.ones(self.num_classes)/self.num_classes
                num_action_blocks = len([item for idx,item in enumerate(dset.y) if dset.y[idx-1] != item])
                prob_new_action = num_action_blocks/len(dset)
                model.transmat_ = (np.eye(self.num_classes) * (1-prob_new_action)) + (np.ones((self.num_classes,self.num_classes))*prob_new_action/self.num_classes)
                model.fit(umapped_latents)
                new_pred_labels = model.predict(umapped_latents)
                new_pred_probs = model.predict_proba(umapped_latents)
                mask = new_pred_probs.max(axis=1) >= prob_thresh
                print('Prob_thresh mask:',sum(mask),sum(mask)/len(new_pred_labels))
                if ARGS.save: np.save('test_umapped_latents.npy',umapped_latents)
                if meta_pivot_pred_labels is not 'none':
                    new_pred_labels = translate_labellings(new_pred_labels,meta_pivot_pred_labels,subsample_size=30000)
                elif epoch_num >0:
                    new_pred_labels = translate_labellings(new_pred_labels,old_pred_labels,subsample_size=30000)
            pseudo_label_dset = deepcopy(dset)
            pseudo_label_dset.y = cudify(new_pred_labels)
            if epoch_num > 0:
                mask2 = new_pred_labels==old_pred_labels
                mask = mask*mask2
                assert (new_pred_labels[mask]==old_pred_labels[mask]).all()
            super_mask*=mask
            mask_to_use = (mask+super_mask)/2
            mlp_preds = self.pseudo_label_train(pseudo_label_dset,mask=cudify(mask_to_use),num_epochs=num_pseudo_label_epochs,position_in_meta_loop=epoch_num)
            #print('translating labelling')
            #print('pseudo label training')
            #counts = {selected_acts[int(item)]:sum(new_pred_labels==item) for item in set(new_pred_labels)}
            #mask_counts = {selected_acts[int(item)]:sum(new_pred_labels[mask]==item) for item in set(new_pred_labels[mask])}
            #print('Counts:',counts)
            y_np = numpyify(dset.y)
            #print('Masked Counts:',mask_counts)
            print('Super Masked Counts:',label_counts(y_np[super_mask]))
            print('Latent accuracy:', accuracy(new_pred_labels,np_gt_labels))
            print('Masked latent accuracy:', accuracy(new_pred_labels[mask],y_np[mask]),mask.sum())
            print('Super Masked latent accuracy:', accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum())
            print('MLP accuracy:', accuracy(mlp_preds,np_gt_labels))
            print('Masked MLP accuracy:', accuracy(mlp_preds[mask],dset.y[mask]),mask.sum())
            print('Super Masked MLP accuracy:', accuracy(mlp_preds[super_mask],dset.y[super_mask]),super_mask.sum())
            rand_idxs = np.array([15,1777,1982,9834,11243,25,7777,5982,5834,250,7717,5912,5134])
            preds_for_printing = translate_labellings(new_pred_labels,np_gt_labels,'none')
            #for action_num in np.unique(np_gt_labels):
            #    action_preds = preds_for_printing[np_gt_labels==action_num]
            #    action_name = selected_acts[action_num]
            #    num_correct = (action_preds==action_num).sum()
            #    total_num = len(action_preds)
            #    print(f"{action_name}: {round(num_correct/total_num,3)} ({num_correct}/{total_num})")
            #print('GT:',dset.y[rand_idxs].int().tolist())
            #print('Old:',old_pred_labels[rand_idxs])
            #print('New:',new_pred_labels[rand_idxs])
            old_pred_labels = deepcopy(new_pred_labels)
        super_super_mask = np.logical_and(super_mask,new_pred_labels==mlp_preds)
        return new_pred_labels, mask, super_mask, super_super_mask

    def pseudo_label_cluster_meta_meta_loop(self,dset,num_meta_meta_epochs,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
        y_np = numpyify(dset.y)
        combined_super_mask_preds = -np.ones(len(dset))
        combined_super_super_mask_preds = -np.ones(len(dset))
        combined_mask_preds = -np.ones(len(dset))
        best_preds_so_far = -np.ones(len(dset))
        got_by_super_masks = np.zeros(len(dset)).astype(np.bool)
        got_by_super_super_masks = np.zeros(len(dset)).astype(np.bool)
        got_by_masks = np.zeros(len(dset)).astype(np.bool)

        preds_histories = []
        super_super_mask_histories = []
        super_mask_histories = []
        mask_histories = []

        for meta_meta_epoch in range(num_meta_meta_epochs):
            print('\nMETA META EPOCH:', meta_meta_epoch)
            meta_pivot_pred_labels = best_preds_so_far if meta_meta_epoch > 0 else 'none'
            preds, mask, super_mask, super_super_mask = self.pseudo_label_cluster_meta_loop(dset,meta_pivot_pred_labels, num_meta_epochs=num_meta_epochs,num_pseudo_label_epochs=num_pseudo_label_epochs,prob_thresh=prob_thresh,selected_acts=selected_acts)
            preds_histories.append(preds)
            super_mask_histories.append(super_mask)
            super_super_mask_histories.append(super_super_mask)
            mask_histories.append(mask)
            got_by_super_masks = np.logical_or(got_by_super_masks,super_mask)
            got_by_super_super_masks = np.logical_or(got_by_super_super_masks,super_super_mask)
            got_by_masks = np.logical_or(got_by_masks,mask)
            combined_super_mask_preds[super_mask] = preds[super_mask]
            combined_super_super_mask_preds[super_super_mask] = preds[super_super_mask]
            combined_mask_preds[mask] = preds[mask]
            if meta_meta_epoch == 0: best_preds_so_far = preds
            best_preds_so_far[got_by_masks] = combined_mask_preds[got_by_masks]
            best_preds_so_far[got_by_super_masks] = combined_super_mask_preds[got_by_super_masks]
            best_preds_so_far[got_by_super_super_masks] = combined_super_super_mask_preds[got_by_super_super_masks]
            print('Acc of best so far', accuracy(best_preds_so_far,y_np))

        surely_correct = np.stack(super_mask_histories).all(axis=0)
        macc = lambda mask: accuracy(best_preds_so_far[mask],y_np[mask])
        print(f"Accuracy on just the masks: {accuracy(combined_mask_preds,y_np)}")
        print(f"Masks masked: {accuracy(combined_mask_preds[got_by_masks],y_np[got_by_masks])}, and full: {accuracy(combined_mask_preds,y_np)}")
        print(f"Label counts for just the masks: {label_counts(combined_mask_preds)}")
        print(f"Label counts missed by masks: {label_counts(y_np[~got_by_masks])}")
        combined_mask_preds[~got_by_masks] = preds[~got_by_masks]
        print(f"Accuracy with gaps filled by preds: {accuracy(combined_mask_preds,y_np)}")

        print(f"Super masks masked: {accuracy(combined_super_mask_preds[got_by_super_masks],y_np[got_by_super_masks])}, and full: {accuracy(combined_super_mask_preds,y_np)}")
        print(f"Label counts for just the super masks: {label_counts(combined_super_mask_preds)}")
        print(f"Label counts missed by super masks: {label_counts(y_np[~got_by_super_masks])}")
        combined_super_mask_preds[~got_by_super_masks] = combined_mask_preds[~got_by_super_masks]
        print(f"Accuracy with gaps filled by mask preds and preds: {accuracy(combined_super_mask_preds,y_np)}")
        print(accuracy(best_preds_so_far,y_np))
        set_trace()
        print(f"Surely corrects: {macc(surely_correct)}")
        print(acc_by_label(best_preds_so_far[surely_correct],y_np[surely_correct]))
        if ARGS.save:
            np_save('super_super_mask_histories.npy',np.stack(super_super_mask_histories))
            np_save('super_mask_histories.npy',np.stack(super_mask_histories))
            np_save('mask_histories.npy',np.stack(mask_histories))
            np_save('preds_histories.npy',np.stack(preds_histories))

        #return pseudo_labels_to_use, mask_to_use

    def full_train(self,user_dsets,args):
        preds_from_users_list = []
        for user_id, (user_dset, sa) in enumerate(user_dsets):
            pseudo_labels, conf_mask, very_conf_mask = self.pseudo_label_cluster_meta_loop(user_dset,args.num_cluster_epochs,num_pseudo_label_epochs=5,prob_thresh=args.prob_thresh,selected_acts=sa)
            pseudo_label_dset = deepcopy(user_dset)
            pseudo_label_dset.y = cudify(pseudo_labels)
            pseudo_label_mask = cudify(very_conf_mask)
            self.train_on(pseudo_label_dset,8,pseudo_label_mask)
            preds_from_this_user_list = []
            for val_user_idx, (val_user_dset,sa) in enumerate(user_dsets):
                acc, f1, preds = self.val_on(val_user_dset)
                preds_from_this_user_list.append(preds)
            preds_from_this_user_array = np.concatenate(preds_from_this_user_list)
            preds_from_users_list.append(preds_from_this_user_array)
        debabled_mega_ultra_preds = debable(preds_from_users_list,pivot='none')
        user_break_points = [sum(len(ud) for ud,sa in user_dsets[:i]) for i in range(len(user_dsets)+1)]
        translated_self_preds = [user_preds[user_break_points[i]:user_break_points[i+1]] for i ,user_preds in enumerate(debabled_mega_ultra_preds)]
        accs = [accuracy(translated_self_preds[i],ud.y) for i, (ud,sa) in enumerate(user_dsets)]
        weighted_sum_acc = sum([a*len(ud) for a, (ud, sa) in zip(accs, user_dsets)])/user_break_points[-1]
        print(accs)
        print(f"Total acc: {weighted_sum_acc}")

    def load_and_find_hard_classes(self,dset,args):
        super_super_mask_histories = np.stack(np.load('super_super_masks.npy'))
        preds_histories = np.stack(np.load('preds.npy'))
        masked_mode_preds = masked_mode(preds_histories,super_super_mask_histories)
        got_by_super_super_masks = super_super_mask_histories.any(axis=0)
        y_np = numpyify(dset.y).astype(int)
        labels_with_enough_preds = unique_labels(masked_mode_preds[got_by_super_super_masks])
        labels_without_enough_preds = [k for k in unique_labels(y_np) if k not in labels_with_enough_preds]
        simplified_labels, compress_dict1, changed = compress_labels(masked_mode_preds)
        catch_all_label = get_num_labels(masked_mode_preds[got_by_super_super_masks])
        assert catch_all_label not in labels_with_enough_preds and catch_all_label-1 in unique_labels(simplified_labels)

        likely_hard_class_histories = np.zeros(len(dset)).astype(np.bool)
        for hard_class in labels_without_enough_preds:
            class_mask = masked_mode_preds==hard_class
            likely_hard_class_histories = np.logical_or(likely_hard_class_histories,class_mask)
        likely_hard_class = likely_hard_class_histories.all(axis=0)

        simplified_labels[likely_hard_class] = catch_all_label
        train_mask = np.logical_or(got_by_super_super_masks,likely_hard_class).astype(float)
        not_to_be_trained_on = simplified_labels==-1
        simplified_labels[not_to_be_trained_on] = catch_all_label # Can't have -1's when training, these points will be masked out anyway
        pseudo_label_dset = deepcopy(dset)
        pseudo_label_dset.y = cudify(simplified_labels)
        gt_trans_dict = {}

        tmp_label = max(y_np) + 1
        assert ~(y_np==tmp_label).any() and (y_np==(tmp_label-1)).any()
        for lab in unique_labels(y_np): # Keep labels the same in surely correct, others will be changed to catch_all
            if lab in unique_labels(y_np[got_by_super_super_masks]): gt_trans_dict[lab] = lab
            else: gt_trans_dict[lab] = tmp_label
        simplified_gt = np.array([gt_trans_dict[lab] for lab in y_np])
        simplified_gt, compress_dict, changed = compress_labels(simplified_gt)
        if compress_dict[tmp_label] != catch_all_label:
            print(f"Warning: the preds contain {catch_all_label} unique labels, but the corresponding gts, only {compress_dict[tmp_label]}")
            compress_dict[tmp_label] = catch_all_label

        x, _ = get_trans_dict(masked_mode_preds[got_by_super_super_masks],y_np[got_by_super_super_masks],'none')
        x1, _ = get_trans_dict(simplified_labels[got_by_super_super_masks],simplified_gt[got_by_super_super_masks],'none')
        for lab in unique_labels(masked_mode_preds[got_by_super_super_masks]): x1[compress_dict1[lab]], compress_dict[x[lab]]
        num_conf_labels = len(labels_with_enough_preds)
        descale_catch_all_label_by = 100*likely_hard_class.sum()*num_conf_labels/got_by_super_super_masks.sum()
        train_mask[likely_hard_class] /= descale_catch_all_label_by
        assert (train_mask[np.logical_and(simplified_labels==catch_all_label,~not_to_be_trained_on)] == 1/descale_catch_all_label_by).all()
        self.mlp = Var_BS_MLP(32,25,num_conf_labels+1).cuda() # Plus one is for the catch_all_label
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(),lr=ARGS.mlp_lr)
        acc, f1, train_preds, train_confs = self.train_on(pseudo_label_dset,args.num_pseudo_label_epochs,multiplicative_mask=cudify(train_mask),rlmbda=0)
        print(acc_by_label(train_preds,simplified_gt))
        print(label_counts(simplified_gt[train_preds==catch_all_label]))
        v_likely_hard_class = train_preds == 5
        vv_likely_hard_class = v_likely_hard_class
        thresh = 0
        set_trace()
        # Select the most confident half of those assigned to catch_all_label
        while vv_likely_hard_class.sum() > len(dset)*len(labels_without_enough_preds)/(self.num_classes*2):
            thresh += 0.1
            vv_likely_hard_class = np.logical_and(v_likely_hard_class,train_confs>thresh)
        np.save('vv_likely_hard_class.npy',vv_likely_hard_class)
        print(acc,f1)

def masked_mode(pred_array,mask):
    x = mask.any(axis=0)
    return np.array([stats.mode([lab for lab,b in zip(pred_array[:,j],mask[:,j]) if b]).mode[0] if bx else -1 for bx,j in zip(x,range(pred_array.shape[1]))])

def recursive_np_or(boolean_arrays):
    if len(boolean_arrays) == 1: return boolean_arrays[0]
    return np.logical_or(boolean_arrays[0],recursive_np_or(boolean_arrays[1:]))

def recursive_np_and(boolean_arrays):
    if len(boolean_arrays) == 1: return boolean_arrays[0]
    return np.logical_and(boolean_arrays[0],recursive_np_and(boolean_arrays[1:]))

def acc_by_label(labels1, labels2, subsample_size='none'):
    labels1 = translate_labellings(labels1,labels2,subsample_size)
    acc_by_labels_from = {}
    for label in np.unique(labels1):
        label_preds = labels2[labels1==label]
        num_correct = (label_preds==label).sum()
        total_num = len(label_preds)
        acc_by_labels_from[label] = round(num_correct/total_num,4)
    acc_by_labels_to = {}
    for label in np.unique(labels2):
        label_preds = labels1[labels2==label]
        num_correct = (label_preds==label).sum()
        total_num = len(label_preds)
        acc_by_labels_to[label] = round(num_correct/total_num,4)
    return acc_by_labels_from, acc_by_labels_to

def select_by_label(labelling,labels_to_select_by):
    return recursive_np_or([labelling==lab for lab in labels_to_select_by])

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
    #dsets_by_id = make_dsets_by_user(args,subj_ids)
    #bad_ids = []
    #for user_id, (dset,sa) in dsets_by_id.items():
    #    n = get_num_labels(dset.y)
    #    if n < num_labels/2:
    #        print(f"Excluding user {user_id}, only has {n} different labels, instead of {num_labels}")
    #        bad_ids.append(user_id)
    #dsets_by_id = [v for k,v in dsets_by_id.items() if k not in bad_ids]
    #har.full_train(dsets_by_id,args)

    #dset, selected_acts = list(dsets_by_id.values())[0]
    #har.load_and_train(dset,args)

    dset_train, dset_val, selected_acts = make_dset_train_val(args,subj_ids)
    if args.load_and_try:
        mask = np.load('super_masks.npy').any(axis=0)
        pseudo_labels = np.load('try_pseudo_labels.npy')
        pseudo_label_dset = deepcopy(dset_train)
        pseudo_labels[~mask] = 3 # Can't have -1's in loss function, these dps will be masked out anyway
        pseudo_label_dset.y = cudify(pseudo_labels)
        pseudo_label_counts = label_counts(pseudo_labels)
        class_counts_per_dp = np.array(list(pseudo_label_counts[t] for t in pseudo_labels))
        class_weights = 1./class_counts_per_dp
        print(class_weights)
        sampler = data.WeightedRandomSampler(class_weights,len(class_weights))
        train_acc, train_f1, train_preds, train_confs = har.train_on(pseudo_label_dset,multiplicative_mask=cudify(mask).int(),num_epochs=10,custom_sampler=sampler)
        print(train_acc,train_f1)
        val_acc, val_f1, val_preds = har.val_on(dset_val)
        print(val_acc,val_f1)
        facc,ff1,fpreds,fconfs = har.train_with_fract_gts_on(dset_train,10,0.1)
        print(f"acc with frac gts:{facc}")
        set_trace()
    elif args.load_and_find:
        har.load_and_find_hard_classes(dset_train,args)
    elif args.sub_train:
        super_mask_histories = np.load('super_masks.npy')
        preds_histories = np.load('preds.npy')
        surely_correct = np.stack(super_mask_histories).all(axis=0)
        masked_mode_labels = masked_mode(preds_histories,super_mask_histories)
        num_already_taken_care_of = get_num_labels(preds_histories[0][surely_correct])
        vv_likely_hard_class = np.load('vv_likely_hard_class.npy')
        x_np = numpyify(dset_train.x)
        x_chunks=np.stack([x_np[i*args.step_size:(i*args.step_size) + args.window_size] for i,b in enumerate(vv_likely_hard_class) if b])
        tdset = ChunkDataset(cudify(x_chunks),cudify(numpyify(dset_train.y)[vv_likely_hard_class]),'cuda')
        mlp = Var_BS_MLP(32,25,num_labels)
        num_hard_classes = num_labels - num_already_taken_care_of
        mlp = Var_BS_MLP(32,25,num_hard_classes).cuda()
        sub_har_learner = HARLearner(enc=enc,dec=dec,mlp=mlp,batch_size=args.batch_size,num_classes=num_hard_classes)
        sub_har_learner.pseudo_label_cluster_meta_meta_loop(tdset,args.num_meta_meta_epochs,args.num_meta_epochs,args.num_pseudo_label_epochs,args.prob_thresh,selected_acts)
    else:
        har.pseudo_label_cluster_meta_meta_loop(dset_train,args.num_meta_meta_epochs,args.num_meta_epochs,args.num_pseudo_label_epochs,args.prob_thresh,selected_acts)
    train_end_time = time.time()
    total_prep_time = asMinutes(train_start_time-prep_start_time)
    total_train_time = asMinutes(train_end_time-train_start_time)
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
    parser.add_argument('--load_and_find',action='store_true')
    parser.add_argument('--load_and_try',action='store_true')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--noise',type=float,default=1.)
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--num_meta_epochs',type=int,default=4)
    parser.add_argument('--num_meta_meta_epochs',type=int,default=4)
    parser.add_argument('--num_pseudo_label_epochs',type=int,default=3)
    parser.add_argument('--num_cluster_epochs',type=int,default=5)
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--prob_thresh',type=float,default=.95)
    parser.add_argument('--rlmbda',type=float,default=.1)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--short_epochs',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--sub_train',action='store_true')
    parser.add_argument('--suppress_prints',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--no_umap',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.test and ARGS.save:
        print("Shouldn't be saving for a test run"); sys.exit()
    if ARGS.test:
        ARGS.num_meta_epochs = 2
        ARGS.num_meta_meta_epochs = 2
        ARGS.num_cluster_epochs = 2
        ARGS.num_pseudo_label_epochs = 2
    elif not ARGS.no_umap: import umap
    if ARGS.short_epochs:
        ARGS.num_meta_epochs = 1
        ARGS.num_cluster_epochs = 1
        ARGS.num_pseudo_label_epochs = 1
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
    main(ARGS,subj_ids=subj_ids)
