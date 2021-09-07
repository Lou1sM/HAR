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
from make_dsets import make_dset_train_val, make_dsets_by_user, ChunkDataset, combine_dsets


class EncByLayer(nn.Module):
    def __init__(self,x_filters,y_filters,x_strides,y_strides,max_pools,show_shapes):
        super(EncByLayer,self).__init__()
        self.show_shapes = show_shapes
        num_layers = len(x_filters)
        assert all(len(x)==num_layers for x in (y_filters,x_strides,y_strides,max_pools))
        ncvs = [1]+[4*2**i for i in range(num_layers)]
        conv_layers = []
        for i in range(num_layers):
            if i<num_layers-1:
                conv_layer = nn.Sequential(
                nn.Conv2d(ncvs[i],ncvs[i+1],(x_filters[i],y_filters[i]),(x_strides[i],y_strides[i])),
                nn.BatchNorm2d(ncvs[i+1]),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(max_pools[i])
                )
            else: #No batch norm on the last layer
                conv_layer = nn.Sequential(
                nn.Conv2d(ncvs[i],ncvs[i+1],(x_filters[i],y_filters[i]),(x_strides[i],y_strides[i])),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(max_pools[i])
                )
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self,x):
        if self.show_shapes: print(x.shape)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if self.show_shapes: print(x.shape)
        return x

class DecByLayer(nn.Module):
    def __init__(self,x_filters,y_filters,x_strides,y_strides,show_shapes):
        super(DecByLayer,self).__init__()
        self.show_shapes = show_shapes
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
        if self.show_shapes: print(x.shape)
        for conv_trans_layer in self.conv_trans_layers:
            x = conv_trans_layer(x)
            if self.show_shapes: print(x.shape)
        return x

class Var_BS_MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Var_BS_MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.LeakyReLU(0.3)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
        if x.shape[0] != 1:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

class HARLearner():
    def __init__(self,enc,mlp,dec,batch_size,num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.enc = enc
        self.dec = dec
        self.mlp = mlp
        self.pseudo_label_lf = avoid_minus_ones_lf_wrapper(nn.CrossEntropyLoss(reduction='none'))
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

    def train_on(self,dset,num_epochs,multiplicative_mask='none',lf=None,compute_acc=True,reinit=True,rlmbda=0,custom_sampler='none',noise=0.):
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
        is_mask = multiplicative_mask is not 'none'
        for epoch in range(num_epochs):
            pred_list = []
            idx_list = []
            conf_list = []
            best_f1 = 0
            for batch_idx, (xb,yb,idx) in enumerate(dl):
                if len(xb) == 1: continue # If last batch is only one element then batchnorm will error
                latent = self.enc(xb)
                if noise > 0: latent = noiseify(latent,noise)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                batch_mask = 'none' if not is_mask  else multiplicative_mask[:self.batch_size] if ARGS.test else multiplicative_mask[idx]
                loss = lf(label_pred,yb.long(),batch_mask)
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
                if ARGS.verbose:
                    print(acc)
                    if is_mask:
                        m = numpyify(multiplicative_mask.bool())
                        print(accuracy(pred_array_ordered[m],dset.y.detach().cpu().numpy()[m]))
                if ARGS.test or acc > best_acc:
                    best_pred_array_ordered = pred_array_ordered
                    best_conf_array_ordered = conf_array_ordered
                    best_acc = acc
                    best_f1 = f1
        return best_acc,best_f1,best_pred_array_ordered,best_conf_array_ordered

    def train_with_fract_gts_on(self,dset,num_epochs,frac_gt_labels,reinit=False,rlmbda=0):
        gt_mask = stratified_sample_mask(len(dset),frac_gt_labels)
        gt_mask = cudify(gt_mask)
        approx = gt_mask.sum()/len(dset)
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

    def pseudo_label_cluster_meta_loop(self,dset,meta_pivot_pred_labels,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
        old_pred_labels = -np.ones(dset.y.shape)
        np_gt_labels = dset.y.detach().cpu().numpy().astype(int)
        super_mask = np.ones(len(dset)).astype(np.bool)
        for epoch_num in range(num_meta_epochs):
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
            mlp_preds = self.train_on(pseudo_label_dset,multiplicative_mask=cudify(mask_to_use),num_epochs=num_pseudo_label_epochs)
            y_np = numpyify(dset.y)
            if ARGS.verbose:
                print('Meta Epoch:', epoch_num)
                print('Super Masked Counts:',label_counts(y_np[super_mask]))
                print('Latent accuracy:', accuracy(new_pred_labels,np_gt_labels))
                print('Masked latent accuracy:', accuracy(new_pred_labels[mask],y_np[mask]),mask.sum())
                print('Super Masked latent accuracy:', accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum())
                print('MLP accuracy:', accuracy(mlp_preds,np_gt_labels))
                print('Masked MLP accuracy:', accuracy(mlp_preds[mask],dset.y[mask]),mask.sum())
                print('Super Masked MLP accuracy:', accuracy(mlp_preds[super_mask],dset.y[super_mask]),super_mask.sum())
            elif epoch_num == num_meta_epochs-1:
                print(f"Latent: {accuracy(new_pred_labels,np_gt_labels)}\tMaskL: {accuracy(new_pred_labels[mask],y_np[mask]),mask.sum()}\tSuperMaskL{accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum()}")
            rand_idxs = np.array([15,1777,1982,9834,11243,25,7777,5982,5834,250,7717,5912,5134])
            preds_for_printing = translate_labellings(new_pred_labels,np_gt_labels,'none')
            old_pred_labels = deepcopy(new_pred_labels)
        super_super_mask = np.logical_and(super_mask,new_pred_labels==mlp_preds)
        return new_pred_labels, mask, super_mask, super_super_mask

    def pseudo_label_cluster_meta_meta_loop(self,dset,num_meta_meta_epochs,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
        y_np = numpyify(dset.y)
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

            super_super_mask_mode_preds = masked_mode(np.stack(preds_histories),np.stack(super_super_mask_histories))
            super_mask_mode_preds = masked_mode(np.stack(preds_histories),np.stack(super_mask_histories))
            mask_mode_preds = masked_mode(np.stack(preds_histories),np.stack(mask_histories))
            best_preds_so_far = masked_mode(np.stack(preds_histories))
            best_preds_so_far[got_by_masks] = mask_mode_preds[got_by_masks]
            best_preds_so_far[got_by_super_masks] = super_mask_mode_preds[got_by_super_masks]
            best_preds_so_far[got_by_super_super_masks] = super_super_mask_mode_preds[got_by_super_super_masks]
            assert not (best_preds_so_far==-1).any()
            best_acc = accuracy(best_preds_so_far,y_np)
            print('Acc of best so far',best_acc)

        surely_correct = np.stack(super_mask_histories).all(axis=0)
        macc = lambda mask: accuracy(best_preds_so_far[mask],y_np[mask])

        print(f"Super masks masked: {accuracy(super_mask_mode_preds[got_by_super_masks],y_np[got_by_super_masks])}, and full: {accuracy(super_mask_mode_preds,y_np)}")
        print(f"Label counts for just the super masks: {label_counts(super_mask_mode_preds)}")
        print(f"Label counts missed by super masks: {label_counts(y_np[~got_by_super_masks])}")
        print(f"Surely corrects: {macc(surely_correct)}")
        print(acc_by_label(best_preds_so_far[surely_correct],y_np[surely_correct]))
        if ARGS.save:
            np_save('super_super_mask_histories.npy',np.stack(super_super_mask_histories))
            np_save('super_mask_histories.npy',np.stack(super_mask_histories))
            np_save('mask_histories.npy',np.stack(mask_histories))
            np_save('preds_histories.npy',np.stack(preds_histories))

    def full_train(self,user_dsets,args):
        preds_from_users_list = []
        accs = []
        accs_from_others = []
        self_accs = []
        for user_id, (user_dset, sa) in enumerate(user_dsets):
            print(f"training on {user_id}")
            pseudo_labels, conf_mask, very_conf_mask, very_very_conf_mask  = self.pseudo_label_cluster_meta_loop(user_dset,'none',args.num_cluster_epochs,num_pseudo_label_epochs=5,prob_thresh=args.prob_thresh,selected_acts=sa)
            self_accs.append(accuracy(pseudo_labels,numpyify(user_dset.y)))
            pseudo_label_dset = deepcopy(user_dset)
            others_dsets = [udset for uid, (udset,sa) in enumerate(user_dsets) if uid!=user_id]
            if len(others_dsets) <= 1: print("not enough dsets to full_train, specify more subj_ids"); sys.exit()
            combined_other_dsets = combine_dsets(others_dsets)
            mask_for_others = stratified_sample_mask(len(combined_other_dsets),args.frac_gt_labels)
            acc,f1,preds,confs=self.train_on(combined_other_dsets,args.num_pseudo_label_epochs,cudify(mask_for_others))
            acc_,f1_,others_language_pseudo_label_preds = self.val_on(pseudo_label_dset)
            translated_pseudo_labels = translate_labellings(pseudo_labels,others_language_pseudo_label_preds,'none')
            pseudo_label_dset.y = cudify(translated_pseudo_labels)
            gt_translation = translate_labellings(pseudo_labels,others_language_pseudo_label_preds)
            translation_accuracy = (translated_pseudo_labels==gt_translation).mean()
            if translation_accuracy<1: print("translation acc is only {translation_accuracy}")
            acc_others,f1_others,preds_others = self.val_on(user_dset)
            print(label_counts(numpyify(user_dset.y)))
            print(f"acc from just others: {acc_others}")
            accs_from_others.append(acc_others)
            full_train_dsets = others_dsets + [pseudo_label_dset]
            full_combined_dsets = combine_dsets(full_train_dsets)
            mask = cudify(np.concatenate([mask_for_others,very_conf_mask]))
            self.train_on(full_combined_dsets,args.num_pseudo_label_epochs,mask)
            acc,f1,preds = self.val_on(user_dset)
            print(f"acc when add pseudo labels: {acc}")
            accs.append(acc)
        total_num_dpoints = sum(len(ud) for ud,sa in user_dsets)
        for n,acc_list in zip((accs_from_others,self_accs,accs),('others','self','both')):
            print(n,sum([a*len(ud) for a, (ud, sa) in zip(acc_list, user_dsets)])/total_num_dpoints)

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
        #simplified_labels[not_to_be_trained_on] = catch_all_label # Can't have -1's when training, these points will be masked out anyway
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

    def find_similar_users(self,user_dsets,args):
        preds_from_users_list = []
        accs = []
        all_similar_users = []
        for user_id, (user_dset, sa) in enumerate(user_dsets):
            user_accs = []
            print(f"training on {user_id}")
            self.train_with_fract_gts_on(user_dset,args.num_pseudo_label_epochs,args.frac_gt_labels)
            for other_id, (other_dset, sa) in enumerate(user_dsets):
                acc,f1,preds = self.val_on(other_dset)
                user_accs.append(acc)
            accs.append(user_accs)
            most_similar_users = numpyify(torch.topk(torch.tensor(user_accs),5)[1]).tolist()
            print(most_similar_users)
            if user_id in most_similar_users: most_similar_users.remove(user_id)
            all_similar_users.append(most_similar_users)
        full_accs = []
        for user_id, (user_dset, sa) in enumerate(user_dsets):
            others_dsets = [udset for uid, (udset,sa) in enumerate(user_dsets) if uid!=user_id]
            combined_other_dsets = combine_dsets(others_dsets)
            acc,f1,preds,confs=self.train_with_fract_gts_on(combined_other_dsets,args.num_pseudo_label_epochs,args.frac_gt_labels)
            acc,f1,preds = self.val_on(user_dset)
            full_accs.append(acc)
        accs.append(full_accs)
        similars_accs = []
        for user_id, (user_dset, sa) in enumerate(user_dsets):
            similars = all_similar_users[user_id]
            others_dsets = [udset for uid, (udset,sa) in enumerate(user_dsets) if uid in similars]
            combined_other_dsets = combine_dsets(others_dsets)
            acc,f1,preds,confs=self.train_with_fract_gts_on(combined_other_dsets,args.num_pseudo_label_epochs*2,args.frac_gt_labels)
            acc,f1,preds = self.val_on(user_dset)
            similars_accs.append(acc)
        accs.append(similars_accs)
        acc_mat = np.array(accs)
        print(acc_mat)

def stratified_sample_mask(population_length, sample_frac):
    if sample_frac == 0:
        sample_idx = np.array([], dtype=np.int)
    elif sample_frac <= 0.5:
        sample_idx = np.arange(population_length, step=int(1/sample_frac))
    else:
        non_sample_idx = np.arange(population_length, step=int(1/(1-sample_frac)))
        sample_idx = np.delete(np.arange(population_length),non_sample_idx)
    sample_mask = np.zeros(population_length)
    sample_mask[sample_idx] = 1
    return sample_mask

def masked_mode(pred_array,mask='none'):
    if mask is 'none':
        return stats.mode(pred_array,axis=0).mode[0]
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

def avoid_minus_ones_lf_wrapper(lf):
    def wrapped_lf(pred,target,multiplicative_mask='none'):
        avoidance_mask = target!=-1
        loss_array = lf(pred[avoidance_mask],target[avoidance_mask])
        if multiplicative_mask is not 'none':
            loss_array *= multiplicative_mask[avoidance_mask]
        return loss_array.mean()
    return wrapped_lf

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
        x_filters = (50,40,7,4)
        y_filters = (5,3,2,1)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = (2,2,2,2)
        x_filters_trans = (45,35,10,6)
        y_filters_trans = (8,8,6,6)
        x_strides_trans = (1,2,2,2)
        y_strides_trans = (1,1,2,1)
        num_labels = 12
    elif args.dset == 'UCI':
        x_filters = (60,40,4,4)
        x_strides = (2,2,1,1)
        y_filters = (1,1,3,2)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        x_strides_trans = (1,3,2,2)
        y_filters_trans = (2,2,2,2)
        y_strides_trans = (2,2,1,1)
        num_labels = 6
    elif args.dset == 'WISDM-v1':
        x_filters = (50,40,5,4)
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
        x_filters = (50,40,8,6)
        y_filters = (2,2,2,2)
        x_strides = (2,2,1,1)
        y_strides = (2,2,1,1)
        max_pools = ((2,1),(3,1),1,1)
        x_filters_trans = (31,30,14,10)
        y_filters_trans = (2,2,2,2)
        x_strides_trans = (1,3,2,2)
        y_strides_trans = (1,1,2,2)
        num_labels = 17
    enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,show_shapes=args.show_shapes)
    dec = DecByLayer(x_filters_trans,y_filters_trans,x_strides_trans,y_strides_trans,show_shapes=args.show_shapes)
    mlp = Var_BS_MLP(32,25,num_labels)
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        mlp.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    dec.cuda()
    mlp.cuda()
    dset_train, dset_val, selected_acts = make_dset_train_val(args,subj_ids)
    if args.show_shapes:
        num_ftrs = dset_train.x.shape[-1]
        print(num_ftrs)
        lat = enc(torch.ones((2,1,512,num_ftrs),device='cuda'))
        dec(lat)
        sys.exit()

    har = HARLearner(enc=enc,mlp=mlp,dec=dec,batch_size=args.batch_size,num_classes=num_labels)
    exp_dir = os.path.join(f'experiments/{args.exp_name}')

    train_start_time = time.time()
    dsets_by_id = make_dsets_by_user(args,subj_ids)
    bad_ids = []
    for user_id, (dset,sa) in dsets_by_id.items():
        n = get_num_labels(dset.y)
        if n < num_labels/2:
            print(f"Excluding user {user_id}, only has {n} different labels, instead of {num_labels}")
            bad_ids.append(user_id)
    dsets_by_id = [v for k,v in dsets_by_id.items() if k not in bad_ids]
    if args.train_type == 'train_frac_gts_as_single':
        print("TRAINING ON WITH FRAC GTS AS SINGLE DSET")
        acc,f1,preds,confs = har.train_with_fract_gts_on(dset_train,args.num_pseudo_label_epochs,args.frac_gt_labels)
        print(acc)
    elif args.train_type == 'full':
        print("FULL TRAINING")
        har.full_train(dsets_by_id,args)
    elif args.train_type == 'find_similar_users':
        print("FULL TRAINING")
        har.find_similar_users(dsets_by_id,args)
    elif args.train_type == 'cluster_as_single':
        print("CLUSTERING AS SINGLE DSET")
        har.pseudo_label_cluster_meta_meta_loop(dset_train,args.num_meta_meta_epochs,args.num_meta_epochs,args.num_pseudo_label_epochs,args.prob_thresh,selected_acts)
    elif args.train_type == 'cluster_individually':
        print("CLUSTERING AS EACH DSET SEPARATELY")
        for user_id, (dset,sa) in enumerate(dsets_by_id):
            print("clustering", user_id)
            har.pseudo_label_cluster_meta_meta_loop(dset,args.num_meta_meta_epochs,args.num_meta_epochs,args.num_pseudo_label_epochs,args.prob_thresh,selected_acts)
    train_end_time = time.time()
    total_prep_time = asMinutes(train_start_time-prep_start_time)
    total_train_time = asMinutes(train_end_time-train_start_time)
    print(f"Prep time: {total_prep_time}\tTrain time: {total_train_time}")


if __name__ == "__main__":

    dset_options = ['PAMAP','UCI','WISDM-v1','WISDM-watch']
    training_type_options = ['full','cluster_as_single','cluster_individually','train_frac_gts_as_single','find_similar_users']
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--num_subjs',type=int)
    group.add_argument('--subj_ids',type=str,nargs='+',default=['first'])
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--alpha',type=float,default=.5)
    parser.add_argument('--batch_size',type=int,default=128)
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
    parser.add_argument('--no_umap',action='store_true')
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
    parser.add_argument('--train_type',type=str,choices=training_type_options)
    parser.add_argument('--show_shapes',action='store_true',help='print the shapes of hidden layers in enc and dec')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.test and ARGS.save:
        print("Shouldn't be saving for a test run"); sys.exit()
    if ARGS.test:
        ARGS.num_meta_epochs = 1
        ARGS.num_meta_meta_epochs = 1
        ARGS.num_cluster_epochs = 1
        ARGS.num_pseudo_label_epochs = 1
    elif not ARGS.no_umap and not ARGS.show_shapes: import umap
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
