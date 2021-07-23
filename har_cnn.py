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
from dl_utils.misc import asMinutes
from dl_utils.label_funcs import accuracy, mean_f1, debable, translate_labellings, compress_labels, get_num_labels, label_counts, dummy_labels
from dl_utils.tensor_funcs import noiseify
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
        self.pseudo_label_lf = nn.CrossEntropyLoss(reduction='mean')
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
                return -1, -1, dummy_labels(self.num_classes,len(dset.y))
            pred_array = np.concatenate(pred_list)
            idx_array = np.concatenate(idx_list)
            pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
            if compute_acc:
                acc = -1 if ARGS.test else accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
                f1 = -1 if ARGS.test else mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
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

    def pseudo_label_train(self,pseudo_label_dset,mask,num_epochs):
        self.enc.train()
        pseudo_label_dl = data.DataLoader(pseudo_label_dset,batch_sampler=data.BatchSampler(data.RandomSampler(pseudo_label_dset),self.batch_size,drop_last=False),pin_memory=False)
        for epoch in range(num_epochs):
            epoch_loss = 0
            best_loss = np.inf
            pred_list = []
            idx_list = []
            for batch_idx, (xb,yb,idx) in enumerate(pseudo_label_dl):
                batch_mask = mask[idx]
                latent = self.enc(xb)
                latent = noiseify(latent,ARGS.noise)
                if batch_mask.any():
                    try:
                        pseudo_label_pred = self.mlp(latent[:,:,0,0])
                    except ValueError: set_trace()
                    pseudo_label_loss = self.pseudo_label_lf(pseudo_label_pred[batch_mask],yb.long()[batch_mask])
                else:
                    pseudo_label_loss = torch.tensor(0,device=self.device)
                if not batch_mask.all():
                    latents_to_rec_train = latent[~batch_mask]
                    rec_pred = self.dec(latents_to_rec_train)
                    rec_loss = self.rec_lf(rec_pred,xb[~batch_mask])/(~batch_mask).sum()
                else:
                    rec_loss = torch.tensor(0,device=pseudo_label_dset.device)
                loss = pseudo_label_loss + rec_loss
                if math.isnan(loss): set_trace()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                pred_list.append(pseudo_label_pred.argmax(axis=1).detach().cpu().numpy())
                idx_list.append(idx.detach().cpu().numpy())
                if ARGS.test: break
            if ARGS.test:
                pred_array_ordered = dummy_labels(self.num_classes,len(pseudo_label_dset))
                break
            pred_array = np.concatenate(pred_list)
            idx_array = np.concatenate(idx_list)
            pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
        return pred_array_ordered

    def pseudo_label_cluster_meta_loop(self,dset,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
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
                print('umapping')
                umapped_latents = latents if ARGS.no_umap else umap.UMAP(min_dist=0,n_neighbors=60,n_components=2,random_state=42).fit_transform(latents.squeeze())
                model = hmm.GaussianHMM(self.num_classes,'full')
                print('modelling')
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
                new_pred_labels = translate_labellings(new_pred_labels,np_gt_labels,subsample_size=30000)
            if epoch_num > 0:
                mask2 = new_pred_labels==old_pred_labels
                print('Sames:', sum(mask2), sum(mask2)/len(new_pred_labels))
                mask = mask*mask2
                assert (new_pred_labels[mask]==old_pred_labels[mask]).all()
            pseudo_label_dset = deepcopy(dset)
            pseudo_label_dset.y = cudify(new_pred_labels)
            mlp_preds = self.pseudo_label_train(pseudo_label_dset,mask,num_pseudo_label_epochs)
            super_mask*=mask
            print('translating labelling')
            print('pseudo label training')
            #print('Counts:',label_counts(new_pred_labels))
            #print('Masked Counts:',label_counts(new_pred_labels[mask]))
            print('Super Masked Counts:',label_counts(new_pred_labels[super_mask]))
            print('Latent accuracy:', accuracy(new_pred_labels,np_gt_labels))
            print('Masked Latent accuracy:', accuracy(new_pred_labels[mask],dset.y[mask]),mask.sum())
            print('Super Masked Latent accuracy:', accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum())
            print('MLP accuracy:', accuracy(mlp_preds,np_gt_labels))
            rand_idxs = np.array([15,1777,1982,9834,11243,25,7777,5982,5834,250,7717,5912,5134])
            #for action_num in np.unique(np_gt_labels):
            #    action_preds = new_pred_labels[np_gt_labels==action_num]
            #    action_name = selected_acts[action_num]
            #    num_correct = (action_preds==action_num).sum()
            #    total_num = len(action_preds)
            #    print(f"{action_name}: {round(num_correct/total_num,3)} ({num_correct}/{total_num})")
            #print('GT:',dset.y[rand_idxs].int().tolist())
            #print('Old:',old_pred_labels[rand_idxs])
            #print('New:',new_pred_labels[rand_idxs])
            old_pred_labels = deepcopy(new_pred_labels)
        return new_pred_labels, mask, super_mask

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
    dsets_by_id = make_dsets_by_user(args,subj_ids)
    bad_ids = []
    for user_id, (dset,sa) in dsets_by_id.items():
        n = get_num_labels(dset.y)
        if n < num_labels/2:
            print(f"Excluding user {user_id}, only has {n} different labels, instead of {num_labels}")
            bad_ids.append(user_id)
    dsets_by_id = [v for k,v in dsets_by_id.items() if k not in bad_ids]
    har.full_train(dsets_by_id,args)
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
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--noise',type=float,default=1.)
    parser.add_argument('--num_epochs',type=int,default=30)
    parser.add_argument('--num_pseudo_label_epochs',type=int,default=3)
    parser.add_argument('--num_cluster_epochs',type=int,default=5)
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--prob_thresh',type=float,default=.95)
    parser.add_argument('--rlmbda',type=float,default=.1)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--short_epochs',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
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
