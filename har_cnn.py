import sys
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from hmmlearn import hmm
from copy import deepcopy
import os
import math
from pdb import set_trace
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils import data
import cl_args
from dl_utils.misc import asMinutes,check_dir
from dl_utils.label_funcs import accuracy, mean_f1, debable, translate_labellings, get_num_labels, label_counts, dummy_labels, avoid_minus_ones_lf_wrapper,masked_mode,acc_by_label
from dl_utils.tensor_funcs import noiseify, numpyify, cudify
from make_dsets import make_dset_train_val, make_dsets_by_user
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score


class DoubleEnc(nn.Module):
    def __init__(self,enc1,enc2):
        self.enc1 = enc1
        self.enc2 = enc2
        self.encs = nn.ModuleList((enc1,enc2))

    def forward(self,x1,x2):
        out1 = self.enc(x1)
        out2 = self.enc1(x2)
        out = torch.cat((out1,out2.view((out2.shape[0],-1,1,1))),axis=1)
        return out

class EncByLayer(nn.Module):
    def __init__(self,x_filters,y_filters,x_strides,y_strides,max_pools,nf1,show_shapes):
        super(EncByLayer,self).__init__()
        self.show_shapes = show_shapes
        num_layers = len(x_filters)
        assert all(len(x)==num_layers for x in (y_filters,x_strides,y_strides,max_pools))
        ncvs = [1]+[nf1*2**i for i in range(num_layers)]
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

class ProximalSampler(data.Sampler):
    def __init__(self, data_source, permute_prob) -> None:
        self.data_source = data_source
        self.permute_prob = permute_prob

    def __iter__(self):
        n = len(self.data_source)
        idxs = torch.arange(n)
        to_permute = torch.FloatTensor(n).uniform_() < self.permute_prob
        idxs[to_permute] = idxs[to_permute][torch.randperm(to_permute.sum())]
        yield from idxs

    def __len__(self) -> int:
        return len(self.data_source)

class HARLearner():
    #def __init__(self,enc,dec,mlp,temp_prox_mlp,batch_size,temp_prox_batch_size,num_classes):
    def __init__(self,enc,mlp,batch_size,temp_prox_batch_size,num_classes):
        self.batch_size = batch_size
        self.temp_prox_batch_size = temp_prox_batch_size
        self.num_classes = num_classes
        self.enc = enc
        self.mlp = mlp
        self.rec_lf = nn.MSELoss(reduction='none')
        self.pseudo_label_lf = avoid_minus_ones_lf_wrapper(nn.CrossEntropyLoss(reduction='none'))
        self.temp_prox_lf = nn.MSELoss()

        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=ARGS.enc_lr)
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(),lr=ARGS.mlp_lr)

    def get_latents(self,dset):
        self.enc.eval()
        collected_latents = []
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),self.batch_size,drop_last=False),pin_memory=False)
        for idx, (xb,yb,tb) in enumerate(determin_dl):
            batch_latent = self.enc(xb)
            batch_latent = batch_latent.view(batch_latent.shape[0],-1).detach().cpu().numpy()
            collected_latents.append(batch_latent)
        collected_latents = np.concatenate(collected_latents,axis=0)
        return collected_latents

    def train_on(self,dset,num_epochs,multiplicative_mask='none',lf=None,compute_acc=True,reinit=False,rlmbda=0,custom_sampler='none',noise=0.):
        if reinit: self.reinit_nets()
        self.enc.train()
        #self.dec.train()
        self.mlp.train()
        best_acc = 0
        best_f1 = 0
        best_pred_array_ordered = -np.ones(len(dset.y))
        sampler = data.RandomSampler(dset) if custom_sampler is 'none' else custom_sampler
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(sampler,self.batch_size,drop_last=False),pin_memory=False)
        is_mask = multiplicative_mask is not 'none'
        temp_prox_sampler = ProximalSampler(dset, permute_prob=ARGS.permute_prob)
        temp_prox_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(temp_prox_sampler,self.temp_prox_batch_size,drop_last=False),pin_memory=False)
        temp_prox_targets_table = torch.log(torch.arange(1,len(dset)+1).float()).cuda()
        for epoch in range(num_epochs):
            pred_list = []
            idx_list = []
            conf_list = []
            best_f1 = 0
            ls = []
            ds = []
            print('temp prox training')
            for batch_idx, (xb,yb,idx) in enumerate(temp_prox_dl):
                if ARGS.skip_train: break
                latent = self.enc(xb)[:,:,0,0]
                latent_dists = torch.norm(latent[:,None] - latent,dim=2)
                ds.append(latent_dists.mean().item())
                temp_prox_targets = temp_prox_targets_table[(idx - idx[:,None]).abs()]
                temp_prox_loss = self.temp_prox_lf(latent_dists,temp_prox_targets)
                ls.append(temp_prox_loss.item())
                temp_prox_loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                if ARGS.short_epochs and batch_idx == 200: break
                if ARGS.test: break
            print('generating')
            for batch_idx, (xb,yb,idx) in enumerate(dl):
                if len(xb) == 1: continue # If last batch is only one element then batchnorm will error
                latent = self.enc(xb)
                if noise > 0: latent = noiseify(latent,noise)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
                label_pred = self.mlp(latent[:,:,0,0])
                loss = self.pseudo_label_lf(label_pred,yb.long())*0.1
                if math.isnan(loss): set_trace()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
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

    def val_on(self,dset):
        self.enc.eval()
        #self.dec.eval()
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
        print(888)
        for m in self.enc.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m,nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
        if hasattr(self,'dec'):
            for m in self.dec.modules():
                if isinstance(m,nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform(m.weight.data)
                    torch.nn.init.zeros_(m.bias.data)
                elif isinstance(m,nn.BatchNorm2d):
                    torch.nn.init.ones_(m.weight.data)
                    torch.nn.init.zeros_(m.bias.data)
        if hasattr(self,'mlp'):
            for m in self.mlp.modules():
                if isinstance(m,nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight.data)
                    torch.nn.init.zeros_(m.bias.data)
                elif isinstance(m,nn.BatchNorm1d):
                    torch.nn.init.ones_(m.weight.data)
                    torch.nn.init.zeros_(m.bias.data)

    def pseudo_label_cluster_meta_loop(self,dset,meta_pivot_pred_labels,num_meta_epochs,num_pseudo_label_epochs,selected_acts):
        np_gt_labels = dset.y.detach().cpu().numpy().astype(int)
        for epoch_num in range(num_meta_epochs):
            start_time = time.time()
            if ARGS.test:
                num_tiles = len(dset.y)//self.num_classes
                gmm_labels = np.tile(np.arange(self.num_classes),num_tiles).astype(np.long)
                additional = len(dset.y) - (num_tiles*self.num_classes)
                if additional > 0:
                    gmm_labels = np.concatenate((gmm_labels,np.ones(additional)))
                gmm_labels = gmm_labels.astype(np.long)
                old_gmm_labels = gmm_labels
            else:
                latents = self.get_latents(dset)
                c = GaussianMixture(n_components=self.num_classes,n_init=5)
                gmm_labels = c.fit_predict(latents)
            pseudo_label_dset = deepcopy(dset)
            pseudo_label_dset.y = cudify(gmm_labels)
            mlp_acc,mlp_f1,mlp_preds,mlp_confs = self.train_on(pseudo_label_dset,num_epochs=num_pseudo_label_epochs)
            gmm_acc = accuracy(gmm_labels,np_gt_labels)
            print('GMM accuracy:', gmm_acc)
            print("Epoch time:", asMinutes(time.time() - start_time))
        return gmm_labels

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
            preds, mask, super_mask, super_super_mask, mlp_accs, cluster_accs = self.pseudo_label_cluster_meta_loop(dset,meta_pivot_pred_labels, num_meta_epochs=num_meta_epochs,num_pseudo_label_epochs=num_pseudo_label_epochs,prob_thresh=prob_thresh,selected_acts=selected_acts)
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
            best_nmi = normalized_mutual_info_score(best_preds_so_far,y_np)
            best_rand_idx = adjusted_rand_score(best_preds_so_far,y_np)
            print('Results of best so far',best_acc,best_nmi,best_rand_idx)

        surely_correct = np.stack(super_mask_histories).all(axis=0)
        macc = lambda mask: accuracy(best_preds_so_far[mask],y_np[mask])

        return best_preds_so_far

    def full_train(self,user_dsets,args):
        preds_from_users_list = []
        accs_from_users_list = []
        self_accs = []
        self_f1s = []
        self_preds = []
        for user_id, (user_dset, sa) in enumerate(user_dsets):
            preds_from_this_user = []
            accs_from_this_user = []
            print(f"training on {user_id}")
            pseudo_labels = self.pseudo_label_cluster_meta_meta_loop(user_dset,num_meta_meta_epochs=args.num_meta_meta_epochs,num_meta_epochs=args.num_meta_epochs,num_pseudo_label_epochs=args.num_pseudo_label_epochs,prob_thresh=args.prob_thresh,selected_acts=sa)
            self_accs.append(accuracy(pseudo_labels,numpyify(user_dset.y)))
            self_f1s.append(mean_f1(pseudo_labels,numpyify(user_dset.y)))
            self_preds.append(pseudo_labels)
            for other_user_id, (other_user_dset, sa) in enumerate(user_dsets):
                acc,f1,preds = self.val_on(other_user_dset)
                accs_from_this_user.append(acc)
                preds_from_this_user.append(preds)
            preds_from_users_list.append(np.concatenate(preds_from_this_user))
            accs_from_users_list.append(accs_from_this_user)
        mega_ultra_preds = np.stack(preds_from_users_list)
        debabled_mega_ultra_preds = debable(mega_ultra_preds,'none')
        start_idxs = [sum([len(d) for d,sa in user_dsets[:i]]) for i in range(len(user_dsets)+1)]
        mlp_self_preds = [debabled_mega_ultra_preds[uid][start_idxs[uid]:start_idxs[uid+1]] for uid in range(len(user_dsets))]
        true_mlp_accs = [accuracy(p,numpyify(d.y)) for p,(d,sa) in zip(mlp_self_preds,user_dsets)]
        true_f1s = [mean_f1(p,numpyify(d.y)) for p,(d,sa) in zip(mlp_self_preds,user_dsets)]
        hmm_self_preds = [translate_labellings(sa,ta) for sa,ta in zip(self_preds,mlp_self_preds)]
        true_hmm_accs = [accuracy(p,numpyify(d.y)) for p,(d,sa) in zip(hmm_self_preds,user_dsets)]
        total_num_dpoints = sum(len(ud) for ud,sa in user_dsets)
        check_dir(f'experiments/{args.exp_name}')
        with open(f'experiments/{args.exp_name}/results.txt','w') as f:
            for n,acc_list in zip(('reflexive','mlp','hmm'),(self_accs,true_mlp_accs,true_hmm_accs)):
                avg_acc = sum([a*len(ud) for a, (ud, sa) in zip(acc_list, user_dsets)])/total_num_dpoints
                print(f'Acc {n}: {round(avg_acc,5)}')
                f.write(f'Acc {n}: {round(avg_acc,5)}\n')
            for n,f1_list in zip(('reflexive','legit'),(self_f1s,true_f1s)):
                avg_f1 = sum([a*len(ud) for a, (ud, sa) in zip(f1_list, user_dsets)])/total_num_dpoints
                print(f'F1 {n}: {round(avg_f1,5)}')
                f.write(f'F1 {n}: {round(avg_f1,5)}\n')
            f.write('\nAll mlp_accs\n')
            f.write(' '.join([str(a) for a in true_mlp_accs])+'\n')
            f.write('\nAll hmm_accs\n')
            f.write(' '.join([str(a) for a in true_hmm_accs])+'\n')
            f.write('All f1s\n')
            f.write(' '.join([str(f) for f in true_f1s]))
            for relevant_arg in cl_args.RELEVANT_ARGS:
                f.write(f"\n{relevant_arg}: {vars(ARGS).get(relevant_arg)}")


def main(args):
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

        x_filters1 = (50,40,7,4)
        y_filters1 = (5,3,2,1)
        x_strides1 = (2,2,1,1)
        y_strides1 = (1,1,1,1)
        max_pools1 = (2,2,2,2)
        x_filters_trans1 = (45,35,10,6)
        y_filters_trans1 = (8,8,6,6)
        x_strides_trans1 = (1,2,2,2)
        y_strides_trans1 = (1,1,2,1)
        #x_filters1 = (20,20,12,7)
        #y_filters1 = (1,1,1,1)
        #x_strides1 = (1,1,1,1)
        #y_strides1 = (1,1,1,1)
        #max_pools1 = ((2,1),(2,1),1,1)
        #x_filters_trans1 = (20,20,9,7)
        #y_filters_trans1 = (1,1,1,1)
        #x_strides_trans1 = (2,2,2,1)
        #y_strides_trans1 = (1,1,1,1)
        true_num_classes = 12
        num_sensors = 39
    elif args.dset == 'UCI':
        x_filters = (60,40,4,4)
        x_strides = (2,2,1,1)
        y_filters = (5,4,2,2)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        x_strides_trans = (1,3,2,2)
        y_filters_trans = (3,3,3,4)
        y_strides_trans = (1,2,1,1)
        true_num_classes = 6
    elif args.dset == 'WISDM-v1':
        x_filters = (50,40,5,4)
        x_strides = (2,2,1,1)
        y_filters = (1,1,3,2)
        y_strides = (1,1,3,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        x_strides_trans = (1,3,2,2)
        y_filters_trans = (2,2,2,2)
        y_strides_trans = (2,2,1,1)
        true_num_classes = 5
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
        true_num_classes = 17
    elif args.dset == 'Capture24':
        x_filters = (50,40,5,4)
        y_filters = (1,1,2,2)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        y_filters_trans = (2,2,1,1)
        x_strides_trans = (1,3,2,2)
        y_strides_trans = (1,1,1,1)
        true_num_classes = 10
    num_classes = args.num_classes if args.num_classes != -1 else true_num_classes
    enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,args.nf1,show_shapes=args.show_shapes)
    #dec = DecByLayer(x_filters_trans,y_filters_trans,x_strides_trans,y_strides_trans,show_shapes=args.show_shapes)
    mlp = Var_BS_MLP(args.nf1*8,25,num_classes)
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        mlp.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    mlp.cuda()
    subj_ids = args.subj_ids
    dset_train, dset_val, selected_acts = make_dset_train_val(args,subj_ids,train_only=False)
    if args.show_shapes:
        dl = data.DataLoader(dset_train,batch_sampler=data.BatchSampler(data.RandomSampler(dset_train),args.batch_size,drop_last=False),pin_memory=False)
        x_time_trial_run, x_freq_trial_run, _, _ = next(iter(dl))
        #lat = enc(torch.ones((2,1,512,num_ftrs_time),device='cuda'))
        #lat = enc(x_time_trial_run)
        #dec(lat)
        sys.exit()

    #har = HARLearner(enc=enc,mlp=mlp,dec=dec,temp_prox_mlp=temp_prox_mlp,batch_size=args.batch_size,temp_prox_batch_size=args.temp_prox_batch_size,num_classes=num_classes)
    har = HARLearner(enc=enc,mlp=mlp,batch_size=args.batch_size,temp_prox_batch_size=args.temp_prox_batch_size,num_classes=num_classes)

    train_start_time = time.time()
    dsets_by_id = make_dsets_by_user(args,subj_ids)
    bad_ids = []
    for user_id, (dset,sa) in dsets_by_id.items():
        n = get_num_labels(dset.y)
        if n < true_num_classes/2:
            print(f"Excluding user {user_id}, only has {n} different labels, instead of {num_classes}")
            bad_ids.append(user_id)
    dsets_by_id = [v for k,v in dsets_by_id.items() if k not in bad_ids]
    if args.train_type == 'simple':
        accs, nmis, rand_idxs = [], [], []
        for user_id, (dset,sa) in enumerate(dsets_by_id):
            print("clustering", user_id)
            har.pseudo_label_cluster_meta_loop(dset,'none',args.num_meta_epochs,args.num_pseudo_label_epochs,selected_acts)
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
        print("CLUSTERING EACH DSET SEPARATELY")
        accs, nmis, rand_idxs = [], [], []
        for user_id, (dset,sa) in enumerate(dsets_by_id):
            print("clustering", user_id)
            acc, nmi, rand_idx = har.pseudo_label_cluster_meta_meta_loop(dset,args.num_meta_meta_epochs,args.num_meta_epochs,args.num_pseudo_label_epochs,args.prob_thresh,selected_acts)
            accs.append(acc); nmis.append(nmi); rand_idxs.append(rand_idx)
        for t in zip(accs,nmis,rand_idxs):
            print(t)
        print(f"{sum(accs)/len(accs)}\tNMIs: {sum(nmis)/len(nmis)}\tRAND IDXs: {sum(rand_idxs)/len(rand_idxs)}")
    train_end_time = time.time()
    total_prep_time = asMinutes(train_start_time-prep_start_time)
    total_train_time = asMinutes(train_end_time-train_start_time)
    print(f"Prep time: {total_prep_time}\tTrain time: {total_train_time}")


if __name__ == "__main__":

    ARGS = cl_args.get_cl_args()
    main(ARGS)
