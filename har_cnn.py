import sys
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

rari = lambda x,y: round(adjusted_rand_score(x,y),4)
rnmi = lambda x,y: round(normalized_mutual_info_score(x,y),4)

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
                #if len(xb) == 1: continue # If last batch is only one element then batchnorm will error
                latent = self.enc(xb)
                if noise > 0: latent = noiseify(latent,noise)
                label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent.squeeze(2).squeeze(2))
                batch_mask = 'none' if not is_mask  else multiplicative_mask[:self.batch_size] if ARGS.test else multiplicative_mask[idx]
                loss = lf(label_pred,yb.long(),batch_mask)
                if math.isnan(loss): set_trace()
                if rlmbda>0:
                    rec_loss = self.rec_lf(self.dec(latent),xb).mean()
                    loss += rlmbda*rec_loss
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
                try:
                    acc = -1 if ARGS.test else accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
                    f1 = -1 if ARGS.test else mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
                except: set_trace()
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

    def val_on(self,dset,test):
        self.enc.eval()
        self.dec.eval()
        self.mlp.eval()
        pred_list = []
        idx_list = []
        #dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),self.batch_size,drop_last=False),pin_memory=False)
        bs = min(8192,len(dset))
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),bs,drop_last=False),pin_memory=False)
        print(len(dset))
        for batch_idx, (xb,yb,idx) in enumerate(dl):
            latent = self.enc(xb)
            label_pred = self.mlp(latent) if latent.ndim == 2 else self.mlp(latent[:,:,0,0])
            pred_list.append(label_pred.argmax(axis=1).detach().cpu().numpy())
            #idx_list.append(idx.detach().cpu().numpy())
            if ARGS.test: break
        pred_array = np.concatenate(pred_list)
        if test:
            return -1, -1, dummy_labels(self.num_classes,len(dset.y))
        #idx_array = np.concatenate(idx_list)
        #print(idx_array)
        #pred_array_ordered = np.array([item[0] for item in sorted(zip(pred_array,idx_array),key=lambda x:x[1])])
        #if get_num_labels(dset.y) == 1: set_trace()
        #acc = -1 if ARGS.test else accuracy(pred_array_ordered,dset.y.detach().cpu().numpy())
        #f1 = -1 if ARGS.test else mean_f1(pred_array_ordered,dset.y.detach().cpu().numpy())
        return pred_array

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

    def pseudo_label_cluster_meta_loop(self,dset,meta_pivot_pred_labels,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts):
        old_pred_labels = -np.ones(dset.y.shape)
        np_gt_labels = dset.y.detach().cpu().numpy().astype(int)
        super_mask = np.ones(len(dset)).astype(np.bool)
        mlp_accs = []
        cluster_accs = []
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
                if meta_pivot_pred_labels is not 'none':
                    new_pred_labels = translate_labellings(new_pred_labels,meta_pivot_pred_labels,subsample_size=30000)
                elif epoch_num > 0:
                    new_pred_labels = translate_labellings(new_pred_labels,old_pred_labels,subsample_size=30000)
            pseudo_label_dset = deepcopy(dset)
            pseudo_label_dset.y = cudify(new_pred_labels)
            if epoch_num > 0:
                mask2 = new_pred_labels==old_pred_labels
                mask = mask*mask2
                assert (new_pred_labels[mask]==old_pred_labels[mask]).all()
            super_mask*=mask
            mask_to_use = (mask+super_mask)/2
            mlp_acc,mlp_f1,mlp_preds,mlp_confs = self.train_on(pseudo_label_dset,multiplicative_mask=cudify(mask_to_use),num_epochs=num_pseudo_label_epochs)
            cluster_acc = accuracy(new_pred_labels,np_gt_labels)
            cluster_accs.append(cluster_acc)
            mlp_accs.append(mlp_acc)
            y_np = numpyify(dset.y)
            if ARGS.verbose:
                print('Meta Epoch:', epoch_num)
                print('Super Masked Counts:',label_counts(y_np[super_mask]))
                print('Latent accuracy:', cluster_acc)
                print('Masked latent accuracy:', accuracy(new_pred_labels[mask],y_np[mask]),mask.sum())
                print('Super Masked latent accuracy:', accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum())
                print('MLP accuracy:', accuracy(mlp_preds,np_gt_labels))
                print('Masked MLP accuracy:', accuracy(mlp_preds[mask],dset.y[mask]),mask.sum())
                print('Super Masked MLP accuracy:', accuracy(mlp_preds[super_mask],dset.y[super_mask]),super_mask.sum())
            elif epoch_num == num_meta_epochs-1:
                print(f"Latent: {accuracy(new_pred_labels,np_gt_labels)}\tMaskL: {accuracy(new_pred_labels[mask],y_np[mask]),mask.sum()}\tSuperMaskL{accuracy(new_pred_labels[super_mask],dset.y[super_mask]),super_mask.sum()}")
            old_pred_labels = deepcopy(new_pred_labels)
        super_super_mask = np.logical_and(super_mask,new_pred_labels==mlp_preds)
        return new_pred_labels, mask, super_mask, super_super_mask, mlp_accs, cluster_accs

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

        return best_preds_so_far,preds

    def full_train(self,user_dsets_as_dict,args):
        train_start_time = time.time()
        preds_from_users_list = []
        accs_from_users_list = []
        self_accs = []
        self_f1s = []
        self_aris = []
        self_nmis = []
        self_best_accs = []
        self_best_f1s = []
        self_best_aris = []
        self_best_nmis = []
        self_preds = []
        self_best_preds = []
        total_align_time = 0
        user_dsets = list(user_dsets_as_dict.values())
        for user_id, (user_dset, sa) in user_dsets_as_dict.items():
            preds_from_this_user = []
            accs_from_this_user = []
            if not ARGS.just_align_time:
                print(f"training on {user_id}")
                best_preds,preds = self.pseudo_label_cluster_meta_meta_loop(user_dset,num_meta_meta_epochs=args.num_meta_meta_epochs,num_meta_epochs=args.num_meta_epochs,num_pseudo_label_epochs=args.num_pseudo_label_epochs,prob_thresh=args.prob_thresh,selected_acts=sa)
                self_accs.append(accuracy(preds,numpyify(user_dset.y)))
                self_best_accs.append(accuracy(best_preds,numpyify(user_dset.y)))
                self_f1s.append(mean_f1(preds,numpyify(user_dset.y)))
                self_best_f1s.append(mean_f1(best_preds,numpyify(user_dset.y)))
                self_aris.append(rari(preds,numpyify(user_dset.y)))
                self_best_aris.append(rari(best_preds,numpyify(user_dset.y)))
                self_nmis.append(rnmi(preds,numpyify(user_dset.y)))
                self_best_nmis.append(rnmi(best_preds,numpyify(user_dset.y)))
                self_preds.append(preds)
                self_best_preds.append(best_preds)
            align_start_time = time.time()
            for other_user_id, (other_user_dset, sa) in user_dsets_as_dict.items():
                preds = self.val_on(other_user_dset,test=ARGS.test)
                preds_from_this_user.append(preds)
            preds_from_users_list.append(np.concatenate(preds_from_this_user))
            print([round(a,4) for a in self_accs])
            total_align_time += time.time() - align_start_time
            print(total_align_time,time.time() - align_start_time,time.time(),align_start_time)
        if ARGS.just_align_time:
            print(f'Total align time: {total_align_time}'); sys.exit()
        mega_ultra_preds = np.stack(preds_from_users_list)
        debabled_mega_ultra_preds = debable(mega_ultra_preds,'none')
        start_idxs = [sum([len(d) for d,sa in user_dsets[:i]]) for i in range(len(user_dsets)+1)]
        mlp_self_preds = [debabled_mega_ultra_preds[uid][start_idxs[uid]:start_idxs[uid+1]] for uid in range(len(user_dsets))]
        hmm_self_preds = [translate_labellings(sa,ta) for sa,ta in zip(self_preds,mlp_self_preds)]

        mlp_accs = [accuracy(p,numpyify(d.y)) for p,(d,sa) in zip(mlp_self_preds,user_dsets)]
        mlp_f1s = [mean_f1(p,numpyify(d.y)) for p,(d,sa) in zip(mlp_self_preds,user_dsets)]
        mlp_aris = [rari(p,numpyify(d.y)) for p,(d,sa) in zip(mlp_self_preds,user_dsets)]
        mlp_nmis = [rnmi(p,numpyify(d.y)) for p,(d,sa) in zip(mlp_self_preds,user_dsets)]
        hmm_accs = [accuracy(p,numpyify(d.y)) for p,(d,sa) in zip(hmm_self_preds,user_dsets)]
        hmm_f1s = [mean_f1(p,numpyify(d.y)) for p,(d,sa) in zip(hmm_self_preds,user_dsets)]
        hmm_aris = [rari(p,numpyify(d.y)) for p,(d,sa) in zip(hmm_self_preds,user_dsets)]
        hmm_nmis = [rnmi(p,numpyify(d.y)) for p,(d,sa) in zip(hmm_self_preds,user_dsets)]
        total_num_dpoints = sum(len(ud) for ud,sa in user_dsets)
        check_dir(f'experiments/{args.exp_name}')
        np.save(f'experiments/{args.exp_name}/debabled_mega_ultra_preds',debabled_mega_ultra_preds)
        with open(f'experiments/{args.exp_name}/results.txt','w') as f:
            for n,acc_list in zip(('self','best_self','mlp','hmm'),(self_accs,self_best_accs,mlp_accs,hmm_accs)):
                avg_acc = sum([a*len(ud) for a, (ud, sa) in zip(acc_list, user_dsets)])/total_num_dpoints
                print(f'Acc {n}: {round(avg_acc,5)}')
                f.write(f'Acc {n}: {round(avg_acc,5)}\n')
            for n,f1_list in zip(('self','best_self','mlp','hmm'),(self_f1s,self_best_f1s,mlp_f1s,hmm_f1s)):
                avg_f1 = sum([a*len(ud) for a, (ud, sa) in zip(f1_list, user_dsets)])/total_num_dpoints
                print(f'F1 {n}: {round(avg_f1,5)}')
                f.write(f'F1 {n}: {round(avg_f1,5)}\n')
            for n,ari_list in zip(('self','best_self','mlp','hmm'),(self_aris,self_best_aris,mlp_aris,hmm_aris)):
                avg_ari = sum([a*len(ud) for a, (ud, sa) in zip(ari_list, user_dsets)])/total_num_dpoints
                print(f'ARI {n}: {round(avg_ari,5)}')
                f.write(f'ARI {n}: {round(avg_ari,5)}\n')
            for n,nmi_list in zip(('self','best_self','mlp','hmm'),(self_nmis,self_best_nmis,mlp_nmis,hmm_nmis)):
                avg_nmi = sum([a*len(ud) for a, (ud, sa) in zip(nmi_list, user_dsets)])/total_num_dpoints
                print(f'NMI {n}: {round(avg_nmi,5)}')
                f.write(f'NMI {n}: {round(avg_nmi,5)}\n')
            f.write('\nAll mlp_accs\n')
            f.write(' '.join([str(a) for a in mlp_accs])+'\n')
            f.write('\nAll mlp_f1s\n')
            f.write(' '.join([str(a) for a in mlp_accs])+'\n')
            f.write('\nAll mlp_aris\n')
            f.write(' '.join([str(a) for a in mlp_accs])+'\n')
            f.write('\nAll mlp_nmis\n')
            f.write(' '.join([str(a) for a in mlp_accs])+'\n')

            f.write('\nAll hmm_accs\n')
            f.write(' '.join([str(a) for a in hmm_accs])+'\n')
            f.write('\nAll hmm_f1s\n')
            f.write(' '.join([str(a) for a in hmm_accs])+'\n')
            f.write('\nAll hmm_aris\n')
            f.write(' '.join([str(a) for a in hmm_accs])+'\n')
            f.write('\nAll hmm_nmis\n')
            f.write(' '.join([str(a) for a in hmm_accs])+'\n')
            for relevant_arg in cl_args.RELEVANT_ARGS:
                f.write(f"\n{relevant_arg}: {vars(ARGS).get(relevant_arg)}")
            train_end_time = time.time()
            total_train_time = asMinutes(train_end_time-train_start_time)
            set_trace()
            f.write(f'Total align time: {total_align_time}')
            f.write(f'Total train time: {total_train_time}')
            cross_accs = np.array([[accuracy(debabled_mega_ultra_preds[pred_id][start_idxs[target_id]:start_idxs[target_id+1]],numpyify(user_dsets[target_id][0].y)) for target_id in range(len(user_dsets))] for pred_id in range(len(user_dsets))])
            cross_aris = np.array([[rari(debabled_mega_ultra_preds[pred_id][start_idxs[target_id]:start_idxs[target_id+1]],numpyify(user_dsets[target_id][0].y)) for target_id in range(len(user_dsets))] for pred_id in range(len(user_dsets))])
            cross_nmis = np.array([[rnmi(debabled_mega_ultra_preds[pred_id][start_idxs[target_id]:start_idxs[target_id+1]],numpyify(user_dsets[target_id][0].y)) for target_id in range(len(user_dsets))] for pred_id in range(len(user_dsets))])
            f.write(f'Mean cross acc: {mean_off_diagonal(cross_accs)}')
            f.write(f'Mean cross ari: {mean_off_diagonal(cross_aris)}')
            f.write(f'Mean cross nmi: {mean_off_diagonal(cross_nmis)}')
            f.write(f'Total align time: {total_align_time}')
            f.write(f'Total train time: {total_train_time}')
        np.save(f'experiments/{args.exp_name}/debabled_mega_ultra_preds',debabled_mega_ultra_preds)
        np.save(f'experiments/{args.exp_name}/cross_accs',cross_accs)
        np.save(f'experiments/{args.exp_name}/cross_aris',cross_aris)
        np.save(f'experiments/{args.exp_name}/cross_nmis',cross_nmis)
        np.save(f'experiments/{args.exp_name}/self_accs',self_accs)
        np.save(f'experiments/{args.exp_name}/self_f1s',self_f1s)
        np.save(f'experiments/{args.exp_name}/self_aris',self_aris)
        np.save(f'experiments/{args.exp_name}/self_nmis',self_nmis)
        np.save(f'experiments/{args.exp_name}/self_best_accs',self_best_accs)
        np.save(f'experiments/{args.exp_name}/self_best_f1s',self_best_f1s)
        np.save(f'experiments/{args.exp_name}/self_best_aris',self_best_aris)
        np.save(f'experiments/{args.exp_name}/self_best_nmis',self_best_nmis)
        print(f'Total align time: {total_align_time}')
        print(f'Total train time: {total_train_time}')
        if ARGS.all_subjs:
            dset_name_dir_dict = {'PAMAP': 'PAMAP2_Dataset', 'REALDISP': 'realdisp', 'UCI': 'UCI2', 'WISDM-v1': 'wisdm_v1', 'WISDM-watch': 'wisdm-dataset'}
            np.save(f'datasets/{dset_name_dir_dict[ARGS.dset]}/full_ygt',np.concatenate([numpyify(d.y) for d,sa in user_dsets]))

def mean_off_diagonal(mat):
    upper_sum = np.triu(mat,1).sum()
    lower_sum = np.tril(mat,-1).sum()
    num_el = np.prod(mat.shape) - mat.shape[0]
    return (upper_sum + lower_sum)/num_el

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

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dset == 'PAMAP':
        x_filters = (50,40,7,4)
        y_filters = (1,1,1,39)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = ((2,1),(2,1),(2,1),(2,1))
        x_filters_trans = (45,35,10,6)
        y_filters_trans = (18,18,6,5)
        x_strides_trans = (1,2,2,2)
        y_strides_trans = (1,1,2,1)
        true_num_classes = 12
    elif args.dset == 'UCI':
        x_filters = (60,40,4,4)
        x_strides = (2,2,1,1)
        y_filters = (1,1,1,6)
        y_strides = (1,1,1,1)
        max_pools = ((2,1),(3,1),(2,1),1)
        x_filters_trans = (30,30,20,10)
        x_strides_trans = (1,3,2,2)
        y_filters_trans = (3,3,3,4)
        y_strides_trans = (1,2,1,1)
        true_num_classes = 6
    elif args.dset == 'WISDM-v1':
        x_filters = (50,40,5,4)
        x_strides = (2,2,1,1)
        y_filters = (1,1,1,3)
        y_strides = (1,1,1,1)
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
    elif args.dset == 'REALDISP':
        x_filters = (50,40,8,6)
        y_filters = (1,1,1,117)
        x_strides = (2,2,1,1)
        y_strides = (1,1,1,1)
        max_pools = ((2,1),(3,1),1,1)
        x_filters_trans = (31,30,14,10)
        y_filters_trans = (2,2,2,2)
        x_strides_trans = (1,3,2,2)
        y_strides_trans = (1,1,2,2)
        true_num_classes = 33
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
    enc = EncByLayer(x_filters,y_filters,x_strides,y_strides,max_pools,show_shapes=args.show_shapes)
    dec = DecByLayer(x_filters_trans,y_filters_trans,x_strides_trans,y_strides_trans,show_shapes=args.show_shapes)
    mlp = Var_BS_MLP(32,256,num_classes)
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        mlp.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    dec.cuda()
    mlp.cuda()
    subj_ids = args.subj_ids
    dset_train, dset_val, selected_acts = make_dset_train_val(args,subj_ids)
    if args.show_shapes:
        num_ftrs = dset_train.x.shape[-1]
        print(num_ftrs)
        lat = enc(torch.ones((2,1,512,num_ftrs),device='cuda'))
        dec(lat)
        sys.exit()

    har = HARLearner(enc=enc,mlp=mlp,dec=dec,batch_size=args.batch_size,num_classes=num_classes)

    dsets_by_id = make_dsets_by_user(args,subj_ids)
    bad_ids = []
    for user_id, (dset,sa) in dsets_by_id.items():
        n = get_num_labels(dset.y)
        if n < true_num_classes/2:
            print(f"Excluding user {user_id}, only has {n} different labels, instead of {num_classes}")
            bad_ids.append(user_id)
    dsets_by_id = {k:v for k,v in dsets_by_id.items() if k not in bad_ids}
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
        print("CLUSTERING EACH DSET SEPARATELY")
        accs, nmis, rand_idxs = [], [], []
        for user_id, (dset,sa) in enumerate(dsets_by_id):
            print("clustering", user_id)
            acc, nmi, rand_idx = har.pseudo_label_cluster_meta_meta_loop(dset,args.num_meta_meta_epochs,args.num_meta_epochs,args.num_pseudo_label_epochs,args.prob_thresh,selected_acts)
            accs.append(acc); nmis.append(nmi); rand_idxs.append(rand_idx)
        for t in zip(accs,nmis,rand_idxs):
            print(t)
        print(f"{sum(accs)/len(accs)}\tNMIs: {sum(nmis)/len(nmis)}\tRAND IDXs: {sum(rand_idxs)/len(rand_idxs)}")


if __name__ == "__main__":

    ARGS, need_umap = cl_args.get_cl_args()
    if need_umap: import umap
    main(ARGS)
