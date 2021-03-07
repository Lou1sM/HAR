from pprint import pprint
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
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import clustering.src.utils as utils
import umap


def display(latents_to_display):
    umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,n_components=2,random_state=42).fit_transform(latents_to_display.squeeze())
    utils.scatter_clusters(umapped_latents,labels=None,show=True)


class StepDataset(data.Dataset):
    def __init__(self,x,y,device,window_size,step_size,pl_training,transforms=[]):
        self.device=device
        self.x, self.y = x,y
        self.window_size = window_size
        self.step_size = step_size
        self.pl_training = pl_training
        for transform in transforms:
            self.x = transform(self.x)
        self.x, self.y = self.x.to(self.device),self.y.to(self.device)
    def __len__(self): return (len(self.x)-self.window_size)//self.step_size + 1
    def __getitem__(self,idx):
        batch_x = self.x[idx*self.step_size:(idx*self.step_size) + self.window_size].unsqueeze(0)
        #if self.pl_training:
        batch_y = self.y[idx]
        return batch_x, batch_y, idx
        #else:
            #batch_y = self.y[idx*self.step_size:(idx*self.step_size) + self.window_size]
            #return batch_x, batch_y, idx

    def temporal_consistency_loss(self,sequence):
        total_loss = 0
        for start_idx in range(len(sequence)-self.window_size):
            window = sequence[start_idx:start_idx+self.window_size]
            mu = window.mean(axis=0)
            window_var = sum([(item-mu) for item in self.window])/self.window_size
            if window_var < self.split_thresh: total_loss += window_var
        return total_loss

class HARLearner():
    def __init__(self,dset,enc,dec,mlp,device,batch_size,num_classes):
        self.dset = dset
        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.enc = enc.to(device)
        self.dec = dec.to(device)
        self.mlp = mlp.to(device)
        self.pseudo_label_lf = nn.CrossEntropyLoss()
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
                latent = utils.noiseify(latent,ARGS.noise)
                pred = self.dec(latent)
                loss = self.rec_lf(pred,xb)
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                epoch_loss += (loss.item()-epoch_loss)/(idx+1)
                if ARGS.test: break
            if ARGS.test: break
            print(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
        torch.save(self.enc.state_dict(),'enc_pretrained.pt')
        torch.save(self.dec.state_dict(),'dec_pretrained.pt')

    def pseudo_label_train(self,mask,pseudo_labels,num_epochs,writer,pseudo_labels2='none',position_in_meta_loop=0):
        self.enc.train()
        if isinstance(pseudo_labels,np.ndarray):
            pseudo_labels = torch.tensor(pseudo_labels)
        if isinstance(pseudo_labels2,np.ndarray):
            pseudo_labels2 = torch.tensor(pseudo_labels2)
        pseudo_label_dset = StepDataset(self.dset.x,pseudo_labels,device='cuda',window_size=self.dset.window_size,step_size=self.dset.step_size, pl_training=False)
        pseudo_label_dl = data.DataLoader(pseudo_label_dset,batch_sampler=data.BatchSampler(data.RandomSampler(pseudo_label_dset),self.batch_size,drop_last=True),pin_memory=False)
        if pseudo_labels2=='none':
            pseudo_label_dset2 = StepDataset(self.dset.x,pseudo_labels,device='cuda',window_size=self.dset.window_size,step_size=self.dset.step_size, pl_training=False)
            pseudo_label_dl2 = data.DataLoader(pseudo_label_dset,batch_sampler=data.BatchSampler(data.RandomSampler(pseudo_label_dset),self.batch_size,drop_last=True),pin_memory=False)
        else:
            pseudo_label_dset2 = StepDataset(self.dset.x,torch.tensor(pseudo_labels2),device='cuda',window_size=self.dset.window_size,step_size=self.dset.step_size, pl_training=False)
            pseudo_label_dl2 = data.DataLoader(pseudo_label_dset2,batch_sampler=data.BatchSampler(data.RandomSampler(pseudo_label_dset2),self.batch_size,drop_last=True),pin_memory=False)
            assert (pseudo_labels==pseudo_labels2).all()
        to_iter = zip(pseudo_label_dl,pseudo_label_dl2)
        all_pseudo_label_losses = []
        start_indexing_at = position_in_meta_loop*num_epochs*len(pseudo_label_dl)
        for epoch in range(num_epochs):
            epoch_rec_loss = 0
            epoch_pseudo_label_loss = 0
            epoch_loss = 0
            epoch_rec_losses = []
            epoch_pseudo_label_losses = []
            epoch_losses = []
            best_loss = np.inf
            assert (pseudo_label_dset.x==self.dset.x).all()
            for batch_idx, (xb,yb,idx) in enumerate(pseudo_label_dl):
                batch_mask = mask[idx]
                latent = self.enc(xb)
                latent = utils.noiseify(latent,ARGS.noise)
                if batch_mask.any():
                    latents_to_pseudo_label_train = latent[batch_mask]
                    try:
                        pseudo_label_pred = self.mlp(latents_to_pseudo_label_train[:,:,0,0])
                    except ValueError: set_trace()
                    pseudo_label_loss = self.pseudo_label_lf(pseudo_label_pred,yb[batch_mask])
                    if pseudo_labels2 != 'none':
                        try:
                            test_loss = self.pseudo_label_lf(pseudo_label_pred,torch.tensor(pseudo_labels2[idx][batch_mask],device='cuda'))
                        except: set_trace()
                        assert test_loss == pseudo_label_loss
                else:
                    pseudo_label_loss = torch.tensor(0,device=self.device)
                if not batch_mask.all():
                    latents_to_rec_train = latent[~batch_mask]
                    rec_pred = self.dec(latents_to_rec_train)
                    rec_loss = self.rec_lf(rec_pred,xb[~batch_mask])/(~batch_mask).sum()
                else:
                    rec_loss = torch.tensor(0,device=self.device)
                loss = pseudo_label_loss + rec_loss
                if math.isnan(loss): set_trace()
                loss.backward()
                self.enc_opt.step(); self.enc_opt.zero_grad()
                self.dec_opt.step(); self.dec_opt.zero_grad()
                self.mlp_opt.step(); self.mlp_opt.zero_grad()
                total_idx = start_indexing_at + epoch*len(pseudo_label_dl) + batch_idx
                if batch_mask.any():
                    #epoch_pseudo_label_loss += pseudo_label_loss.item()/batch_mask.sum()
                    writer.add_scalar('Loss/pseudo_label_loss',pseudo_label_loss.item()/batch_mask.sum(),total_idx)
                    epoch_pseudo_label_losses.append(pseudo_label_loss.item()/batch_mask.sum())
                    all_pseudo_label_losses.append(pseudo_label_loss)
                if not batch_mask.all():
                    epoch_loss += (loss.item()-epoch_loss)/(batch_idx+1)
                    writer.add_scalar('Loss/rec_loss',rec_loss.item()/(~batch_mask).sum(),total_idx)
                    epoch_rec_losses.append(rec_loss)
                epoch_losses.append(loss)
                if ARGS.test: break
            if ARGS.test: break
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                count = 0
            else:
                count += 1
            if count > 4: break
            #print(f'Rec Loss: {epoch_rec_loss}\tPseudo_label loss: {epoch_pseudo_label_loss}\t Total:{epoch_loss}')
            assert (pseudo_label_dset.x==self.dset.x).all()
            #plt.plot(epoch_pseudo_label_losses, label='pseudo_label')
            #plt.plot(epoch_rec_losses, label='rec')
            #plt.plot(epoch_losses, label='total')
            #plt.legend()
            #plt.show(block=False)
        return all_pseudo_label_losses

    def train_meta_loop(self,num_pre_epochs,num_meta_epochs,num_pseudo_label_epochs,prob_thresh,selected_acts,exp_dir):
        writer = SummaryWriter()
        self.rec_train(num_pre_epochs)
        old_pred_labels = -np.ones(self.dset.y.shape)
        plt.switch_backend('agg')
        print('modelling')
        for epoch_num in range(num_meta_epochs):
            print('Meta Epoch:', epoch_num)
            if ARGS.test:
                num_tiles = len(self.dset.y)//self.num_classes
                new_pred_labels = np.tile(np.arange(self.num_classes),num_tiles).astype(np.long)
                additional = len(self.dset.y) - (num_tiles*self.num_classes)
                if additional > 0:
                    new_pred_labels = np.concatenate((new_pred_labels,np.ones(additional)))
                new_pred_labels = new_pred_labels.astype(np.long)
                mask = torch.ones(len(self.dset.y)).bool()
                old_pred_labels = new_pred_labels
            else:
                latents = self.get_latents()
                print('umapping')
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
                mask = new_pred_probs.max(axis=1) >= prob_thresh
                fig = plt.figure()
                utils.scatter_clusters(umapped_latents,self.dset.y,show=False)
                writer.add_figure(f'umapped_latents/{epoch_num}',fig)
                print('Prob_thresh mask:',sum(mask),sum(mask)/len(new_pred_labels))
                if ARGS.save: np.save('test_umapped_latents.npy',umapped_latents)
                new_pred_labels = utils.translate_labellings(new_pred_labels,self.dset.y,subsample_size=30000)
            if epoch_num > 0:
                mask2 = new_pred_labels==old_pred_labels
                print('Sames:', sum(mask2), sum(mask2)/len(new_pred_labels))
                mask = mask*mask2
                assert (new_pred_labels[mask]==old_pred_labels[mask]).all()
                self.pseudo_label_train(mask=mask,pseudo_labels2=old_pred_labels,pseudo_labels=new_pred_labels,num_epochs=num_pseudo_label_epochs,writer=writer,position_in_meta_loop=epoch_num)
            else:
                self.pseudo_label_train(mask=mask,pseudo_labels=new_pred_labels,num_epochs=num_pseudo_label_epochs,writer=writer)
            print('translating labelling')
            print('pseudo label training')
            counts = {selected_acts[item]:sum(new_pred_labels==item) for item in set(new_pred_labels)}
            mask_counts = {selected_acts[item]:sum(new_pred_labels[mask]==item) for item in set(new_pred_labels[mask])}
            print('Counts:',counts)
            print('Masked Counts:',mask_counts)
            print('Latent accuracy:', utils.accuracy(new_pred_labels,self.dset.y))
            print('Masked Latent accuracy:', utils.accuracy(new_pred_labels[mask],self.dset.y[mask]),mask.sum())
            rand_idxs = np.array([15,1777,1982,9834,11243,25,7777,5982,5834,41203,250,7717,5912,5134,41843])
            np_gt_labels = self.dset.y.detach().cpu().numpy().astype(int)
            for action_num in np.unique(np_gt_labels):
                action_preds = new_pred_labels[np_gt_labels==action_num]
                action_name = selected_acts[action_num]
                num_correct = (action_preds==action_num).sum()
                total_num = len(action_preds)
                print(f"{action_name}: {num_correct/total_num} ({num_correct}/{total_num})")
            print('GT:',self.dset.y[rand_idxs].int().tolist())
            print('Old:',old_pred_labels[rand_idxs])
            print('New:',new_pred_labels[rand_idxs])
            old_pred_labels = copy.deepcopy(new_pred_labels)
        # Save models
        if ARGS.save:
            with open('hmm_model.pkl', 'wb') as f: pickle.dump(model,f)
            utils.torch_save({'enc':self.enc,'dec':self.dec,'mlp':self.mlp},exp_dir,'har_learner.pt')


class Enc(nn.Module):
    def __init__(self):
        super(Enc,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,4,(50,5),(2,1)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4,8,(40,3),(2,1)),
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(3),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(8,16,(5,2),(1,1)),
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(16,32,(3,1),(1,1)),
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

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


def train(args):
    # Make dataset
    subj_ids = [101,102,103,104,105,106,107,108,109] if args.all_subjs else [101]

    action_name_dict = {1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'Nordic walking',9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing soccer',24:'rope jumping'}
    x = np.concatenate([np.load(f'PAMAP2_Dataset/np_data/subject{subj_id}.npy') for subj_id in subj_ids])
    y = np.concatenate([np.load(f'PAMAP2_Dataset/np_data/subject{subj_id}_labels.npy') for subj_id in subj_ids])
    x = x[y!=0]
    y = y[y!=0]
    xnans = np.isnan(x).any(axis=1)
    x = x[~xnans]
    y = y[~xnans]
    selected_ids = set(y)
    selected_acts = [action_name_dict[act_id] for act_id in selected_ids]
    num_windows = (len(x) - args.window_size)//args.step_size + 1
    mode_labels = np.concatenate([stats.mode(y[w*args.step_size:w*args.step_size + args.window_size]).mode for w in range(num_windows)])
    mode_labels = utils.compress_labels(mode_labels)
    assert len(selected_acts) == len(set(mode_labels))
    pprint(list(zip(selected_ids,selected_acts)))
    num_labels = len(set(mode_labels))
    x = torch.tensor(x,device='cuda').float()
    y = torch.tensor(mode_labels,device='cuda').float()
    dset = StepDataset(x,y,device='cuda',window_size=args.window_size,step_size=args.step_size, pl_training=False)

    # Make nets
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

    mlp = nn.Sequential(
        nn.Linear(32,25),
        nn.BatchNorm1d(25),
        nn.LeakyReLU(0.3),
        nn.Linear(25,num_labels),
        nn.Softmax()
        )

    mlp = Var_BS_MLP(32,25,num_labels)
    enc = Enc()
    if args.load_pretrained:
        enc.load_state_dict(torch.load('enc_pretrained.pt'))
        dec.load_state_dict(torch.load('dec_pretrained.pt'))
    enc.cuda()
    dec.cuda()
    mlp.cuda()

    har = HARLearner(enc=enc,dec=dec,mlp=mlp,dset=dset,device='cuda',batch_size=args.batch_size,num_classes=num_labels)
    exp_dir = os.path.join(f'experiments/{args.exp_name}')

    har.train_meta_loop(num_pre_epochs=args.num_pre_epochs, num_meta_epochs=args.num_meta_epochs, num_pseudo_label_epochs=args.num_pseudo_label_epochs, prob_thresh=args.prob_thresh,selected_acts=selected_acts,exp_dir=exp_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--exp_name',type=str)
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--noise',type=float,default=1.)
    parser.add_argument('--num_meta_epochs',type=int,default=30)
    parser.add_argument('--num_pre_epochs',type=int,default=5)
    parser.add_argument('--num_pseudo_label_epochs',type=int,default=5)
    parser.add_argument('--prob_thresh',type=float,default=.95)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.test and ARGS.save:
        print("Shouldn't be saving for a test run"); sys.exit()
    if ARGS.test: ARGS.num_meta_epochs = 2
    train(ARGS)
