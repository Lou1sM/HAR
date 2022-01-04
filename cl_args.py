import argparse
import sys
import project_config


def get_cl_args():
    dset_options = ['PAMAP','UCI','WISDM-v1','WISDM-watch','REALDISP','Capture24']
    training_type_options = ['full','cluster_as_single','cluster_individually','train_frac_gts_as_single','find_similar_users']
    parser = argparse.ArgumentParser()
    subjs_group = parser.add_mutually_exclusive_group(required=False)
    subjs_group.add_argument('--num_subjs',type=int)
    subjs_group.add_argument('--subj_ids',type=str,nargs='+',default=['first'])
    epochs_group = parser.add_mutually_exclusive_group(required=False)
    epochs_group.add_argument('--full_epochs',action='store_true')
    epochs_group.add_argument('--short_epochs',action='store_true')
    parser.add_argument('--ablate_label_gather',action='store_true')
    parser.add_argument('--all_subjs',action='store_true')
    parser.add_argument('--batch_size_train',type=int,default=256)
    parser.add_argument('--batch_size_val',type=int,default=1024)
    parser.add_argument('--clusterer',type=str,choices=['HMM','GMM'],default='HMM')
    parser.add_argument('--compute_cross_metrics',action='store_true')
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--dset',type=str,default='UCI',choices=dset_options)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--exp_name',type=str,default="try")
    parser.add_argument('--frac_gt_labels',type=float,default=0.1)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--just_align_time',action='store_true')
    parser.add_argument('--load_pretrained',action='store_true')
    parser.add_argument('--mlp_lr',type=float,default=1e-3)
    parser.add_argument('--no_umap',action='store_true')
    parser.add_argument('--noise',type=float,default=1.)
    parser.add_argument('--num_classes',type=int,default=-1)
    parser.add_argument('--num_meta_epochs',type=int,default=1)
    parser.add_argument('--num_meta_meta_epochs',type=int,default=1)
    parser.add_argument('--num_pseudo_label_epochs',type=int,default=5)
    parser.add_argument('--prob_thresh',type=float,default=.95)
    parser.add_argument('--reload_users_so_far',action='store_true')
    parser.add_argument('--rlmbda',type=float,default=.1)
    parser.add_argument('--show_transitions',action='store_true')
    parser.add_argument('--step_size',type=int,default=5)
    parser.add_argument('--subject_independent',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--train_type',type=str,choices=training_type_options,default='full')
    parser.add_argument('--show_shapes',action='store_true',help='print the shapes of hidden layers in enc and dec')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--window_size',type=int,default=512)
    ARGS = parser.parse_args()

    if ARGS.reload_users_so_far and not ARGS.all_subjs:
        print("Can only reload when running on all subjects"); sys.exit()

    need_umap = False
    if ARGS.short_epochs:
        ARGS.num_meta_meta_epochs = 1
        ARGS.num_meta_epochs = 1
        ARGS.num_pseudo_label_epochs = 1
    elif ARGS.full_epochs:
        ARGS.num_meta_meta_epochs = 10
        ARGS.num_meta_epochs = 10
        ARGS.num_pseudo_label_epochs = 5
    if ARGS.test:
        ARGS.num_meta_epochs = 1
        ARGS.num_meta_meta_epochs = 1
        ARGS.num_pseudo_label_epochs = 1
    elif not ARGS.no_umap and not ARGS.show_shapes: need_umap = True
    print(ARGS)
    dset_info_object = project_config.get_dataset_info_object(ARGS.dset)
    all_possible_ids = dset_info_object.possible_subj_ids
    if ARGS.all_subjs: ARGS.subj_ids=all_possible_ids
    elif ARGS.num_subjs is not None: ARGS.subj_ids = all_possible_ids[:ARGS.num_subjs]
    elif ARGS.subj_ids == ['first']: ARGS.subj_ids = all_possible_ids[:1]
    bad_ids = [x for x in ARGS.subj_ids if x not in all_possible_ids]
    if len(bad_ids) > 0:
        print(f"You have specified non-existent ids: {bad_ids}\nExistent ids are {all_possible_ids}"); sys.exit()
    return ARGS, need_umap

RELEVANT_ARGS = ['clusterer','dset','no_umap','num_meta_epochs','num_meta_meta_epochs','num_pseudo_label_epochs','step_size','subject_independent']
