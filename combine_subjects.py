import pickle
from hmmlearn import hmm
from clustering.src import utils
import numpy as np
import os
from pdb import set_trace


def complete_trans_dict(trans_dict,to_labels,total_from_labels):
    """Fill in missing targets in case of not all having been translated to."""
    untranslated = sorted([x for x in utils.unique_labels(total_from_labels) if x not in trans_dict.keys()])
    untranslated_to = sorted([x for x in utils.unique_labels(to_labels) if x not in trans_dict.values()])
    #assert len(untranslated_to) == len(untranslated)
    # assign randomly
    for k,v in zip(untranslated,untranslated_to):
        print(f"adding {k} {v}")
        trans_dict[k] = v


def translate_subj_predictions(trans_from_id,trans_to_id,exp_dir):
    trans_from_dir = os.path.join(exp_dir,f'jim{trans_from_id}')
    trans_to_dir = os.path.join(exp_dir,f'jim{trans_to_id}')
    trans_from_latents = np.load(os.path.join(trans_from_dir,f'umapped_latentsjim{trans_from_id}.npy'))
    trans_to_latents = np.load(os.path.join(trans_to_dir,f'umapped_latentsjim{trans_to_id}.npy'))
    concat_latents = np.concatenate((trans_from_latents,trans_to_latents))
    trans_from_orig_preds = np.load(os.path.join(trans_from_dir,f'predsjim{trans_from_id}.npy'))
    trans_to_orig_preds = np.load(os.path.join(trans_to_dir,f'predsjim{trans_to_id}.npy'))
    orig_preds = np.concatenate((trans_from_orig_preds,trans_to_orig_preds))
    #try: assert utils.get_num_labels(trans_to_orig_preds) == utils.get_num_labels(trans_from_orig_preds)
    #except: set_trace()
    num_classes = utils.get_num_labels(trans_to_orig_preds)
    model = hmm.GaussianHMM(num_classes,'full')
    model.params = 'mc'
    model.init_params = 'mc'
    model.startprob_ = np.ones(num_classes)/num_classes
    num_action_blocks = len([item for idx,item in enumerate(orig_preds) if orig_preds[idx-1] != item])
    prob_new_action = num_action_blocks/len(orig_preds)
    model.transmat_ = (np.eye(num_classes) * (1-prob_new_action)) + (np.ones((num_classes,num_classes))*prob_new_action/num_classes)
    model.fit(concat_latents)
    # Sometimes not all activities are represented in the from (or to) half of
    # new_preds, atm just translating missing ones randomly, but this hurts acc
    new_preds = model.predict(concat_latents)
    subsample_size = min(30000,len(orig_preds))
    print(utils.get_label_counts(new_preds))
    c_new_preds1 = new_preds[:len(trans_from_orig_preds)]
    c_new_preds2 = new_preds[len(trans_from_orig_preds):]
    print(utils.get_label_counts(c_new_preds1))
    print(utils.get_label_counts(c_new_preds2))
    print(utils.get_label_counts(trans_from_orig_preds))
    print(utils.get_label_counts(trans_to_orig_preds))
    trans_dict1,leftovers1 = utils.get_trans_dict(trans_from_orig_preds,c_new_preds1,'none')
    trans_dict2,leftovers2 = utils.get_trans_dict(c_new_preds2,trans_to_orig_preds,subsample_size='none')
    complete_trans_dict(trans_dict1,total_from_labels=list(range(max(new_preds)+1)),to_labels=list(range(max(new_preds)+1)))
    complete_trans_dict(trans_dict2,total_from_labels=list(range(max(new_preds)+1)),to_labels=list(range(max(new_preds)+1)))
    trans_preds = np.array([trans_dict2[trans_dict1[item]] for item in trans_from_orig_preds])
    return trans_preds

pivot_id = 101
exp_dir = f'experiments/'
for subj_id in [102,103,104,105,106,107,108,109]:
    print(subj_id)
    trans_preds = translate_subj_predictions(subj_id,pivot_id,exp_dir)
    utils.np_save(trans_preds,exp_dir,f'trans_preds{subj_id}.npy')
