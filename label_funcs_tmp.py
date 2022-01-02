import numpy as np
import torch
import warnings
from pdb import set_trace
from scipy.optimize import linear_sum_assignment
from scipy import stats
from dl_utils.tensor_funcs import numpyify, recursive_np_or


class TranslationError(Exception):
    pass
def unique_labels(labels):
    if isinstance(labels,np.ndarray) or isinstance(labels,list):
        return set(labels)
    elif isinstance(labels,torch.Tensor):
        unique_tensor = labels.unique()
        return set(unique_tensor.tolist())
    else:
        print("Unrecognized type for labels:", type(labels))
        raise TypeError

def label_assignment_cost(labels1,labels2,label1,label2):
    if len(labels1) != len(labels2):
        raise TranslationError(f"len labels1 {len(labels1)} must equal len labels2 {len(labels2)}")
    return len([idx for idx in range(len(labels2)) if labels1[idx]==label1 and labels2[idx] != label2])

def translate_labellings(from_labels,to_labels,subsample_size='none',preserve_sizes=False):
    if from_labels.shape != to_labels.shape:
        raise TranslationError(f"to_labels: {to_labels.shape} doesn't equal to_labels shape: {from_labels.shape}")
    if len(from_labels) == 0:
        warnings.warn("You're translating an empty labelling")
        return from_labels
    trans_dict, leftovers = get_trans_dict(from_labels,to_labels,subsample_size,preserve_sizes)
    return np.array([trans_dict[l] for l in from_labels])

def get_trans_dict(from_labels,to_labels,subsample_size='none',preserve_sizes=False):
    # First some checks
    if from_labels.shape != to_labels.shape:
        raise TranslationError(f"from_labels: {from_labels.shape} doesn't equal to_labels shape: {to_labels.shape}")
    num_from_labs = get_num_labels(from_labels)
    num_to_labs = get_num_labels(to_labels)
    if subsample_size != 'none' and subsample_size < max(num_from_labs,num_to_labs):
        raise TranslationError(f"subsample_size is too small, it must be at least min of the number of different from labels and the number of different to labels, which in this case are {num_from_labs} and {num_to_labs}")
        subsample_size = min(len(from_labels),subsample_size)

    # Compress each labelling, retain compression dicts for decompression later
    from_labels_compressed, tdf, _ = compress_labels(from_labels)
    to_labels_compressed, tdt, _ = compress_labels(to_labels)
    reverse_tdf = {v:k for k,v in tdf.items()}
    reverse_tdt = {v:k for k,v in tdt.items()}
    if num_from_labs <= num_to_labs:
        trans_dict = get_fanout_trans_dict(from_labels_compressed,to_labels_compressed,subsample_size)
        leftovers = set([x for x in unique_labels(to_labels_compressed) if x not in trans_dict.values()])
    else:
        trans_dict,leftovers = get_fanin_trans_dict(from_labels_compressed,to_labels_compressed,subsample_size,preserve_sizes)
    if not len(trans_dict) + len(leftovers) == max(num_from_labs,num_to_labs): set_trace()
    # Account for the possible changes in the above compression
    trans_dict = {reverse_tdf[k]:reverse_tdt[v] for k,v in trans_dict.items()}
    if preserve_sizes and num_from_labs > num_to_labs:
        for i in leftovers:
            tl = reverse_tdf[i]
            assert tl not in trans_dict.keys()
            missing_target = min([i for i in range(num_from_labs) if i not in trans_dict.values()])
            trans_dict[tl]=missing_target
        if not set(trans_dict.keys()) == unique_labels(from_labels): set_trace()

    trans_dict[-1] = -1
    return trans_dict,leftovers

def get_fanout_trans_dict(from_labels,to_labels,subsample_size):
    unique_from_labels = [l for l in unique_labels(from_labels) if l != -1]
    unique_to_labels = [l for l in unique_labels(to_labels) if l != -1]
    if subsample_size == 'none':
        cost_matrix = np.array([[label_assignment_cost(from_labels,to_labels,l1,l2) for l2 in unique_to_labels] for l1 in unique_from_labels])
    else:
        num_trys = 0
        while True:
            num_trys += 1
            if num_trys == 5: set_trace()
            if subsample_size > len(from_labels):
                from_labels_subsample,to_labels_subsample = from_labels,to_labels
            else:
                sample_indices = np.random.choice(range(from_labels.shape[0]),subsample_size,replace=False)
                from_labels_subsample = from_labels[sample_indices]
                to_labels_subsample = to_labels[sample_indices]
            if unique_labels(from_labels_subsample) == set(unique_from_labels) and unique_labels(to_labels_subsample) == set(unique_to_labels): break
        cost_matrix = np.array([[label_assignment_cost(from_labels_subsample,to_labels_subsample,l1,l2) for l2 in unique_from_labels] for l1 in unique_from_labels])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    if len(col_ind) != len(set(from_labels[from_labels != -1])):
        raise TranslationError(f"then translation cost matrix is the wrong size for some reason")
    trans_dict = {l:col_ind[l] for l in unique_labels(from_labels)}
    return trans_dict

def get_fanin_trans_dict(from_labels,to_labels,subsample_size,preserve_sizes):
    unique_from_labels = [l for l in unique_labels(from_labels) if l != -1]
    unique_to_labels = [l for l in unique_labels(to_labels) if l != -1]
    if subsample_size == 'none':
        cost_matrix = np.array([[label_assignment_cost(to_labels,from_labels,l1,l2) for l2 in unique_from_labels] for l1 in unique_to_labels])
    else:
        while True: # Keep trying random indices unitl you reach one that contains all labels
            if subsample_size > len(from_labels):
                from_labels_subsample,to_labels_subsample = from_labels,to_labels
            else:
                sample_indices = np.random.choice(range(from_labels.shape[0]),subsample_size,replace=False)
                from_labels_subsample = from_labels[sample_indices]
                to_labels_subsample = to_labels[sample_indices]
            if [l for l in unique_labels(from_labels_subsample) if l != -1] == unique_from_labels and [l for l in unique_labels(to_labels_subsample) if l != -1] == unique_to_labels: break
        cost_matrix = np.array([[label_assignment_cost(to_labels_subsample,from_labels_subsample,l1,l2) for l2 in unique_from_labels] for l1 in unique_to_labels])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    if len(col_ind) != get_num_labels(to_labels):
        raise TranslationError(f"then translation cost matrix is the wrong size for some reason")
    cl = col_ind.tolist()
    trans_dict = {f:cl.index(f) for f in cl}
    num_trys = 0
    while True:
        num_trys += 1
        if num_trys == 5: set_trace()
        untranslated = [i for i in unique_from_labels if i not in trans_dict.keys()]
        unique_untranslated = unique_labels(untranslated)
        if len(untranslated) == 0 or preserve_sizes: break
        # Now assign the additional, unassigned items
        cost_matrix2 = np.array([[label_assignment_cost(from_labels,to_labels,l1,l2) for l2 in unique_to_labels] for l1 in unique_untranslated])
        row_ind2, col_ind2 = linear_sum_assignment(cost_matrix2)
        for u,t in zip(untranslated,col_ind2): trans_dict[u]=t
    return trans_dict, unique_untranslated

def get_confusion_mat(labels1,labels2):
    if max(labels1) != max(labels2):
        print('Different numbers of clusters, no point trying'); return
    trans_labels = translate_labellings(labels1,labels2)
    num_labels = max(labels1)+1
    confusion_matrix = np.array([[len([idx for idx in range(len(labels2)) if labels1[idx]==l1 and labels2[idx]==l2]) for l2 in range(num_labels)] for l1 in range(num_labels)])
    confusion_matrix = confusion_matrix[:,trans_labels]
    idx = np.arange(num_labels)
    confusion_matrix[idx,idx]=0
    return confusion_matrix

def debable(labellings,pivot,subsample_size='none'):
    #labellings_list.sort(key=lambda x: x.max())
    if isinstance(labellings,np.ndarray):
        if labellings.ndim != 2:
            raise TranslationError(f"If debabling array, it should have 2 dimensions, but here it has {labellings.ndim}")
        labellings_list = [r for r in labellings]
    else:
        labellings_list = labellings
    if pivot == 'none':
        pivot = labellings_list.pop(0)
        translated_list = [pivot]
    else:
        translated_list = []
    for not_lar in labellings_list:
        not_lar_translated = translate_labellings(not_lar,pivot,subsample_size=subsample_size)
        translated_list.append(not_lar_translated)
    return translated_list

def accuracy(preds,targets,subsample_size='none',precision=4):
    if preds.shape != targets.shape:
        raise TranslationError(f"preds shape: {preds.shape} doesn't equal targets shape: {targets.shape}")
    if len(preds) == 0:
        warnings.warn("You're translating an empty labelling")
        return 0
    trans_labels = translate_labellings(preds,targets,subsample_size)
    return round(sum(trans_labels==numpyify(targets))/len(preds),precision)

def f1(bin_classifs_pred,bin_classifs_gt,precision=4):
    tp = sum(bin_classifs_pred*bin_classifs_gt)
    if tp==0: return 0
    fp = sum(bin_classifs_pred*~bin_classifs_gt)
    fn = sum(~bin_classifs_pred*bin_classifs_gt)

    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    return round((2*prec*rec)/(prec+rec),precision)

def mean_f1(labels1,labels2,subsample_size='none',precision=4):
    trans_labels = translate_labellings(labels1,labels2,subsample_size)
    lab_f1s = []
    for lab in unique_labels(trans_labels):
        lab_booleans1 = trans_labels==lab
        lab_booleans2 = labels2==lab
        lab_f1s.append(f1(lab_booleans1,lab_booleans2,precision=15))
    return round(sum(lab_f1s)/len(lab_f1s),precision)

def compress_labels(labels):
    if isinstance(labels,torch.Tensor): labels = labels.detach().cpu().numpy()
    x = sorted([lab for lab in set(labels) if lab != -1])
    trans_dict = {lab:x.index(lab) for lab in set(labels) if lab != -1}
    trans_dict[-1] = -1
    new_labels = np.array([trans_dict[lab] for lab in labels])
    changed = any([k!=v for k,v in trans_dict.items()])
    return new_labels,trans_dict,changed

def get_num_labels(labels):
    assert labels.ndim == 1
    return len([lab for lab in unique_labels(labels) if lab != -1])

def label_counts(labels):
    assert labels.ndim == 1
    return {x:sum(labels==x) for x in unique_labels(labels)}

def dummy_labels(num_classes,size):
    main_chunk = np.tile(np.arange(num_classes),size//num_classes)
    extra_chunk = np.arange(num_classes)[:size%num_classes]
    combined = np.concatenate((main_chunk,extra_chunk), axis=0)
    assert combined.shape[0] == size
    return combined

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

def masked_mode(pred_array,mask='none'):
    if mask is 'none':
        return stats.mode(pred_array,axis=0).mode[0]
    x = mask.any(axis=0)
    return np.array([stats.mode([lab for lab,b in zip(pred_array[:,j],mask[:,j]) if b]).mode[0] if bx else -1 for bx,j in zip(x,range(pred_array.shape[1]))])

