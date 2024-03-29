from pdb import set_trace
from collections import Counter
from scipy.fft import fft
from scipy import stats
from mpmath import mp, mpf
#from dl_utils import misc, label_funcs
from dl_utils import misc
import label_funcs_tmp
import os
from os.path import join
import sys
import numpy as np
import pandas as pd
import json

def array_from_txt(inpath):
    with open(inpath) as f:
        d = f.readlines()
        array = np.array([[float(x) for x in line.split()] for line in d])
    return array

def array_expanded(a,expanded_length):
    if a.shape[0] >= expanded_length: return a
    insert_every = mpf(a.shape[0]/(expanded_length-a.shape[0]))
    additional_idxs = (np.arange(expanded_length-a.shape[0])*insert_every).astype(np.int)
    values = a[additional_idxs]
    expanded = np.insert(a,additional_idxs,values,axis=0)
    assert expanded.shape[0] == expanded_length
    return expanded

def convert(inpath,outpath):
    array = array_from_txt(inpath)
    timestamps = array[:,0]
    labels = array[:,1].astype(np.int)
    # Delete orientation data, which webpage says is 'invalid in this data, and timestamp and label
    array = np.delete(array,[0,1,2,13,14,15,16,30,31,32,33,47,48,49,50],1)
    np.save(outpath+'_timestamps',timestamps)
    np.save(outpath+'_labels',labels)
    np.save(outpath,array)

def expand_and_fill_labels(a,propoer_length):
    start_filler = -np.ones(a[0,3])
    end_filler = -np.ones(propoer_length-a[-1,4])
    nested_lists = [[a[i,2] for _ in range(a[i,4]-a[i,3])] + [-1]*(a[i+1,3]-a[i,4]) for i in range(len(a)-1)] + [[a[-1,2] for _ in range(a[-1,4]-a[-1,3])]]
    middle = np.array([item for sublist in nested_lists for item in sublist])
    total_label_array = np.concatenate((start_filler,middle,end_filler)).astype(np.int)
    return total_label_array

def add_dtft(signal):
    fft_signal_complex = fft(signal,axis=-1)
    fft_signal_modulusses = np.abs(fft_signal_complex)
    return np.concatenate((signal,fft_signal_modulusses),axis=-1)

if __name__ == "__main__":
    if sys.argv[1] == 'PAMAP':
        data_dir = 'PAMAP2_Dataset/Protocol'
        np_dir = 'PAMAP2_Dataset/np_data'
        print("\n#####Preprocessing PAMAP2#####\n")
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        for filename in os.listdir(data_dir):
            print(filename)
            inpath = join(data_dir,filename)
            outpath = join(np_dir,filename.split('.')[0])
            convert(inpath,outpath)

    elif sys.argv[1] == 'UCI-raw':
        data_dir = 'UCI2/RawData'
        np_dir = 'UCI2/np_data'
        print("\n#####Preprocessing UCI#####\n")
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        raw_label_array = array_from_txt(join(data_dir,'labels.txt')).astype(int)
        def two_digitify(x): return '0'+str(x) if len(str(x))==1 else str(x)
        fnames = os.listdir(data_dir)
        for idx in range(1,31):
            print("processing user",idx)
            acc_array_list = []
            gyro_array_list = []
            label_array_list = []
            user_idx = two_digitify(idx)
            acc_fpaths = sorted([fn for fn in fnames if f'user{user_idx}' in fn and 'acc' in fn])
            gyro_fpaths = sorted([fn for fn in fnames if f'user{user_idx}' in fn and 'gyro' in fn])
            assert len(acc_fpaths) == len(gyro_fpaths)
            for fna,fng in zip(acc_fpaths,gyro_fpaths):
                acc_exp_id = int(fna.split('exp')[1][:2])
                gyro_exp_id = int(fng.split('exp')[1][:2])
                assert acc_exp_id==gyro_exp_id
                new_acc_array = array_from_txt(join(data_dir,fna))
                new_gyro_array = array_from_txt(join(data_dir,fna))
                label_array_block = raw_label_array[raw_label_array[:,0]==acc_exp_id]
                filled_label_array_block = expand_and_fill_labels(label_array_block,new_acc_array.shape[0])
                assert filled_label_array_block.shape[0] == new_acc_array.shape[0]
                assert filled_label_array_block.shape[0] == new_gyro_array.shape[0]
                label_array_list.append(filled_label_array_block)
                acc_array_list.append(new_acc_array)
                gyro_array_list.append(new_gyro_array)
            label_array = np.concatenate(label_array_list)
            acc_array = np.concatenate(acc_array_list)
            gyro_array = np.concatenate(gyro_array_list)
            total_array = np.concatenate((acc_array,gyro_array),axis=1)
            outpath = join(np_dir,f'user{user_idx}.npy')
            np.save(outpath,total_array)
            label_outpath = join(np_dir,f'user{user_idx}_labels.npy')
            np.save(label_outpath,label_array)

    elif sys.argv[1] == 'WISDM-v1':
        with open('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt') as f: text = f.readlines()
        print("\n#####Preprocessing WISDM-v1#####\n")
        activities_list = ['Jogging','Walking','Upstairs','Downstairs','Standing','Sitting']
        X_list = []
        y_list = []
        users_list = []
        num_zeros = 0
        def process_line(line_to_process):
            global num_zeros
            if float(line_to_process.split(',')[2]) == 0: num_zeros += 1#print("Timestamp zero, discarding")
            else:
                X_list.append([float(x) for x in line_to_process.split(',')[3:]])
                y_list.append(activities_list.index(line_to_process.split(',')[1]))
                users_list.append(line_to_process.split(',')[0])
        for i,raw_line in enumerate(text):
            #line = line.replace(';','').replace('\n','')
            if raw_line == '\n': continue
            elif raw_line.endswith(',;\n'): line = raw_line[:-3]
            elif raw_line.endswith(';\n'): line = raw_line[:-2]
            elif raw_line.endswith(',\n'): line = raw_line[:-2]
            else: set_trace()
            if len(line.split(',')) == 6:
                try: process_line(line)
                except: print(f"Can't process line {i}, even though length 6: {raw_line}\n")
            else:
                print(f"Bad format at line {i}:\n{raw_line}")
                try:
                    line1, line2 = line.split(';')
                    process_line(line1); process_line(line2)
                    print(f"I think this was two lines erroneously put on one line. Processing separately as\n{line1}\nand\n{line2}")
                except: print("Can't process this line at all, omitting")
        one_big_X_array = np.array(X_list)
        one_big_y_array = np.array(y_list)
        one_big_users_array = np.array(users_list)
        print(one_big_X_array.shape)
        print(one_big_y_array.shape)
        print(one_big_users_array.shape)
        print(f"Number of zero lines: {num_zeros}")
        misc.np_save(one_big_X_array,'wisdm_v1','X.npy')
        misc.np_save(one_big_y_array,'wisdm_v1','y.npy')
        misc.np_save(one_big_users_array,'wisdm_v1','users.npy')

    elif sys.argv[1] == 'WISDM-watch':
        p_dir = 'wisdm-dataset/raw/phone'
        w_dir = 'wisdm-dataset/raw/watch'
        np_dir = 'wisdm-dataset/np_data'
        print("\n#####Preprocessing WISDM-watch#####\n")
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        mp.dps = 100 # Avoid floating point errors in label insertion function
        for user_idx in range(1600,1651):
            print('user', user_idx)
            phone_acc_path = join(p_dir,'accel',f'data_{user_idx}_accel_phone.txt')
            watch_acc_path = join(w_dir,'accel',f'data_{user_idx}_accel_watch.txt')
            phone_gyro_path = join(p_dir,'gyro',f'data_{user_idx}_gyro_phone.txt')
            watch_gyro_path = join(w_dir,'gyro',f'data_{user_idx}_gyro_watch.txt')

            label_codes_list = list('ABCDEFGHIJKLMOPQRS') # Missin 'N' is deliberate
            def two_arrays_from_txt(inpath):
                with open(inpath) as f:
                    d = f.readlines()
                    arr = np.array([[float(x) for x in line.strip(';\n').split(',')[3:]] for line in d])
                    label_array = np.array([label_codes_list.index(line.split(',')[1]) for line in d])
                return arr, label_array

            phone_acc, label_array1 = two_arrays_from_txt(phone_acc_path)
            watch_acc, label_array2 = two_arrays_from_txt(watch_acc_path)
            phone_gyro, label_array3 = two_arrays_from_txt(phone_gyro_path)
            watch_gyro, label_array4 = two_arrays_from_txt(watch_gyro_path)
            user_arrays = [phone_acc,watch_acc,phone_gyro,watch_gyro]
            label_arrays = [label_array1,label_array2,label_array3,label_array4]
            max_len = max([a.shape[0] for a in user_arrays])
            equalized_user_arrays = [array_expanded(a,max_len) for a in user_arrays]
            equalized_label_arrays = [array_expanded(lab_a,max_len) for lab_a in label_arrays]
            total_user_array = np.concatenate(equalized_user_arrays,axis=1)
            mode_object = stats.mode(np.stack(equalized_label_arrays,axis=1),axis=1)
            mode_labels = mode_object.mode[:,0]
            # Print how many windows contained just 1 label, how many 2 etc.
            #print('Agreement in labels:',label_funcs_tmp.label_counts(mode_object.count[:,0]))
            certains = (mode_object.count == 4)[:,0]
            user_fn = f'{user_idx}.npy'
            misc.np_save(total_user_array,np_dir,user_fn)
            user_labels_fn = f'{user_idx}_labels.npy'
            misc.np_save(mode_labels,np_dir,user_labels_fn)
            user_certains_fn = f'{user_idx}_certains.npy'
            misc.np_save(certains,np_dir,user_certains_fn)

    elif sys.argv[1] == 'REALDISP':
        data_dir = 'realdisp/RawData'
        np_dir = 'realdisp/np_data'
        print("\n#####Preprocessing REALDISP#####\n")
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        for filename in os.listdir(data_dir):
            print(filename)
            if filename == 'dataset manual.pdf': continue
            if not filename.split('_')[1].startswith('ideal'):
                continue
            with open(join(data_dir,filename)) as f: xy = f.readlines()
            ar = np.array([[float(item) for item in line.split('\t')] for line in xy])
            x = ar[:,:-1]
            y = ar[:,-1].astype(int)

            np.save(join(np_dir,filename.split('_')[0]), x)
            np.save(join(np_dir,filename.split('_')[0])+'_labels', y)

    elif sys.argv[1] == 'Capture24':
        np_dir = 'capture24/np_data'
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)
        name_df = pd.read_csv('capture24/annotation-label-dictionary.csv')
        #name_conversion_dict = dict(zip(name_df['annotation'],name_df['label:DohertySpecific2018']))
        name_df = name_df[['annotation','label:DohertySpecific2018']]
        int_label_converter_df = pd.DataFrame(enumerate(name_df['label:DohertySpecific2018'].unique()),columns=['int_label','label:DohertySpecific2018'])
        int_label_converter_dict = dict(enumerate(name_df['label:DohertySpecific2018'].unique()))
        with open('capture24/int_label_converter_df.json','w') as f:
            json.dump(int_label_converter_dict,f)
        name_df = name_df.merge(int_label_converter_df)
        for fname in os.listdir('capture24'):
            if fname.endswith('.gz'): continue
            subj_id = fname.split('.')[0]
            if not subj_id.startswith('P') and not len(subj_id) == 4: continue # Skip metadata files
            print(f"converting {fname} to np")
            try: df = pd.read_csv(join('capture24',fname))
            except: set_trace()
            translated_df = df.merge(name_df)
            x = translated_df[['x','y','z']].to_numpy()
            y = translated_df['int_label'].to_numpy()
            np.save(join(np_dir,f'{subj_id}.npy'),x)
            np.save(join(np_dir,f'{subj_id}_labels.npy'),y)

    elif sys.argv[1] == 'HHAR':
        data_dir = 'Activity recognition exp'
        np_dir = 'hhar/np_data'
        print("\n#####Preprocessing HHAR#####\n")
        if not os.path.isdir(np_dir):
            os.makedirs(np_dir)

        pandaload = lambda path: pd.read_csv(join(data_dir,'Phones_accelerometer.csv')).set_index('Creation_Time').drop(['Index','Arrival_Time','Model','Device'],axis=1).dropna()
        print('loading dataframes\n')
        phone_acc_df = pandaload('Phones_accelerometer.csv')
        phone_gyro_df = pandaload('Phones_gyroscope.csv')
        watch_acc_df = pandaload('Watch_accelerometer.csv')
        watch_gyro_df = pandaload('Watch_gyroscope.csv')
        activities_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown']
        user_list = list('abcdefghi')

        for user_letter_name in user_list:
            print('processing user', user_letter_name)
            user_phone_acc = phone_acc_df.loc[phone_acc_df.User==user_letter_name]
            user_phone_gyro = phone_gyro_df.loc[phone_gyro_df.User==user_letter_name]
            user_watch_acc = watch_acc_df.loc[watch_acc_df.User==user_letter_name]
            user_watch_gyro = watch_acc_df.loc[watch_gyro_df.User==user_letter_name]
            assert all([user_watch_gyro.shape==d.shape for d in (user_phone_acc,user_phone_gyro,user_watch_acc)])
            comb_phone = user_phone_acc.join(user_phone_gyro,how='outer',lsuffix='_acc',rsuffix='_gyro')
            comb_watch = user_watch_acc.join(user_watch_gyro,how='outer',lsuffix='_acc',rsuffix='_gyro')
            #if not (comb_watch.gt_acc == comb_watch.gt_gyro).all(): set_trace()
            #if not (comb_phone.gt_acc == comb_phone.gt_gyro).all(): set_trace()
            comb = comb_phone.join(comb_watch,how='outer',lsuffix='_phone',rsuffix='_watch')
            duplicate_rows = [x for x,count in Counter(comb.index).items() if count > 1]
            if len(duplicate_rows) > 10: set_trace()
            elif len(duplicate_rows) > 0:
                print( f"removing {len(duplicate_rows)} duplicate rows")
                comb = comb.drop(duplicate_rows)
            if not (comb.gt_acc_phone == comb.gt_acc_watch).all(): set_trace()
            user_X_array = comb.drop([c for c in comb.columns if 'User' in c or 'gt' in c],axis=1).to_numpy()
            user_y_array = np.array([activities_list.index(a) for a in comb['gt_acc_phone']])
            save_path = join(np_dir,f"{user_list.index(user_letter_name)+1}.npy")
            label_save_path = join(np_dir,f"{user_list.index(user_letter_name)+1}_labels.npy")
            np.save(save_path,user_X_array,allow_pickle=False)
            np.save(label_save_path,user_y_array,allow_pickle=False)
            # Make smaller option for testing
            np.save(join(np_dir,f"0.npy"),user_X_array[::1000],allow_pickle=False)
            np.save(join(np_dir,f"0_labels.npy"),user_y_array[::1000],allow_pickle=False)

    else: print('\nIncorrect or no dataset specified\n')
