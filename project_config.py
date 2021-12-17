import sys


class HAR_Dataset_Container():
    def __init__(self,code_name,dataset_dir_name,possible_subj_ids,num_channels,num_classes,action_name_dict):
        self.code_name = code_name
        self.dataset_dir_name = dataset_dir_name
        self.possible_subj_ids = possible_subj_ids
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.action_name_dict = action_name_dict


# PAMAP
pamap_ids = [str(x) for x in range(101,110)]
pamap_action_name_dict = {1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'Nordic walking',9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing soccer',24:'rope jumping'}
PAMAP_INFO = HAR_Dataset_Container(
            code_name = 'PAMAP',
            dataset_dir_name = 'PAMAP2_Dataset',
            possible_subj_ids = pamap_ids,
            num_channels = 39,
            num_classes = 12,
            action_name_dict = pamap_action_name_dict)

# UCI
def two_digitify(x): return '0' + str(x) if len(str(x))==1 else str(x)
uci_ids = [two_digitify(x) for x in range(1,30)]
uci_action_name_dict = {1:'walking',2:'walking upstairs',3:'walking downstairs',4:'sitting',5:'standing',6:'lying',7:'stand_to_sit',9:'sit_to_stand',10:'sit_to_lit',11:'lie_to_sit',12:'stand_to_lie',13:'lie_to_stand'}
UCI_INFO = HAR_Dataset_Container(
            code_name = 'UCI',
            dataset_dir_name = 'UCI2',
            possible_subj_ids = uci_ids,
            num_channels = 6,
            num_classes = 6,
            action_name_dict = uci_action_name_dict)

# WISDM-v1
wisdmv1_ids = [str(x) for x in range(1,37)] #Paper says 29 users but ids go up to 36
activities_list = ['Jogging','Walking','Upstairs','Downstairs','Standing','Sitting']
wisdmv1_action_name_dict = dict(zip(range(len(activities_list)),activities_list))
WISDMv1_INFO = HAR_Dataset_Container(
            code_name = 'WISDM-v1',
            dataset_dir_name = 'wisdm_v1',
            possible_subj_ids = wisdmv1_ids,
            num_channels = 3,
            num_classes = 6,
            action_name_dict = wisdmv1_action_name_dict)

# WISDM-watch
wisdmwatch_ids = [str(x) for x in range(1600,1651)]
with open('datasets/wisdm-dataset/activity_key.txt') as f: r=f.readlines()
activities_list = [x.split(' = ')[0] for x in r if ' = ' in x]
wisdmwatch_action_name_dict = dict(zip(range(len(activities_list)),activities_list))
WISDMwatch_INFO = HAR_Dataset_Container(
            code_name = 'WISDM-watch',
            dataset_dir_name = 'wisdm-dataset',
            possible_subj_ids = wisdmwatch_ids,
            num_channels = 12,
            num_classes = 17,
            action_name_dict = wisdmwatch_action_name_dict)

# REALDISP
realdisp_ids = [str(x) for x in range(1,18)]
activities_list = ['Walking','Jogging','Running','Jump up','Jump front & back','Jump sideways','Jump leg/arms open/closed','Jump rope','Trunk twist','Trunk twist','Waist bends forward','Waist rotation','Waist bends','Reach heels backwards','Lateral bend','Lateral bend with arm up','Repetitive forward stretching','Upper trunk and lower body opposite twist','Lateral elevation of arms','Frontal elevation of arms','Frontal hand claps','Frontal crossing of arms','Shoulders high-amplitude rotation','Shoulders low-amplitude rotation','Arms inner rotation','Knees','Heels','Knees bending','Knees','Rotation on the knees','Rowing','Elliptical bike','Cycling']
realdisp_action_name_dict = {i+1:act for i,act in enumerate(activities_list)}
REALDISP_INFO = HAR_Dataset_Container(
            code_name = 'REALDISP',
            dataset_dir_name = 'realdisp',
            possible_subj_ids = realdisp_ids,
            num_channels = 117,
            num_classes = 33,
            action_name_dict = realdisp_action_name_dict)

DSET_OBJECTS = [PAMAP_INFO, UCI_INFO, WISDMv1_INFO, WISDMwatch_INFO,REALDISP_INFO]


def get_dataset_info_object(dset_name):
    dsets_by_that_name = [d for d in DSET_OBJECTS if d.code_name == dset_name]
    if len(dsets_by_that_name)==0: print(f"{dset_name} is not a recognized dataset"); sys.exit()
    assert len(dsets_by_that_name)==1
    return dsets_by_that_name[0]
