import sys
import sys
sys.path.append("/home/jjia/data/AIMIRA")
from filelock import FileLock
import shutil
import random
import mlflow
from mlflow import log_metric, log_metrics, log_param, log_params
from mlflow.tracking import MlflowClient
from modules.set_args import get_args
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sklearn
from torchsummary import summary
import torchio as tio
import warnings
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.simplefilter('ignore')
import cv2
cv2.setNumThreads(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = get_args()

def try_func(func):
    def _try_fun(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err:
            print(err, file=sys.stderr)
            pass
    return _try_fun

log_metric = try_func(log_metric)
log_metrics = try_func(log_metrics)

class scratch_nn(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=6, out_channels=9, kernel_size=3, padding='same', groups=3)
        self.conv2 = nn.Conv3d(in_channels=9, out_channels=9, kernel_size=3, padding='same', groups=3)
        self.conv3 = nn.Conv3d(in_channels=9, out_channels=18, kernel_size=3, padding='same', groups=3)
        self.conv4 = nn.Conv3d(in_channels=18, out_channels=18, kernel_size=3, padding='same', groups=3)
        self.conv5 = nn.Conv3d(in_channels=18, out_channels=18, kernel_size=3, padding='same', groups=3)
        self.conv6 = nn.Conv3d(in_channels=18, out_channels=36, kernel_size=3, padding='same', groups=3)
        self.conv1_bn = nn.BatchNorm3d(9)
        self.conv2_bn = nn.BatchNorm3d(9)
        self.conv3_bn = nn.BatchNorm3d(18)
        self.conv4_bn = nn.BatchNorm3d(18)
        self.conv5_bn = nn.BatchNorm3d(18)
        self.conv6_bn = nn.BatchNorm3d(36)
        #self.softmax = nn.Softmax(dim = 1)
        self.fc1 = nn.Linear(9216, num_classes)

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.dropout(x, 0.3)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.dropout(x, 0.3)

        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.dropout(x, 0.3)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.dropout(x, 0.3)

        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.dropout(x, 0.3)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.dropout(x, 0.4)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x


class RAMRISDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data.astype('float32'), label


def data_prepare():
    # The code snippet you provided is not complete and seems to be missing the actual code. It looks
    # like you have a variable `current_file_path` but it is not being assigned any value or used in
    # any way. If you provide more context or complete the code snippet, I can help you understand
    # what it is doing.
    # The code snippet you provided is not complete and seems to be missing the actual code that
    # performs an action. The variable `current_file_path` is declared but not assigned a value or
    # used in any way. If you provide more context or complete the code snippet, I can help explain
    # what it does.
    current_file_path = Path(__file__).resolve()
    data_dir = current_file_path.parent
    # data_dir = parent_path / Path('Treatment_response_data')
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    # data_dir = Path("/exports/lkeb-hpc-data/Aimira/data")
    wrist_ramris_path_train_1 = data_dir / Path('wrist+MCP+MTP(cor+tra).npy')
    
    
    
    
    # label_path = data_dir / Path('label_TRT_3.npy')
    # RAMRIS = np.load(wrist_ramris_path_train_1)
    # label = np.load(label_path)
    
    
    
    
    
    just_wrist = np.array(np.load(data_dir / Path('just_wrist.npy')))
    just_MCP = np.array(np.load(data_dir / Path('just_MCP.npy')))
    just_MTP = np.array(np.load(data_dir / Path('just_MTP.npy')))
    just_wrist_MCP = np.array(np.load(data_dir / Path('just_wrist_MCP.npy')))
    just_wrist_MTP = np.array(np.load(data_dir / Path('just_wrist_MTP.npy')))
    just_MCP_MTP = np.array(np.load(data_dir / Path('just_MCP_MTP.npy')))
    
    # baseline_score = np.array(np.load(data_dir / Path('FirstTP_TRT_SUM_wrist_MCP_MTP.npy')))
    # just_axial = np.array(np.load(data_dir / Path('just_axial.npy')))
    # just_coronal = np.array(np.load(data_dir / Path('just_coronal.npy')))
    # just_wrist_MCP_MTP_axial = np.array(np.load(data_dir / Path('just_wrist_MCP_MTP(axial).npy')))
    # just_wrist_MCP_MTP_coronal = np.array(np.load(data_dir / Path('just_wrist_MCP_MTP(coronal).npy')))
    
    label_path = data_dir / Path('label_TRT_3.npy')
    # patient = np.load('/exports/lkeb-hpc/thassanzadehkoohi/final_treatment_prediction_data/exclude_outlier/patient_list_num.npy')
    # new_CM = []
    # label_new = []
    # outlier = [40, 138, 149, 287, 410, 416, 454]
    RAMRIS = np.load(wrist_ramris_path_train_1)
    label = np.load(label_path)

    # for i in range(len(patient)):
    #     if patient[i] not in outlier:
    #         new_CM.append(RAMRIS[i])
    #         label_new.append(label[i])
    #
    # RAMRIS =  new_CM
    # label = label_new
    RAMRIS = np.array(RAMRIS)
    label = np.array(label)
    # RAMRIS = np.expand_dims(RAMRIS, axis = -1)
    # RAMRIS = RAMRIS.transpose(0, 4, 2, 3, 1)
    # just_wrist = just_wrist.transpose(0, 4, 2, 3, 1)
    # just_MCP = just_MCP.transpose(0, 4, 2, 3, 1)
    # just_MTP = just_MTP.transpose(0, 4, 2, 3, 1)
    # just_wrist_MCP = just_wrist_MCP.transpose(0, 4, 2, 3, 1)
    # just_wrist_MTP = just_wrist_MTP.transpose(0, 4, 2, 3, 1)
    # just_MCP_MTP = just_MCP_MTP.transpose(0, 4, 2, 3, 1)
    # just_axial = just_axial.transpose(0, 4, 2, 3, 1)
    # just_coronal = just_coronal.transpose(0, 4, 2, 3, 1)
    # just_wrist_MCP_MTP_axial = just_wrist_MCP_MTP_axial.transpose(0, 4, 2, 3, 1)
    # just_wrist_MCP_MTP_coronal = just_wrist_MCP_MTP_coronal.transpose(0, 4, 2, 3, 1)

    all_data_dt = {'RAMRIS':RAMRIS,
                   'label':label,
                   'just_wrist':just_wrist,
                   'just_MCP':just_MCP, 
                   'just_MTP':just_MTP,
                   'just_wrist_MCP':just_wrist_MCP,
                   'just_wrist_MTP':just_wrist_MTP,
                   'just_MCP_MTP':just_MCP_MTP,
                #    'baseline_score':baseline_score,
                #    'just_axial':just_axial,
                #    'just_coronal':just_coronal,
                #    'just_wrist_MCP_MTP_axial':just_wrist_MCP_MTP_axial,
                #    'just_wrist_MCP_MTP_coronal':just_wrist_MCP_MTP_coronal,
                   
                   }
    all_data_dt = {k:v.transpose(0, 4, 2, 3, 1) if k not in ['label', 'baseline_score'] else v for k,v in all_data_dt.items() }
    return all_data_dt


def train_step(model, optimizer, criterion, train_loader, test_dataloader, num_classes, random_label=False):
    model.train()
    avg_loss = []
    running_loss = []
    running_loss_val = []
    for subjects_batch in tqdm(train_loader):
        # print('i')
        x = subjects_batch['t1'][tio.DATA]
        y = subjects_batch['label']
        x = x.to(device)
        y = y.type(torch.LongTensor).to(device)
        y_pred = model(x=x)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss.append(loss.item())
        ####################train loss and val loss

        # running_loss += loss.item()
        # running_loss = loss.item()
        # print(f' loss: {running_loss :.3f}')
        running_loss.append(loss.item())


    for z, w in test_dataloader:
        z = z.to(device)
        w = w.type(torch.LongTensor).to(device)
        model.eval()
        y_pred_val = model(x=z)
        loss_val = criterion(y_pred_val, w)
        # running_loss_val += loss_val.item()
        running_loss_val.append(loss_val.item())
        # running_loss_val = loss_val.item()


    #print(f' loss: {np.mean(running_loss) :.3f} ----------------> loss_val: {np.mean(running_loss_val) :.3f}')

    return sum(avg_loss) / len(avg_loss)

def predict_conf(model, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    conf_before_max = []
    conf_after_max = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x=x)
            probs = torch.nn.functional.softmax(pred, dim=1)
            conf_before_max.append(probs)
            # print('probs=', probs )
            conf, classes = torch.max(probs, 1)
            y_pred = torch.argmax(pred, dim=1)
            conf_after_max.append(conf)
            # print('conf=', conf)
            # print('class =', classes)
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
            # total_conf =
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), conf_before_max, conf_after_max


def predict(model, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x=x)
            y_pred = torch.argmax(pred, dim=1)
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def aug(dataset):
    # data shape is (175,2). data[i][0] is X_train and data[i][1] is y_train
    # https://torchio.readthedocs.io/transforms/transforms.html#torchio.transforms.Transform
    subject_list = []
    for i in range (len(dataset)):
        tensor_4d = dataset[i][0]
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=tensor_4d),
            label=(dataset[i][1]),
        )
        subject_list.append(subject)
    #######rescale
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))
    #######spatial
    spatial = tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    },
        p=0.75,
    )
    #######flip
    flip = tio.RandomFlip(axes=(0,), flip_probability = 0.6)
    #######intensity
    intensity = tio.OneOf({
      tio.RandomNoise(mean = 0, std = (0, 0.25)),
      tio.RandomMotion(degrees = 10, translation= 10, num_transforms=2, image_interpolation='linear'),
      tio.RandomBlur(std = (0, 2)),
      tio.RandomGamma(log_gamma= (-0.2, 0.2))})
    #############
    transforms = [spatial]
    transforms = tio.Compose(transforms)
    transforms_1 = [rescale]
    transforms_1 = tio.Compose(transforms_1)
    transforms_2 = [flip]
    transforms_2 = tio.Compose(transforms_2)
    transforms_3 = [intensity]
    transforms_3 = tio.Compose(transforms_3)
    com_1 = [flip, intensity]
    com_2 = [spatial, flip]
    com_3 = [spatial,rescale,flip]
    com_4 = [rescale,flip,intensity]
    com_5 = [spatial, flip,intensity]
    transforms_4 = tio.Compose(com_1)
    transforms_5 = tio.Compose(com_2)
    transforms_6 = tio.Compose(com_3)
    transforms_7 = tio.Compose(com_4)
    transforms_8 = tio.Compose(com_5)

    return subject_list, transforms, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, transforms_6, transforms_7, transforms_8


# def train(j, i, model, dataset, val_set, val_set_just_wrist , val_set_just_MCP, val_set_just_MTP, val_set_just_wrist_MCP, val_set_just_wrist_MTP, val_set_just_MCP_MTP, baseline, val_set_just_axial, val_set_just_coronal, val_set_just_wrist_MCP_MTP_axial, val_set_just_wrist_MCP_MTP_coronal, lr=0.01, num_epoch=200):
def train(j, i, model, dataset_split_dt):
    lr = 0.01
    weight_dec = 0.0001
    max_accuracy = 0
    max_f1 = 0
    max_accuracy_fixed = 0
    max_f1_fixed = 0
    subject_list, transform, transform_1, transform_2, transform_3, transform_4, transform_5, transform_6, transform_7, transform_8  = aug(dataset_split_dt['train_set'])  # augment train dataset
    subjects_dataset = tio.SubjectsDataset(subject_list)
    subjects_dataset_0 = tio.SubjectsDataset(subject_list, transform=transform)
    subjects_dataset_1 = tio.SubjectsDataset(subject_list, transform=transform_1)
    subjects_dataset_2 = tio.SubjectsDataset(subject_list, transform=transform_2)
    subjects_dataset_3 = tio.SubjectsDataset(subject_list, transform=transform_3)
    subjects_dataset_4 = tio.SubjectsDataset(subject_list, transform=transform_4)
    subjects_dataset_5 = tio.SubjectsDataset(subject_list, transform=transform_5)
    subjects_dataset_6 = tio.SubjectsDataset(subject_list, transform=transform_6)
    subjects_dataset_7 = tio.SubjectsDataset(subject_list, transform=transform_7)
    subjects_dataset_8 = tio.SubjectsDataset(subject_list, transform=transform_8)

    subjects_dataset = np.concatenate((subjects_dataset, subjects_dataset_0, subjects_dataset_1, subjects_dataset_2, subjects_dataset_3, subjects_dataset_4, subjects_dataset_5, subjects_dataset_6, subjects_dataset_7, subjects_dataset_8), axis =0)

    print('subjects_dataset=', np.shape(subjects_dataset))
    train_dataloader = DataLoader(subjects_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset_split_dt['test_set'], batch_size=1, shuffle=False, num_workers=4)
    test_fixed_dataloader = DataLoader(dataset_split_dt['test_set_fixed'], batch_size=1, shuffle=False, num_workers=4)

    
    test_just_wrist_dataloader = DataLoader(dataset_split_dt['test_just_wrist'], batch_size=1, shuffle=False, num_workers=4)
    test_just_MCP_dataloader = DataLoader(dataset_split_dt['test_just_MCP'], batch_size=1, shuffle=False, num_workers=4)
    test_just_MTP_dataloader = DataLoader(dataset_split_dt['test_just_MTP'], batch_size=1, shuffle=False, num_workers=4)
    test_just_wrist_MCP_dataloader = DataLoader(dataset_split_dt['test_just_wrist_MCP'], batch_size=1, shuffle=False, num_workers=4)
    test_just_wrist_MTP_dataloader = DataLoader(dataset_split_dt['test_just_wrist_MTP'], batch_size=1, shuffle=False, num_workers=4)
    test_just_MCP_MTP_dataloader = DataLoader(dataset_split_dt['test_just_MCP_MTP'], batch_size=1, shuffle=False, num_workers=4)
    # test_just_axial_dataloader = DataLoader(val_set_just_axial, batch_size=1, shuffle=False, num_workers=4)
    # test_just_coronal_dataloader = DataLoader(val_set_just_coronal, batch_size=1, shuffle=False, num_workers=4)
    # test_just_wrist_MCP_MTP_axial_dataloader = DataLoader(val_set_just_wrist_MCP_MTP_axial, batch_size=1, shuffle=False, num_workers=4)
    # test_just_wrist_MCP_MTP_coronal_dataloader = DataLoader(val_set_just_wrist_MCP_MTP_coronal, batch_size=1, shuffle=False, num_workers=4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_dec)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_dec)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    num_epoch = 50
    for epoch in range(num_epoch):
        tarin_loss = train_step(model, optimizer, criterion, train_dataloader, test_dataloader, num_classes=2)  # finish one epoch
        log_metric('tarin_loss', tarin_loss, epoch)

        G, P = predict(model, test_dataloader)
        accuracy = accuracy_score(G, P)
        auc1 = sklearn.metrics.roc_auc_score(G, P)
        
        log_metric('acc', accuracy, epoch)
        log_metric('auc', auc1, epoch)

        G_fixed, P_fixed = predict(model, test_fixed_dataloader)
        accuracy_fixed = accuracy_score(G_fixed, P_fixed)
        auc1_fixed = sklearn.metrics.roc_auc_score(G_fixed, P_fixed)
        
        log_metric('acc_fixed', accuracy_fixed, epoch)
        log_metric('auc_fixed', auc1_fixed, epoch)

        if accuracy > max_accuracy:
            print('accuracy=', accuracy)
            print('auc=', auc1)
            print('groundtruth =', G)
            print('prediction =', P)
            max_accuracy = accuracy
            max_auc = auc1
            max_accuracy_fixed = accuracy_fixed
            max_auc_fixed = auc1_fixed
            # torch.save(model.state_dict(), '/exports/lkeb-hpc/thassanzadehkoohi/Treatment_response_result/TP_wrist_MCP_MTP_3_TC/model_CM_joint_old_'+str(j)+'_fold='+str(i)+'_'+str(max_accuracy) +'___'+ str(max_auc)+'.pth')
            # torch.save(model.state_dict(), '/exports/lkeb-hpc/thassanzadehkoohi/Treatment_response_result/TP_wrist_MCP_MTP_3_TC/model_CM_joint_old_'+str(j)+'_fold='+str(i)+'_'+str(max_accuracy) +'___'+ str(max_auc)+'.pt')
            G, P, conf_before_max, conf_after_max = predict_conf(model, test_dataloader)
            print('conf_before_max =', conf_before_max)
            print('conf_before_max =', conf_after_max)

            # print('baseline scores =', np.array(dataset_split_dt['baseline']))

            G, P = predict(model, test_just_wrist_dataloader)
            print('just wrist accuracy =', accuracy_score(G, P))
            print('just wrist AUC =', sklearn.metrics.roc_auc_score(G, P))

            G, P = predict(model, test_just_MCP_dataloader)
            print('just MCP accuracy =', accuracy_score(G, P))
            print('just MCP AUC =', sklearn.metrics.roc_auc_score(G, P))

            G, P = predict(model, test_just_MTP_dataloader)
            print('just MTP accuracy =', accuracy_score(G, P))
            print('just MTP AUC =', sklearn.metrics.roc_auc_score(G, P))

            G, P = predict(model, test_just_wrist_MCP_dataloader)
            print('just wrist_MCP accuracy =', accuracy_score(G, P))
            print('just wrist_MCP AUC =', sklearn.metrics.roc_auc_score(G, P))

            G, P = predict(model, test_just_wrist_MTP_dataloader)
            print('just wrist_MTP accuracy =', accuracy_score(G, P))
            print('just wrist_MTP AUC =', sklearn.metrics.roc_auc_score(G, P))

            G, P = predict(model, test_just_MCP_MTP_dataloader)
            print('just MCP_MTP accuracy =', accuracy_score(G, P))
            print('just MCP_MTP AUC =', sklearn.metrics.roc_auc_score(G, P))

            # G, P = predict(model, test_just_axial_dataloader)
            # print('just axial accuracy =', accuracy_score(G, P))
            # print('just axial AUC =', sklearn.metrics.roc_auc_score(G, P))

            # G, P = predict(model, test_just_coronal_dataloader)
            # print('just coronal accuracy =', accuracy_score(G, P))
            # print('just coronal AUC =', sklearn.metrics.roc_auc_score(G, P))

            # G, P = predict(model, test_just_wrist_MCP_MTP_axial_dataloader)
            # print('just wrist MCp MTP axial accuracy =', accuracy_score(G, P))
            # print('just wrist MCp MTP axial AUC =', sklearn.metrics.roc_auc_score(G, P))

            # G, P = predict(model, test_just_wrist_MCP_MTP_coronal_dataloader)
            # print('just wrist MCp MTPcoronal accuracy =', accuracy_score(G, P))
            # print('just wrist MCp MTP coronal AUC =', sklearn.metrics.roc_auc_score(G, P))

    print('max accuracy in this fold:%s and max auc in this fold: %s' %(max_accuracy ,max_auc))
    log_param(f'acc_repeat_{j}_fold_{i}', max_accuracy)
    log_param(f'auc_repeat_{j}_fold_{i}', max_auc)
    log_param(f'acc_repeat_{j}_fold_{i}', max_accuracy_fixed)
    log_param(f'auc_repeat_{j}_fold_{i}', max_auc_fixed)
    return model

def get_df_id(record_file: str):
    """Get the current experiment ID. It equals to the latest experiment ID + 1.

    Args:
        record_file: A file to record experiments details (super-parameters and metrics).

    Returns:
        dataframe and new_id

    Examples:
        :func:`lung_function.modules.tool.record_1st`

    """
    if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
        new_id = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(record_file)  # read the record file,
        last_id = df['ID'].to_list()[-1]  # find the last ID
        new_id = int(last_id) + 1
    return df, new_id

def record_1st(record_file) -> int:
    Path(record_file).parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(record_file + ".lock")  # lock the file, avoid other processes write other things
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a'):
            df, new_id = get_df_id(record_file)
            idatime = {'ID': new_id}
            if len(df) == 0:  # empty file
                df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
            else:
                index = df.index.to_list()[-1]  # last index
                for key, value in idatime.items():  # write new line
                    df.at[index + 1, key] = value  #

            df.to_csv(record_file, index=False)
            shutil.copy(record_file, record_file + "_cp")

    return new_id
   
def main():
    SEED = 4
    # set_determinism(SEED)  # set seed for this run
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment('AIMIRA')
    
    RECORD_FPATH = f"{Path(__file__).absolute().parent}/results/record.log"
    # write super parameters from set_args.py to record file.
    id = record_1st(RECORD_FPATH)
 
    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        args.id = id  # do not need to pass id seperately to the latter function
        tmp_args_dt = vars(args)
        log_params(tmp_args_dt)
                
        print(device)
        # path = '/exports/lkeb-hpc-data/Aimira/Treatment_response_data/wrist+MCP+MTP/split_103/'
        # path = Path(__file__).resolve().parent.parent / 'data/split_103'
        #path_results = '/exports/lkeb-hpc/thassanzadehkoohi/final_treatment_prediction_result/wrist/CM/'
        # RAMRIS, label, just_wrist, just_MCP, just_MTP, just_wrist_MCP, just_wrist_MTP, just_MCP_MTP, baseline_score, just_axial, just_coronal, just_wrist_MCP_MTP_axial, just_wrist_MCP_MTP_coronal = data_prepare()
        all_data_dt = data_prepare()

        for j in range (1):  # 5 repetitions
            # kf = KFold(n_splits=10, shuffle=True)
            # counter = 0
            # for x_train, x_test in kf.split(RAMRIS):
            patient_id_ls = np.arange(args.NB_patients-10)
            test_id_ls = np.arange(args.NB_patients-10, args.NB_patients)
            
            kf = KFold(n_splits=args.total_folds, shuffle=True, random_state=711)  # for future reproduction
            kf_list = list(kf.split(patient_id_ls))
           
            
            for i in range(args.total_folds):  # 10-fold cross-validation
                tr_pt_idx, vd_pt_idx = kf_list[i - 1]
                x_train = patient_id_ls[tr_pt_idx]
                x_test = patient_id_ls[vd_pt_idx]
                x_test_fixed = test_id_ls
                print(f"x_train: {x_train}")  
                print(f"x_test: {x_test}")  
                print(f"x_test_fixed: {x_test_fixed}")
                # print(x_train)
                # print(x_test)
                print(all_data_dt['RAMRIS'][x_train].shape)
                print(all_data_dt['RAMRIS'][x_test].shape)
                print(all_data_dt['RAMRIS'][x_test_fixed].shape)
                data_split_dt = {}
                for k,v in all_data_dt.items():
                    if k == 'RAMRIS':
                        data_split_dt['train_set'] = v[x_train]
                        data_split_dt['test_set'] = v[x_test]
                        data_split_dt['test_set_fixed'] = v[x_test_fixed]
                    elif k == 'baseline_score':
                        data_split_dt[k] = v[x_test]
                        
                    elif 'just' in k:
                        data_split_dt['test_'+k] = v[x_test]
                    elif k == 'label':
                        data_split_dt[k] = v
                    else:
                        raise Exception('Unknown')
                        
                dataset_split_dt = {}
                for k,v in data_split_dt.items():
                    if 'train_set'==k:
                        dataset_split_dt[k] = RAMRISDataset(v, data_split_dt['label'][x_train])
                    elif 'test_set_fixed'==k:
                        dataset_split_dt[k] = RAMRISDataset(v, data_split_dt['label'][x_test_fixed])
                    else:
                        dataset_split_dt[k] = RAMRISDataset(v, data_split_dt['label'][x_test])  # all the others are testing 
                    
                # dataset_split_dt = {k: RAMRISDataset(v, data_split_dt['label'][x_train]) if 'train_set'==k else RAMRISDataset(v, data_split_dt['label'][x_test]) for k,v in data_split_dt.items() }
          
                model = scratch_nn(num_classes=2)
                model = model.to(device)
                summary(model, (6, 128, 128, 14))
                model = train(j, i, model=model, dataset_split_dt=dataset_split_dt)
                #torch.save(model.state_dict(), path_results + 'model_CM_v2_'+str(j) +'_'+ str(i)+'pth')
        
if __name__ == '__main__':
    main()
