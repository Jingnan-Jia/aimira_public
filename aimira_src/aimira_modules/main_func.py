import os  # for paths
# from dataset.datasets import CatDogDataset   # load the dataset
import torch.nn as nn   # used for bulid the neural networks
# from models.model import scratch_nn  # the model
from tqdm import tqdm  # just for visualization
from torchvision import transforms  # transformation
import copy  # not important
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from medutils.medutils import save_itk
# train is HIGHLY recommanded to be a seperate function, as well as the train step, so your code could be cleaner
# train:
# (1) hyper-parameters: lr, weight_dec, optimizer, criterion(loss function), batch_size
# (2) dataset ready
# (3) give a name for saving the weights
# (4) start train_step
# (5) inference on the validation set
# (6) save the weights


# train_step can keep unchanged for all the tasks if you use the same logic as mine:
# (1) Load model, optimizer, criterion, train_loader
# (2) Calculate, get the loss
# (3) Return the avg loss


# Predict is the train without train_step


def train_step(model, optimizer, criterion, train_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()
    avg_loss = []
    for x,y, path in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)
        # print(z)
        y_pred = model(x=x)
        print('label', y)
        print('pred', y_pred)
        print('-next----------')
        loss = criterion(y_pred, y)
        # next three lines are unchanged for all the tasks
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # end
        avg_loss.append(loss.item())
    return sum(avg_loss)/len(avg_loss)


    
def predict(model, test_loader, mode='valid', device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), mypath=None, save_results=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_paths = []
    # model.to(torch.device("cpu"))
    with torch.no_grad():
        for x,y,path in tqdm(test_loader):
            # path = [i.split('Treat')[-1][:4] for i in path]
            # save_itk(f"/exports/lkeb-hpc/jjia/project/project/aimira/jinanan_method/{path}_Yanli.mha", x[0].numpy(), [1,1,1], [1,1,1], dtype='float')

            x = x.to(device)
            y = y.to(device)
            pred = model(x=x)
            print('label', y)
            print('pred', pred)
            print('-next----------')
            y_pred = pred
            # y_pred = torch.argmax(pred, dim=1)
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
            total_paths += list(path)
            
    G = total_labels.numpy().flatten()
    P = total_preds.numpy().flatten()
    id = np.array(total_paths)
    if save_results:
        # 使用column_stack()函数将这三个数组拼接成一个二维数组
        id_G = np.column_stack((id, G))
        id_P = np.column_stack((id, P))

        id_G = pd.DataFrame(id_G, columns=['pat_id', 'score'])
        id_G.to_csv(mypath.save_label_fpath(mode), index=False)

        id_P = pd.DataFrame(id_P, columns=['pat_id', 'score'])
        id_P.to_csv(mypath.save_pred_fpath(mode), index=False)


    return G, P, id


def train(model, dataset, val_dataset, lr=0.001, num_epoch:int=100,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    lr = lr
    weight_dec = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_dec)
    criterion = nn.CrossEntropyLoss()
    batch_size = 32

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    max_accuracy = 0
    model = model.to(device)

    model_file_name = "./current_best_model.model"
    
    for epoch in range(1, num_epoch + 1):
        train_loss = train_step(model, optimizer, criterion, dataloader)
        if epoch % 50 == 0:
            print(f"Loss at epoch {epoch} is {train_loss}")
        G,P = predict(model, val_dataloader)
        accuracy = accuracy_score(G, P)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), model_file_name)
            print("saving best model with accuracy: ", accuracy)



def pretrained(model, model_file_name= "/exports/lkeb-hpc/jjia/project/project/aimira/aimira/data/SYN_TSY_BME__Wrist_2dirc_fold0Sum.model"):
    # load the weight
    if os.path.isfile(model_file_name):
        checkpoint = torch.load(model_file_name, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print(f"model loaded from {model_file_name}")
    else:
        raise Exception(f"no model weightes to load at {model_file_name}")
    return model


# def dataset_generator(data_dir = './kaggle/working/extracted/train'):
#     # define your transformation here, or you can also define it within the Dataset object
#     data_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((256, 256)),
#         transforms.ColorJitter(),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]
#     ])
#     val_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((224, 224)),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]
#     ])
#     train_dir = data_dir
#     train_img_list = os.listdir(train_dir)[:9999] + os.listdir(train_dir)[12500:22499] # 0-12499 cats, 12500-24999 dogs
#     val_img_list = os.listdir(train_dir)[10000:12499] + os.listdir(train_dir)[22499:]

#     train_dataset = CatDogDataset(train_dir, train_img_list, transform = data_transform)
#     val_dataset = CatDogDataset(train_dir, val_img_list, transform = val_transform)

#     return train_dataset, val_dataset


# def model_generator(in_channel:int=3, num_classes:int=2):
#     model = scratch_nn(in_channel=in_channel, num_classes=num_classes)
#     return model
