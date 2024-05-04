import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from main_func import dataset_generator, train, pretrained, predict
from aimira.modules.clip_model import ModelClip


# basic workflow of pytorch:
# 1. define dataset:
#       dataset = Dataset(...) torch.utils.data.Dataset object -- including the data and label -- find examples in the dataset.datasets.py
#       dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4) -- a iterator and decide the batch_size and shuffle, just a function to organize the dataset
# 2. define model:
#       model = scratch_nn()  # you can pass any attributes into it if there is a required attribute.
# 3. Train the model / OR load the weights
#       train(model, dataset, val_dataset, lr, num_epoch, num_classes)
# 4. Inference

def main_process(data_dir='./kaggle/working/extracted/train'):

    # Step. 1 get the dataset: (highly recommand a function to generate the dataset, to make the code clean)
    train_dataset, val_dataset = dataset_generator(data_dir=data_dir)

    # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
    model = ModelClip(group_num=2, group_cap=7, out_ch=1, width=2, dimension=2, extra_fc=False)

    model_fpath = "/exports/lkeb-hpc/jjia/project/project/aimira/aimira/data/esmira_SYN_TSY_BME__WR_2dirc_fold0Sum.model"
    ckpt = torch.load(model_fpath)
    model.load_state_dict(ckpt)
    print(f"model is loaded arom {model_fpath}")
    
    # Step. 3 Train the model /OR load the weights
    train(model=model, dataset=train_dataset, val_dataset=val_dataset, lr=0.0001, num_epoch=100, num_classes=2)

    # Step. 4 Load the weights and predict
    # model = pretrained(model=model)
    

    
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    G,P = predict(model, val_dataloader)
    print(classification_report(G,P))   


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_process(data_dir='./kaggle/working/extracted/train')