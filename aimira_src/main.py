import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from aimira_modules.main_func import train, pretrained, predict
# from models.model import scratch_nn
from aimira_modules.model import ModelClip
from aimira_generators.aimira_generator import AIMIRA_generator

# basic workflow of pytorch:
# 1. define dataset:
#       dataset = Dataset(...) torch.utils.data.Dataset object -- including the data and label -- find examples in the dataset.datasets.py
#       dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4) -- a iterator and decide the batch_size and shuffle, just a function to organize the dataset
# 2. define model:
#       model = scratch_nn()  # you can pass any attributes into it if there is a required attribute.
# 3. Train the model / OR load the weights
#       train(model, dataset, val_dataset, lr, num_epoch, num_classes)
# 4. Inference

def main_process(site):

    # Step. 1 get the dataset: (highly recommand a function to generate the dataset, to make the code clean)
    # train_dataset, val_dataset = dataset_generator(data_dir=data_dir)
    generator= AIMIRA_generator(data_root="/exports/lkeb-hpc/jjia/project/project/aimira/aimira/data/TRT_ori", 
                                target_category=['TRT'], 
                        target_site=[site], target_dirc=['TRA', 'COR'], target_reader=['Reader1', 'Reader2'], 
                        target_timepoint=['1'], task_mode='clip', score_sum=True,
                        working_dir="/exports/lkeb-hpc/jjia/project/project/aimira/aimira_src", print_flag=True, max_fold=2)
    
    train_dataset, val_dataset = generator.returner(fold_order=0, material='img',
                 monai=True, full_img=7, dimension=2,
                 contrast=False, data_balance=False,
                 path_flag=False)
    
    print('yes')
    # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
    model = ModelClip(group_num=2, group_cap=7, out_ch=1, width=2, dimension=2, extra_fc=False)

    # Step. 4 Load the weights and predict
    model = pretrained(model=model, 
                       model_file_name=f"/exports/lkeb-hpc/jjia/project/project/aimira/aimira/data/SYN_TSY_BME__{site}_2dirc_fold0Sum.model")
    model.to(device)

    # Step. 3 Train the model /OR load the weights
    # train(model=model, dataset=train_dataset, val_dataset=val_dataset, lr=0.0001, num_epoch=2)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    G,P = predict(model, val_dataloader)
    
    import matplotlib.pyplot as plt
    plt.scatter(G,P)
    plt.xlim(-2, 30)
    plt.ylim(-2, 30)
    plt.show()
    plt.savefig('scatter_plot_train.png')
    
    
    print(classification_report(G,P))   


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_process(site='Wrist')