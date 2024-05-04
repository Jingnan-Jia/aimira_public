import sys

sys.path.append("../..")
import csv
import glob
import os
import threading
from typing import List

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import ndimage
import glob
import os
import seaborn as sns
# sns.set_theme(color_codes=True)
import scipy.stats as stats
import statsmodels
from statsmodels import api
import matplotlib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import cohen_kappa_score

from aimira.modules import my_bland

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def icc(label_fpath, pred_fpath, ignore_1st_column=False):
    icc_dict = {}

    label = pd.read_csv(label_fpath)
    pred = pd.read_csv(pred_fpath)
    if ignore_1st_column:
        pred = pred.iloc[: , 1:]
        label = label.iloc[: , 1:]
    if 'ID' == label.columns[0]:
        del label["ID"]
    if 'ID' == pred.columns[0]:
        del pred["ID"]

    original_columns = label.columns

    # ori_columns = list(label.columns)

    label['ID'] = np.arange(1, len(label) + 1)
    label['rater'] = 'label'

    pred['ID'] = np.arange(1, len(pred) + 1)
    pred['rater'] = 'pred'

    data = pd.concat([label, pred], axis=0)

    for column in original_columns:
        icc = pg.intraclass_corr(data=data, targets='ID', raters='rater', ratings=column).round(2)
        icc = icc.set_index("Type")
        icc = icc.loc['ICC2']['ICC']
        prefix = label_fpath.split("/")[-1].split("_")[0]
        icc_dict['icc_' + prefix + '_' + column] = icc

    return icc_dict

def metrics(pred_fpath, label_fpath, ignore_1st_column=False, xy_same_max=True):
    """
    ignore_1st_column: ignore the index column
    """
    
    r_dict, p_dict = {}, {}
    df_pred = pd.read_csv(pred_fpath)
    df_label = pd.read_csv(label_fpath)
    

    
    if ignore_1st_column:
        df_pred = df_pred.iloc[: , 1:]
        df_label = df_label.iloc[: , 1:]
    print('len_df_label', len(df_label))


    lower_y_ls, upper_y_ls = [], []
    lower_x_ls, upper_x_ls = [], []

    row_nb = 1
    col_nb = len(df_label.columns)
    height_fig = 4
    length_fig = height_fig * col_nb
    # if col_nb < 10:
    fig = plt.figure(figsize=(length_fig, height_fig))  # for bland-altman plot
    fig_2 = plt.figure(figsize=(length_fig, height_fig))  # for scatter plot
    # fig_3 = plt.figure(figsize=(length_fig, height_fig))  # for scatter plot with 95% CI
    # else:
    #     raise Exception(f"the columns number is greater than 10: {df_label.columns}")

    basename = os.path.dirname(pred_fpath)
    prefix = pred_fpath.split("/")[-1].split("_")[0]
    if  col_nb <= 11:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#808000']
    elif col_nb <= 19:
        colors = ['#7f0000', '#808000', '#3cb371', '#7f007f', '#008080', '#7f007f', '#ff0000', '#ff8c00', '#ffd700',
                  '#0000cd',
                  '#00ff7f', '#00ffff', '#adff2f', '#00bfff', '#ff00ff', '#f0e68c', '#dda0dd', '#ff1493', '#ffa07a',
                  ]
    else:
        raise Exception(f"the columns number is greater than 20: {df_label.columns}")

    print(df_pred.columns.tolist(),'yyaayy')

    
    for plot_id, column in enumerate(df_label.columns):
        if column=='pat_id':
            continue
        label = df_label[column].to_numpy().reshape(-1, 1)
        pred = df_pred[column].to_numpy().reshape(-1, 1)

        # bland-altman plot
        ax = fig.add_subplot(row_nb, col_nb, plot_id + 1)
        ax_2 = fig_2.add_subplot(row_nb, col_nb, plot_id + 1)
        # ax_3 = fig_3.add_subplot(row_nb, col_nb, plot_id + 1)
        
       

        # f, ax = plt.subplots(1, figsize=(8, 5))
        scatter_kwds = {'c': colors[plot_id], 'label': column, 's': 8}


        f = my_bland.mean_diff_plot(pred, label, ax=ax, 
        scatter_kwds=scatter_kwds,
                              bland_in_1_mean_std=None,
                              adap_markersize=False)
        f_2 = my_bland.mean_diff_plot(pred, label, ax=ax_2, sd_limit=0, 
        scatter_kwds=scatter_kwds,
                                bland_in_1_mean_std=None,
                                adap_markersize=False, 
                                ynotdiff=True, 
                                class_name=column)
        ax = sns.regplot(x=label, y=pred, color=scatter_kwds['c'], scatter_kws={'s': 8},
                         label = f"95% confidence interval")
        # ax.set_title(column, fontsize=15)
        # ax_2.set_title(column, fontsize=15)

        
        # plot linear regression line
        m, b = np.polyfit(label.flatten(), pred.flatten(), 1)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label.flatten(), pred.flatten())
        r_dict['r_' + prefix + '_' + column] = r_value
        p_dict['p_of_r' + prefix + '_' + column] = p_value

        x_reference = np.array([np.min(label), np.max(label)])
        print(column, 'linear regression m, b:', m, b)
        print(column, 'linear regression m, b, r^2:', slope, intercept, r_value ** 2)

        if p_value < 0.001:
            p_txt = 'p < 0.001'
        elif p_value < 0.01:
            p_txt = 'p < 0.01'
        elif p_value < 0.05:
            p_txt = 'p < 0.05'
        else:
            p_txt = f'p = {p_value:.2f}'
            
        ax_2.text(0.1, 0.8, f'R = {r_value: .2f}\n{p_txt}',
                  ha="left", fontsize='large', transform=ax_2.transAxes)

        min_xy = min(np.min(label), np.min(pred))
        max_xy = int(max(np.max(label), np.max(pred))) + 1

        
        ax_2.plot([min_xy*1.2, max_xy*1.2], [0, max_xy*1.2], '--', color = 'gray')
        
        ax_2.set_xlim(min_xy*1.1, max_xy*1.1)
        ax_2.set_ylim(min_xy*1.1, max_xy*1.1)
        
        # 生成自定义的刻度
        step = 2 if max_xy > 8 else 1
        ticks = list(range(0, max_xy, step))



        lower_y, upper_y = ax.get_ybound()  # set these plots as the same scale for comparison
        lower_x, upper_x = ax.get_xbound()
        lower_y_ls.append(lower_y)
        upper_y_ls.append(upper_y)
        lower_x_ls.append(lower_x)
        upper_x_ls.append(upper_x)

        diff = pred - label
        abs_diff = np.abs(diff)
        ave_mae = np.mean(abs_diff)
        std_mae = np.std(abs_diff)
        mean = np.mean(diff)
        std = np.std(diff)

        print(f"ave_mae for {column} is {ave_mae}")
        print(f"std_mae for {column} is {std_mae}")
        print(f"mean for {column} is {mean}")
        print(f"std for {column} is {std}")
        print("Finish plot of ", column)

    f.tight_layout()
    f.savefig(basename + '/' + prefix + '_bland_altman.png')
    plt.close(f)

    # f_2.suptitle(prefix.capitalize() + " Prediction Scatter Plot", fontsize=26)
    f_2.tight_layout()
    f_2.savefig(basename + '/' + prefix + '_scatter.png')
    plt.close(f_2)


    all_dt = {**r_dict, **p_dict}
    return all_dt

if __name__ == "__main__":
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/LKT2_16patients.csv"
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/results/models/1405_1404_1411_1410/16pats_pred.csv"
    # label_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/ground_truth_16patients.csv"

    pred_fpath = "/data1/jjia/lung_function/lung_function/scripts/results/experiments/202/traininfernoaug_pred.csv"
    label_fpath = "/data1/jjia/lung_function/lung_function/scripts/results/experiments/202/traininfernoaug_label.csv"

    metrics(pred_fpath, label_fpath)
    icc_value = icc(label_fpath, pred_fpath)
    print('icc:', icc_value)