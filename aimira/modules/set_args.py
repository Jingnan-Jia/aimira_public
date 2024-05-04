# -*- coding: utf-8 -*-
# @Time    : 3/6/21 9:58 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-

import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args(jupyter=False):
    parser = argparse.ArgumentParser(description="SSc score prediction.")
    parser.add_argument('--NB_patients', help='number of patients', type=int, default=115)
    parser.add_argument('--total_folds', help='4-fold training', type=int, default=2)
    parser.add_argument('--kfold_seed', help='4-fold kfold_seed', type=int, default=711)


    parser.add_argument('--batch_size', help='batch size', type=int, default=4)
    parser.add_argument('--loss', help='loss', type=str, default='mse')
    parser.add_argument('--target', help='target', choices=['IFM', 'IFM_change'], type=str, default='IFM')  # ifm-bme-tsy-syn -BME-SYN-TSY
    parser.add_argument('--input_position_code', help='input_position_code', choices=['WR', 'MC', 'MT'], type=str, default='WR')  # ifm-bme-tsy-syn
    parser.add_argument('--net', help='network name', choices=['vgg11_3d'], type=str, default='vgg11_3d')  # ifm-bme-tsy-syn
    parser.add_argument('--view_fution', help='method for view fution', choices=['input_concatenation','after_first_conv', 'before_last_conv'], type=str, default='input_concatenation')  # ifm-bme-tsy-syn
    parser.add_argument('--nb_slices', help='nb_slices', type=int, default=7)  # ifm-bme-tsy-syn
    parser.add_argument('--workers', help='workers', type=int, default=4)  # ifm-bme-tsy-syn
    parser.add_argument('--weight_decay', help='weight_decay', type=float, default=1e-4)  # ifm-bme-tsy-syn
    parser.add_argument('--lr', help='lr', type=float, default=1e-4)  # ifm-bme-tsy-syn
    parser.add_argument('--pretrained_id', help='pretrained_id', type=str, default=None)  # ifm-bme-tsy-syn
    parser.add_argument('--mode', help='lr', type=str, default='infer')  # ifm-bme-tsy-syn
    parser.add_argument('--epochs', help='epochs', type=int, default=1)  # ifm-bme-tsy-syn
    parser.add_argument('--valid_period', help='valid_period', type=int, default=5)  # ifm-bme-tsy-syn
    # parser.add_argument('--lr', help='lr', type=float, default=1e-4)  # ifm-bme-tsy-syn
    # parser.add_argument('--lr', help='lr', type=float, default=1e-4)  # ifm-bme-tsy-syn
    parser.add_argument('--freeze', help='freeze layer', type=str, choices=['encoder', 'encoder_decoder'], default='encoder_decoder')  # ifm-bme-tsy-syn
    parser.add_argument('--extra_fc', help='extra_fc layer', type=bool, default=False)  # ifm-bme-tsy-syn


    # others
    parser.add_argument( '--outfile', help='output file when running by script', type=str)
    parser.add_argument('--hostname', help='hostname of the server', type=str)
    parser.add_argument('--remark', help='comments on this experiment',
                        type=str, default='None')
    parser.add_argument('--jobid', help='slurm job_id', type=int, default=0)
    # For jupyter notebooks
    if jupyter:
        parser.add_argument(
            "--f", help="a dummy argument to fool ipython", default="0")

        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    args.pretrained_id = f"esmira_SYN_TSY_BME__{args.input_position_code}_2dirc_fold0Sum.model"
    return args


if __name__ == "__main__":
    get_args()
