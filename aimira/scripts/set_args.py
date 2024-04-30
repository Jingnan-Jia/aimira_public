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
    parser.add_argument('--total_folds', help='4-fold training', type=int, default=4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4)
    parser.add_argument('--loss', help='loss', type=str, default='mse')
    parser.add_argument('--target', help='target', type=str, default='ifm')  # ifm-bme-tsy-syn

    # others
    parser.add_argument(
        '--outfile', help='output file when running by script instead of pycharm', type=str)
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
    # if args.target == 'IFM':
    #     args.input_mode = ''
    # args.input_mode = ''
    return args


if __name__ == "__main__":
    get_args()
